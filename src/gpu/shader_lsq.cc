/*
 * Copyright (c) 2013 Mark D. Hill and David A. Wood
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Joel Hestness, Jason Power
 *
 */

#include "debug/ShaderLSQ.hh"
#include "gpu/shader_lsq.hh"

using namespace std;

ShaderLSQ::ShaderLSQ(Params *p)
    : MemObject(p), controlPort(name() + ".ctrl_port", this),
      writebackBlocked(false), cachePort(name() + ".cache_port", this),
      warpSize(p->warp_size), maxNumWarpsPerCore(p->warp_contexts),
      atomsPerSubline(p->atoms_per_subline),
      flushing(false), flushingPkt(NULL), forwardFlush(p->forward_flush),
      warpInstBufPoolSize(p->num_warp_inst_buffers), dispatchWarpInstBuf(NULL),
      perWarpInstructionQueues(p->warp_contexts),
      perWarpOutstandingAccesses(p->warp_contexts),
      overallLatencyCycles(p->latency), l1TagAccessCycles(p->l1_tag_cycles),
      tlb(p->data_tlb), sublineBytes(p->subline_bytes),
      nextAllowedInject(Cycles(0)), injectWidth(p->inject_width),
      mshrsFull(false), ejectWidth(p->eject_width), cacheLineAddrMaskBits(-1),
      lastWarpInstBufferChange(0), numActiveWarpInstBuffers(0),
      dispatchInstEvent(this), injectAccessesEvent(this),
      ejectAccessesEvent(this), commitInstEvent(this)
{
    // Create the lane ports based on the number threads per warp
    for (int i = 0; i < warpSize; i++) {
        lanePorts.push_back(
                new LanePort(csprintf("%s-lane-%d", name(), i), this, i));
    }

    for (int i = 0; i < maxNumWarpsPerCore; i++) {
        perWarpOutstandingAccesses[i] = 0;
    }

    warpInstBufPool = new WarpInstBuffer*[warpInstBufPoolSize];
    for (int i = 0; i < warpInstBufPoolSize; i++) {
        warpInstBufPool[i] = new WarpInstBuffer(warpSize, atomsPerSubline);
        availableWarpInstBufs.push(warpInstBufPool[i]);
    }

    // Set the delay cycles to model appropriate memory access latency
    // Functionally, this LSQ has 4 pipeline stages:
    //  1) Coalesce + address translations [Always 1 cycle given TLB hits]
    //  2) Translations complete, queued waiting to issue accesses [0+ cycles]
    //  3) L1 access [1 cycle given uncontended L1 hit]
    //  4) Warp instruction completion [1 cycle given no commit contention]
    // l1TagAccessCycles replicates L1 tag access time during which subsequent
    // instructions from the same warp cannot issue to the caches
    assert(l1TagAccessCycles <= (Cycles(overallLatencyCycles - 4)));
    completeCycles = Cycles(overallLatencyCycles - 4 - l1TagAccessCycles);
    assert(completeCycles > Cycles(0));

    // Set the number of bits to mask for cache line addresses
    cacheLineAddrMaskBits = log2(p->cache_line_size);
}

ShaderLSQ::~ShaderLSQ()
{
    for (int i = 0; i < warpSize; i++)
        delete lanePorts[i];
    for (int i = 0; i < warpInstBufPoolSize; i++)
        delete warpInstBufPool[i];
    delete [] warpInstBufPool;
}

BaseMasterPort &
ShaderLSQ::getMasterPort(const string &if_name, PortID idx)
{
    if (if_name == "cache_port") {
        return cachePort;
    } else {
        return MemObject::getMasterPort(if_name, idx);
    }
}

BaseSlavePort &
ShaderLSQ::getSlavePort(const string &if_name, PortID idx)
{
    if (if_name == "lane_port") {
        if (idx >= static_cast<PortID>(lanePorts.size())) {
            panic("RubyPort::getSlavePort: unknown index %d\n", idx);
        }

        return *lanePorts[idx];
    } else if (if_name == "control_port") {
        return controlPort;
    } else {
        // pass it along to our super class
        return MemObject::getSlavePort(if_name, idx);
    }
}

AddrRangeList
ShaderLSQ::LanePort::getAddrRanges() const
{
    // at the moment the assumption is that the master does not care
    AddrRangeList ranges;
    return ranges;
}

bool
ShaderLSQ::LanePort::recvTimingReq(PacketPtr pkt)
{
    return lsq->addLaneRequest(laneId, pkt);
}

Tick
ShaderLSQ::LanePort::recvAtomic(PacketPtr pkt)
{
    panic("ShaderLSQ::LanePort::recvAtomic() not implemented!\n");
    return 0;
}

void
ShaderLSQ::LanePort::recvFunctional(PacketPtr pkt)
{
    panic("ShaderLSQ::LanePort::recvFunctional() not implemented!\n");
}

void
ShaderLSQ::LanePort::recvRespRetry()
{
    lsq->retryCommitWarpInst();
}

AddrRangeList
ShaderLSQ::ControlPort::getAddrRanges() const
{
    // at the moment the assumption is that the master does not care
    AddrRangeList ranges;
    return ranges;
}

bool
ShaderLSQ::ControlPort::recvTimingReq(PacketPtr pkt)
{
    if (pkt->isFlush()) {
        return lsq->addFlushRequest(pkt);
    } else {
        panic("Don't know how to handle control packet");
        return false;
    }
}

Tick
ShaderLSQ::ControlPort::recvAtomic(PacketPtr pkt)
{
    panic("ShaderLSQ::ControlPort::recvAtomic() not implemented!\n");
    return 0;
}

void
ShaderLSQ::ControlPort::recvFunctional(PacketPtr pkt)
{
    panic("ShaderLSQ::ControlPort::recvFunctional() not implemented!\n");
}

void
ShaderLSQ::ControlPort::recvRespRetry()
{
    panic("ShaderLSQ::ControlPort::recvRespRetry() not implemented!\n");
}

bool
ShaderLSQ::CachePort::recvTimingResp(PacketPtr pkt)
{
    return lsq->recvResponsePkt(pkt);
}

void
ShaderLSQ::CachePort::recvReqRetry()
{
    lsq->scheduleRetryInject();
}

bool
ShaderLSQ::addFlushRequest(PacketPtr pkt)
{
    // TODO: When flush is for a particular warp (e.g. membar instruction),
    // set a flag that blocks incoming requests to that particular warp until
    // it is freed, and then send a response to the membar flush
    assert(pkt->req->getPaddr() == Addr(0));
    flushing = true;
    flushingPkt = pkt;
    DPRINTF(ShaderLSQ, "Received flush request\n");
    if (numActiveWarpInstBuffers == 0) processFlush();
    return true;
}

void
ShaderLSQ::incrementActiveWarpInstBuffers()
{
    if (lastWarpInstBufferChange > 0) {
        Tick sinceLastChange = curTick() - lastWarpInstBufferChange;
        activeWarpInstBuffers.sample(numActiveWarpInstBuffers, sinceLastChange);
    }
    numActiveWarpInstBuffers++;
    lastWarpInstBufferChange = curTick();
}

void
ShaderLSQ::decrementActiveWarpInstBuffers()
{
    if (lastWarpInstBufferChange > 0) {
        Tick sinceLastChange = curTick() - lastWarpInstBufferChange;
        activeWarpInstBuffers.sample(numActiveWarpInstBuffers, sinceLastChange);
    }
    numActiveWarpInstBuffers--;
    lastWarpInstBufferChange = curTick();
}

bool
ShaderLSQ::addLaneRequest(int lane_id, PacketPtr pkt)
{
    if (flushing) {
        // ShaderLSQ does not currently support starting further requests
        // while it is flushing at the end of a kernel
        // TODO: If adding flush support at a finer granularity than the
        // complete LSQ, this code path will need to be updated appropriately
        panic("ShaderLSQ does not support adding requests while flushing\n");
        return false;
    }

    if (!dispatchWarpInstBuf) {
        assert(!dispatchInstEvent.scheduled());
        assert(pkt->req->threadId() < maxNumWarpsPerCore);

        // TODO: Consider putting in a per-warp limitation on number of
        // concurrent warp instructions in the LSQ
        if (availableWarpInstBufs.empty()) {
            // Simple deadlock detection
            if (ticksToCycles(curTick() - lastWarpInstBufferChange) > Cycles(1000000)) {
                panic("LSQ deadlocked by running out of buffers!");
            }
            return false;
        }

        // Allocate and initialize a warp instruction dispatch buffer to
        // gather the requests before coalescing into cache accesses
        dispatchWarpInstBuf = availableWarpInstBufs.front();
        dispatchWarpInstBuf->initializeInstBuffer(pkt);
        availableWarpInstBufs.pop();
        incrementActiveWarpInstBuffers();

        // Schedule an event for when the dispatch buffer should be handled
        schedule(dispatchInstEvent, clockEdge(Cycles(0)));
        DPRINTF(ShaderLSQ,
                "[%d: ] Starting %s instruction (pc: 0x%x) at tick: %llu\n",
                pkt->req->threadId(), dispatchWarpInstBuf->getInstTypeString(),
                pkt->req->getPC(), clockEdge(Cycles(0)));
    }

    bool request_added = dispatchWarpInstBuf->addLaneRequest(lane_id, pkt);

    if (request_added) {
        DPRINTF(ShaderLSQ,
                "[%d:%d] Received %s request for vaddr: %p, size: %d\n",
                pkt->req->threadId(), lane_id,
                dispatchWarpInstBuf->getInstTypeString(),
                pkt->req->getVaddr(), pkt->getSize());
    } else {
        DPRINTF(ShaderLSQ,
                "[%d:%d] Rejected %s request for vaddr: %p, size: %d\n",
                pkt->req->threadId(), lane_id,
                dispatchWarpInstBuf->getInstTypeString(),
                pkt->req->getVaddr(), pkt->getSize());
    }

    return request_added;
}

void
ShaderLSQ::dispatchWarpInst()
{
    // Queue the warp instruction to begin issuing accesses after
    // translations complete
    perWarpInstructionQueues[dispatchWarpInstBuf->getWarpId()].push(dispatchWarpInstBuf);

    if (dispatchWarpInstBuf->isFence()) {
        unsigned warp_id = dispatchWarpInstBuf->getWarpId();
        dispatchWarpInstBuf->startFence();
        if (perWarpOutstandingAccesses[warp_id] == 0 &&
            perWarpInstructionQueues[warp_id].front() == dispatchWarpInstBuf) {

            clearFenceAtQueueHead(warp_id);
            assert(perWarpInstructionQueues[warp_id].empty());
        }
    } else {
        // Coalesce memory requests for the dispatched warp instruction
        dispatchWarpInstBuf->coalesceMemRequests();

        // Issue translation requests for the coalesced accesses
        issueWarpInstTranslations(dispatchWarpInstBuf);
    }

    // Clear the dispatch buffer
    dispatchWarpInstBuf = NULL;
}

void
ShaderLSQ::issueWarpInstTranslations(WarpInstBuffer *warp_inst)
{
    BaseTLB::Mode mode;
    if (warp_inst->isLoad()) {
        mode = BaseTLB::Read;
    } else if(warp_inst->isStore() || warp_inst->isAtomic()) {
        mode = BaseTLB::Write;
    } else {
        panic("Trying to issue translations for unknown instruction type!");
    }

    const list<WarpInstBuffer::CoalescedAccess*> *coalesced_accesses =
            warp_inst->getCoalescedAccesses();
    warpCoalescedAccesses.sample(coalesced_accesses->size());
    list<WarpInstBuffer::CoalescedAccess*>::const_iterator iter =
            coalesced_accesses->begin();
    for (; iter != coalesced_accesses->end(); iter++) {
        WarpInstBuffer::CoalescedAccess *mem_access = *iter;
        RequestPtr req = mem_access->req;
        DPRINTF(ShaderLSQ, "[%d: ] Translating vaddr: %p\n",
                mem_access->getWarpId(), req->getVaddr());

        req->setExtraData((uint64_t)mem_access);

        WholeTranslationState *state =
                new WholeTranslationState(req, NULL, NULL, mode);
        DataTranslation<ShaderLSQ*> *translation
                = new DataTranslation<ShaderLSQ*>(this, state);

        mem_access->tlbStartCycle = curCycle();
        tlb->beginTranslateTiming(req, translation, mode);
    }
}

void
ShaderLSQ::finishTranslation(WholeTranslationState *state)
{
    if (state->getFault() != NoFault) {
        // The ShaderLSQ and ShaderTLBs do not currently have a way to signal
        // to a CPU core how a fault should be handled. With current
        // organization, this should not occur unless there are bugs in GPU
        // memory handling
        panic("Translation encountered fault (%s) for address 0x%x\n",
              state->getFault()->name(), state->mainReq->getVaddr());
    }

    WarpInstBuffer::CoalescedAccess *mem_access =
        (WarpInstBuffer::CoalescedAccess*)state->mainReq->getExtraData();

    DPRINTF(ShaderLSQ,
            "[%d: ] Finished translation for vaddr: %p, paddr: %p\n",
            mem_access->getWarpId(), state->mainReq->getVaddr(),
            state->mainReq->getPaddr());

    // Initialize the packet using the translated access and in the case that
    // this is a write access, set the data to be sent to cache
    PacketPtr pkt = mem_access;
    mem_access->reinitFromRequest();
    if (pkt->isWrite()) {
        mem_access->moveDataToPacket();
    } else {
        assert(pkt->isRead());
        pkt->allocate();
    }

    if (state->delay) {
        tlbMissLatency.sample(curCycle() - mem_access->tlbStartCycle);
    }

    delete state;

    WarpInstBuffer *warp_inst = mem_access->getWarpBuffer();
    warp_inst->setTranslated(mem_access);

    if (warp_inst == perWarpInstructionQueues[warp_inst->getWarpId()].front()) {
        pushToInjectBuffer(mem_access);
        if (!injectAccessesEvent.scheduled() && !mshrsFull) {
            // Schedule inject event to incur delay
            schedule(injectAccessesEvent,
                clockEdge(Cycles(mem_access->getInjectCycle() - curCycle())));
        }
    }
}

void
ShaderLSQ::pushToInjectBuffer(WarpInstBuffer::CoalescedAccess *mem_access)
{
    mem_access->setInjectCycle(Cycles(curCycle() + l1TagAccessCycles));
    injectBuffer.push_back(mem_access);
    DPRINTF(ShaderLSQ,
            "[%d: ] Queuing access for paddr: %p, size: %d, tick: %llu\n",
            mem_access->getWarpId(), mem_access->req->getPaddr(),
            mem_access->getSize(),
            clockEdge(mem_access->getInjectCycle()));
}

void
ShaderLSQ::injectCacheAccesses()
{
    assert(!mshrsFull);
    assert(!injectBuffer.empty());
    unsigned num_injected = 0;
    WarpInstBuffer::CoalescedAccess *mem_access = injectBuffer.front();
    while (!injectBuffer.empty() && num_injected < injectWidth &&
           curCycle() >= nextAllowedInject &&
           curCycle() >= mem_access->getInjectCycle()) {

        Addr line_addr = addrToLine(mem_access->req->getPaddr());
        if (blockedLineAddrs[line_addr]) {
            // Unblock inject buffer by queuing access to wait for prior access
            // NOTE: This path must inspect the CoalescedAccess to see if it
            // can be injected. This could be counted against the injection
            // width for this cycle, but it is not currently counted here
            blockedAccesses[line_addr].push(mem_access);
            injectBuffer.pop_front();
            mshrHitQueued++;
            DPRINTF(ShaderLSQ,
                    "[%d: ] Line blocked %s access for paddr: %p\n",
                    mem_access->getWarpId(),
                    mem_access->getWarpBuffer()->getInstTypeString(),
                    mem_access->req->getPaddr());
        } else {
            if (!cachePort.sendTimingReq(mem_access)) {
                DPRINTF(ShaderLSQ,
                        "[%d: ] MSHR blocked %s access for paddr: %p\n",
                        mem_access->getWarpId(),
                        mem_access->getWarpBuffer()->getInstTypeString(),
                        mem_access->req->getPaddr());
                mshrsFull = true;
                mshrsFullStarted = curCycle();
                mshrsFullCount++;
                return;
            } else {
                DPRINTF(ShaderLSQ,
                        "[%d: ] Injected %s access for paddr: %p\n",
                        mem_access->getWarpId(),
                        mem_access->getWarpBuffer()->getInstTypeString(),
                        mem_access->req->getPaddr());
                blockedLineAddrs[line_addr] = true;
                if (mem_access->isWrite()) {
                    // Block issue while the store data is being serialized
                    // through the port to the cache (1 cyc/subline)
                    unsigned num_sublines = mem_access->getSize() / sublineBytes;
                    nextAllowedInject = Cycles(curCycle() + num_sublines);
                }
                injectBuffer.pop_front();
                num_injected++;
                perWarpOutstandingAccesses[mem_access->getWarpId()]++;
                accessesOutstandingToCache++;
                WarpInstBuffer *warp_inst = mem_access->getWarpBuffer();
                warp_inst->removeCoalesced(mem_access);
                if (warp_inst->coalescedAccessesSize() == 0) {
                    int warp_id = warp_inst->getWarpId();
                    // All accesses have entered cache hierarchy, so remove
                    // this warp instruction from the issuing position (head)
                    // to let the next warp instruction from this warp inject
                    perWarpInstructionQueues[warp_id].pop();
                    if (!perWarpInstructionQueues[warp_id].empty()) {
                        WarpInstBuffer *next_warp_inst =
                                perWarpInstructionQueues[warp_id].front();
                        if (!next_warp_inst->isFence()) {
                            const list<WarpInstBuffer::CoalescedAccess*> *translated_accesses =
                                    next_warp_inst->getTranslatedAccesses();
                            list<WarpInstBuffer::CoalescedAccess*>::const_iterator iter =
                                    translated_accesses->begin();
                            for (; iter != translated_accesses->end(); iter++) {
                                pushToInjectBuffer(*iter);
                            }
                        }
                    }
                }
            }
        }

        // Get the next access to check if it can also be injected
        mem_access = injectBuffer.front();
    }

    if (!injectBuffer.empty()) {
        assert(!mshrsFull); // Shouldn't reach this code if cache is blocked
        WarpInstBuffer::CoalescedAccess *next_access = injectBuffer.front();
        if (curCycle() >= next_access->getInjectCycle()) {
            schedule(injectAccessesEvent, nextCycle());
        } else {
            schedule(injectAccessesEvent,
                clockEdge(Cycles(next_access->getInjectCycle() - curCycle())));
        }
    }
}

void
ShaderLSQ::scheduleRetryInject()
{
    assert(mshrsFull);
    assert(!injectBuffer.empty());
    assert(!injectAccessesEvent.scheduled());
    mshrsFull = false;
    mshrsFullCycles += curCycle() - mshrsFullStarted;
    DPRINTF(ShaderLSQ, "[ : ] Unblocking MSHRs, restarting injection\n");
    schedule(injectAccessesEvent, clockEdge(Cycles(0)));
}

bool
ShaderLSQ::recvResponsePkt(PacketPtr pkt)
{
    if (pkt->isFlush()) {
        assert(pkt->isResponse());
        assert(forwardFlush);
        finalizeFlush();
        delete pkt->req;
        delete pkt;
        return true;
    }
    WarpInstBuffer::CoalescedAccess *mem_access =
            dynamic_cast<WarpInstBuffer::CoalescedAccess*>(pkt);
    assert(mem_access);

    // Push the completed memory access into eject buffer
    ejectBuffer.push(mem_access);

    // Check for unblocked accesses, and schedule inject if possible
    Addr line_addr = addrToLine(mem_access->req->getPaddr());
    assert(blockedLineAddrs[line_addr]);
    blockedLineAddrs.erase(line_addr);
    if (blockedAccesses.count(line_addr) > 0) {
        assert(!blockedAccesses[line_addr].empty());
        // Previously blocked accesses get priority, so add one to the
        // front of the inject buffer, and schedule inject event
        // NOTE: Pushing unblocked memory accesses to the front of the inject
        // queue constitutes an arbitration decision, which could be changed
        // in the future. Unblocked accesses could be pushed at any point in
        // the queue (as long as per-warp instruction ordering is preserved)
        WarpInstBuffer::CoalescedAccess *next_access =
                                            blockedAccesses[line_addr].front();
        blockedAccesses[line_addr].pop();
        if (blockedAccesses[line_addr].empty()) {
            blockedAccesses.erase(line_addr);
        }
        // Assert that the unblocked access has been tried for inject previously
        assert(curCycle() > next_access->getInjectCycle());
        injectBuffer.push_front(next_access);
        if (!mshrsFull) {
            if (injectAccessesEvent.scheduled()) {
                reschedule(injectAccessesEvent, clockEdge(Cycles(0)));
            } else {
                schedule(injectAccessesEvent, clockEdge(Cycles(0)));
            }
        }
    }

    // If not scheduled, schedule ejectResponsesEvent
    if (!ejectAccessesEvent.scheduled()) {
        schedule(ejectAccessesEvent, clockEdge(Cycles(0)));
    }
    DPRINTF(ShaderLSQ, "[%d: ] Received %s response for paddr: %p\n",
            mem_access->getWarpId(),
            mem_access->getWarpBuffer()->getInstTypeString(),
            mem_access->req->getPaddr());
    accessesOutstandingToCache--;
    return true;
}

void
ShaderLSQ::ejectAccessResponses()
{
    // TODO: Consider separating readEjectWidth from writeEjectWidth, since
    // reads are only responses that consume bandwidth on return to the core
    assert(!ejectBuffer.empty());
    unsigned num_ejected = 0;
    while (!ejectBuffer.empty() && num_ejected < ejectWidth) {
        WarpInstBuffer::CoalescedAccess *mem_access = ejectBuffer.front();
        WarpInstBuffer *warp_inst = mem_access->getWarpBuffer();
        DPRINTF(ShaderLSQ,
                "[%d: ] Ejected %s for vaddr: %p, paddr: %p\n",
                warp_inst->getWarpId(),
                warp_inst->getInstTypeString(),
                mem_access->req->getVaddr(), mem_access->req->getPaddr());
        perWarpOutstandingAccesses[mem_access->getWarpId()]--;
        bool inst_complete = warp_inst->finishAccess(mem_access);
        if (inst_complete) {
            pushToCommitBuffer(warp_inst);

            // If there is a fence at the head of the per-warp instruction queue
            // and all prior per-warp memory accesses are complete, clear it
            int warp_id = warp_inst->getWarpId();
            if (perWarpOutstandingAccesses[warp_id] == 0 &&
                !perWarpInstructionQueues[warp_id].empty() &&
                perWarpInstructionQueues[warp_id].front()->isFence()) {

                clearFenceAtQueueHead(warp_id);
            }
        }
        ejectBuffer.pop();
        num_ejected++;
    }
    if (!ejectBuffer.empty())
        schedule(ejectAccessesEvent, nextCycle());
}

void
ShaderLSQ::clearFenceAtQueueHead(int warp_id) {
    assert(perWarpOutstandingAccesses[warp_id] == 0);
    assert(!perWarpInstructionQueues[warp_id].empty());
    WarpInstBuffer *next_warp_inst = perWarpInstructionQueues[warp_id].front();
    assert(next_warp_inst->isFence());
    perWarpInstructionQueues[warp_id].pop();
    assert(perWarpInstructionQueues[warp_id].empty());
    next_warp_inst->arriveAtFence();
    pushToCommitBuffer(next_warp_inst);
}

void
ShaderLSQ::pushToCommitBuffer(WarpInstBuffer *warp_inst) {
    warp_inst->setCompleteTick(clockEdge(completeCycles));
    commitInstBuffer.push(warp_inst);
    if (!commitInstEvent.scheduled() && !writebackBlocked) {
        assert(commitInstBuffer.size() == 1);
        schedule(commitInstEvent, clockEdge(completeCycles));
    }
}

void
ShaderLSQ::commitWarpInst()
{
    assert(!writebackBlocked);
    WarpInstBuffer *warp_inst = commitInstBuffer.front();
    assert(curTick() >= warp_inst->getCompleteTick());
    if (warp_inst->isLoad() || warp_inst->isFence() || warp_inst->isAtomic()) {
        PacketPtr* lane_request_pkts = warp_inst->getLaneRequestPkts();
        for (int i = 0; i < warpSize; i++) {
            PacketPtr pkt = lane_request_pkts[i];
            if (pkt) {
                if (warp_inst->isFence()) {
                    pkt->makeTimingResponse();
                }
                if (!lanePorts[i]->sendTimingResp(pkt)) {
                    // Fence responses are always accepted by the CudaCore
                    assert(!warp_inst->isFence());
                    writebackBlocked = true;
                    writebackBlockedCycles++;
                    return;
                }
                lane_request_pkts[i] = NULL;
            }
        }
    }
    DPRINTF(ShaderLSQ, "[%d: ] Completing %s instruction\n",
            warp_inst->getWarpId(), warp_inst->getInstTypeString());
    commitInstBuffer.pop();
    if (!commitInstBuffer.empty()) {
        schedule(commitInstEvent,
                 max(commitInstBuffer.front()->getCompleteTick(), nextCycle()));
    }
    if (warp_inst->isLoad()) {
        warpLatencyRead.sample(ticksToCycles(warp_inst->getLatency()));
    } else if (warp_inst->isStore()) {
        warpLatencyWrite.sample(ticksToCycles(warp_inst->getLatency()));
    } else if (warp_inst->isFence()) {
        warpLatencyFence.sample(ticksToCycles(warp_inst->getLatency()));
    } else if (warp_inst->isAtomic()) {
        warpLatencyAtomic.sample(ticksToCycles(warp_inst->getLatency()));
    } else {
        panic("Don't know how to record latency for this instruction\n");
    }

    warp_inst->resetState();
    decrementActiveWarpInstBuffers();
    availableWarpInstBufs.push(warp_inst);
    if (flushing && numActiveWarpInstBuffers == 0) processFlush();
}

void
ShaderLSQ::retryCommitWarpInst()
{
    writebackBlocked = false;
    assert(!commitInstEvent.scheduled());
    if (!commitInstBuffer.empty()) {
        schedule(commitInstEvent, clockEdge(Cycles(0)));
    }
}

void
ShaderLSQ::processFlush()
{
    DPRINTF(ShaderLSQ, "Processing flush request\n");
    assert(flushing && flushingPkt);
    assert(numActiveWarpInstBuffers == 0);
    Tick since_last_change = curTick() - lastWarpInstBufferChange;
    activeWarpInstBuffers.sample(numActiveWarpInstBuffers, since_last_change);
    lastWarpInstBufferChange = 0;
    if (forwardFlush) {
        MasterID master_id = flushingPkt->req->masterId();
        int asid = 0;
        Addr addr(0);
        Request::Flags flags;
        RequestPtr req = new Request(asid, addr, flags, master_id);
        PacketPtr flush_pkt = new Packet(req, MemCmd::FlushAllReq);
        if (!cachePort.sendTimingReq(flush_pkt)) {
            panic("Unable to forward flush to cache!\n");
        }
    } else {
        finalizeFlush();
    }
}

void ShaderLSQ::finalizeFlush()
{
    assert(flushing && flushingPkt);
    flushingPkt->makeTimingResponse();
    if (!controlPort.sendTimingResp(flushingPkt)) {
        panic("Unable to respond to flush message!\n");
    }
    flushingPkt = NULL;
    flushing = false;
    DPRINTF(ShaderLSQ, "Flush request complete\n");
}

void
ShaderLSQ::regStats()
{
    activeWarpInstBuffers
        .name(name()+".warpInstBufActive")
        .desc("Histogram of number of active warp inst buffers at a given time")
        .init(warpInstBufPoolSize+1)
        ;
    accessesOutstandingToCache
        .name(name()+".cacheAccesses")
        .desc("Average number of concurrent outstanding cache accesses")
        ;
    writebackBlockedCycles
        .name(name()+".writebackBlockedCycles")
        .desc("Number of cycles blocked for core writeback stage")
        ;
    mshrHitQueued
        .name(name()+".mshrHitQueued")
        .desc("Number of hits on blocked lines in MSHRs")
        ;
    mshrsFullCycles
        .name(name()+".mshrsFullCycles")
        .desc("Number of cycles stalled waiting for an MSHR")
        ;
    mshrsFullCount
        .name(name()+".mshrsFullCount")
        .desc("Number of times MSHRs filled")
        ;
    warpCoalescedAccesses
        .name(name() + ".warpCoalescedAccesses")
        .desc("Number of coalesced accesses per warp instruction")
        .init(33)
        ;
    warpLatencyRead
        .name(name() + ".warpLatencyRead")
        .desc("Latency in cycles for whole warp to finish the read")
        .init(16)
        ;
    warpLatencyWrite
        .name(name() + ".warpLatencyWrite")
        .desc("Latency in cycles for whole warp to finish the write")
        .init(16)
        ;
    warpLatencyFence
        .name(name() + ".warpLatencyFence")
        .desc("Latency in cycles for whole warp to finish the fence")
        .init(16)
        ;
    warpLatencyAtomic
        .name(name() + ".warpLatencyAtomic")
        .desc("Latency in cycles for whole warp to finish the atomic")
        .init(16)
        ;
    tlbMissLatency
        .name(name() + ".tlbMissLatency")
        .desc("Latency in cycles for TLB miss")
        .init(16)
        ;
}


ShaderLSQ *ShaderLSQParams::create() {
    return new ShaderLSQ(this);
}
