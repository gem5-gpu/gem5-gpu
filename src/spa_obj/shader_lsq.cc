/*
 * Copyright (c) 2012 Mark D. Hill and David A. Wood
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
 */

#include <deque>
#include <list>
#include <set>
#include <vector>

#include "shader_lsq.hh"
#include "shader_tlb.hh"
#include "debug/ShaderLSQ.hh"

using namespace std;

ShaderLSQ::ShaderLSQ(Params *p)
	: MemObject(p), responsePortBlocked(false), cachePort(name() + "cache_port", this),
      warpSize(p->warp_size), coalescingRegister(NULL), requestBufferDepth(p->request_buffer_depth),
      tlb(p->data_tlb), coalesceEvent(this), sendResponseEvent(this)
{
    // create the lane ports based on the number of connected ports
    for (int i = 0; i < p->port_lane_port_connection_count; i++) {
        lanePorts.push_back(new LanePort(csprintf("%s-lane-%d", name(), i),
                                          i, this));
    }

    requestBuffers.resize(p->request_buffers);

    for (int i=0; i < warpSize; i++) {
        sendRubyRequestEvents.push_back(new SendRubyRequestEvent(this, i));
    }
}

ShaderLSQ::~ShaderLSQ()
{
    for (int i=0; i < warpSize; i++) {
        delete sendRubyRequestEvents[i];
        delete lanePorts[i];
    }
}

BaseMasterPort &
ShaderLSQ::getMasterPort(const std::string &if_name, PortID idx)
{
    if (if_name == "cache_port") {
        return cachePort;
    } else {
        return MemObject::getMasterPort(if_name, idx);
    }
}

BaseSlavePort &
ShaderLSQ::getSlavePort(const std::string &if_name, PortID idx)
{
    if (if_name != "lane_port") {
        // pass it along to our super class
        return MemObject::getSlavePort(if_name, idx);
    } else {
        if (idx >= static_cast<PortID>(lanePorts.size())) {
            panic("RubyPort::getSlavePort: unknown index %d\n", idx);
        }

        return *lanePorts[idx];
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
	// Called for each lane in the cycle that the address is read from the RF

	// NOTE: If this is going to fail, it needs to fail early. The first port
	//       in lanePorts must fail so that the lanes are in the same state

	DPRINTF(ShaderLSQ, "Address 0x%llx\n", pkt->req->getVaddr());

    // Get the cross-lane warp request from the lsq which owns this port
    ShaderLSQ &lsq = (ShaderLSQ&)owner;
    WarpRequest *warpRequest = lsq.coalescingRegister;

    if (warpRequest && warpRequest->occupiedTick != curTick()) {
        // There is an old request currently occupying this register
        DPRINTF(ShaderLSQ, "Stall for coalescer\n");
        return false;
    }
    if (!warpRequest) {
        // I must be the first lane here setup cross lane info
        lsq.coalescingRegister = new WarpRequest(lsq.warpSize);
        warpRequest = lsq.coalescingRegister;
        warpRequest->occupiedCycle = lsq.curCycle();
        warpRequest->occupiedTick = curTick();
        warpRequest->size = pkt->req->getSize();
        warpRequest->pc = pkt->req->getPC();
        warpRequest->cid = pkt->req->contextId();
        warpRequest->warpId = pkt->req->threadId();
        warpRequest->masterId = pkt->req->masterId();
        if (pkt->isRead()) {
            warpRequest->read = true;
        } else if (pkt->isWrite()) {
            warpRequest->write = true;
        } else {
            panic("ShaderLSQ::LanePort::recvTimingReq only supports reads / writes\n");
        }

        lsq.schedule(lsq.coalesceEvent, lsq.nextCycle());
    }

    // Requests should be the same size for all lanes
    assert(warpRequest->size == pkt->req->getSize());
    assert(warpRequest->read ? pkt->isRead() : pkt->isWrite());
    assert(warpRequest->write ? pkt->isWrite() : pkt->isRead());
    assert(warpRequest->pc == pkt->req->getPC());
    assert(warpRequest->warpId == pkt->req->threadId());

    warpRequest->setValid(laneId);
    warpRequest->laneRequests[laneId] = pkt;

	return true;
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
ShaderLSQ::LanePort::recvRetry()
{
    assert(isBlocked);
    isBlocked = false;

    DPRINTF(ShaderLSQ, "RecvRetry for lane %d\n", laneId);

    ShaderLSQ &lsq = (ShaderLSQ&)owner;
    lsq.responsePortBlocked = false;

    if (!lsq.sendResponseEvent.scheduled()) {
        lsq.schedule(lsq.sendResponseEvent, curTick());
    }
}

bool
ShaderLSQ::CachePort::recvTimingResp(PacketPtr pkt)
{
	// Called when the RubyPort is returning data

    DPRINTF(ShaderLSQ, "Received response for addr 0x%llx\n",
            pkt->req->getVaddr());

    ShaderLSQ &lsq = (ShaderLSQ&)owner;

    return lsq.prepareResponse(pkt);
}

void
ShaderLSQ::CachePort::recvRetry()
{
	DPRINTF(ShaderLSQ, "Got retry from Ruby\n");

    ShaderLSQ &lsq = (ShaderLSQ&)owner;

    lsq.rescheduleBlockedRequestBuffers();
}

bool
ShaderLSQ::prepareResponse(PacketPtr pkt)
{
    CoalescedRequest *request =
        dynamic_cast<CoalescedRequest*>(pkt->senderState);

    assert(request);

    DPRINTF(ShaderLSQ, "Got repsonse for %d threads\n", request->activeLanes.size());

    WarpRequest *warpRequest = request->warpRequest;

    vector<int>::iterator iter = request->activeLanes.begin();
    for ( ; iter != request->activeLanes.end(); iter++) {
        PacketPtr respPkt = warpRequest->laneRequests[(*iter)];
        if (request->read) {
            int offset = respPkt->req->getVaddr() - pkt->req->getVaddr();
            assert(offset < pkt->getSize());
            assert(offset >= 0);
            memcpy(respPkt->getPtr<uint8_t>(), pkt->getPtr<uint8_t>()+offset,
                   respPkt->getSize());
        }
        respPkt->makeTimingResponse();
    }

    // remove this coalesced request from its the ld/st buffer
    removeRequestFromBuffer(request);

    // remove this request from the warp request
    warpRequest->coalescedRequests.remove(request);

    delete request;

    delete pkt->req;
    delete pkt;

    if (warpRequest->coalescedRequests.empty()) {
        // All coalesced requests generated for this warp request have finished
        DPRINTF(ShaderLSQ, "Warp request (%d) completely finished!\n", warpRequest->read);
        DPRINTF(ShaderLSQ, "Responses to send: %d\n", responseQueue.size());

        if (warpRequest->read) {
            // For now this is an infinite queue.
            // If it is finite in the future, then this function should return
            // false if the queue is full.
            responseQueue.push_front(warpRequest);
            DPRINTF(ShaderLSQ, "Responses to send: %d\n", responseQueue.size());
            if (!sendResponseEvent.scheduled()) {
                schedule(sendResponseEvent, curTick());
            }
        } else {
            // No need to send repsonse for writes
            vector<PacketPtr>::iterator it = warpRequest->laneRequests.begin();
            for (; it != warpRequest->laneRequests.end(); it++) {
                // Should do this in the requestor, but writes don't need a
                // response and it would take adding a new memory request to
                // do it right. Should not set the pkt->NeedsResponse flag
                if (*it) {
                    delete (*it)->req;
                    delete *it;
                }
            }
            delete warpRequest;
        }

    }

    return true;
}

void
ShaderLSQ::processSendResponseEvent()
{
    assert(!responseQueue.empty());

    if (responsePortBlocked) {
        // This may happen if a recvResp gets called multiple times in a cycle
        DPRINTF(ShaderLSQ, "Repsonse port is blocked!\n");
        return;
    }

    WarpRequest *warpRequest = responseQueue.back();

    assert(warpRequest->read);

    DPRINTF(ShaderLSQ, "Trying to send resp\n");

    for (int i=0; i<warpSize; i++) {
        if (!warpRequest->isValid(i)) {
            continue;
        }
        PacketPtr respPkt = warpRequest->laneRequests[i];
        assert(!lanePorts[i]->isBlocked);
        if (!lanePorts[i]->sendTimingResp(respPkt)) {
            // Like when the shader core is sending requests to the LSQ, this
            // must fail for the first lane if it's going to fail!
            lanePorts[i]->isBlocked = true; // Just for sanity.
            responsePortBlocked = true;
            DPRINTF(ShaderLSQ, "Sending response to core failed for lane %d (0x%llx)!\n",
                    i, respPkt->req->getVaddr());
            return;
        }
    }

    responseQueue.pop_back();
    DPRINTF(ShaderLSQ, "Responses to send: %d\n", responseQueue.size());
    delete warpRequest;

    if (!responseQueue.empty()) {
        schedule(sendResponseEvent, nextCycle());
    }
}

void
ShaderLSQ::processCoalesceEvent()
{
    assert(coalescingRegister);
    if (coalescedRequests.empty()) {
        // Haven't actually coalesced yet!
        DPRINTF(ShaderLSQ, "Doing the coalescing\n");
        coalesce(coalescingRegister);
    }

    DPRINTF(ShaderLSQ, "Have %d coalesced requests left to send\n",
            coalescedRequests.size());

    assert(!coalescedRequests.empty());

    list<CoalescedRequest*>::iterator iter = coalescedRequests.begin();
    while (iter != coalescedRequests.end()) {
        if (!insertRequestIntoBuffer(*iter)) {
            // Skip this request and try again later
            iter++;
        } else {
            beginTranslation(*iter);
            coalescedRequests.erase(iter++);
            DPRINTF(ShaderLSQ, "Now have %d coalesced requests\n", coalescedRequests.size());
        }
    }

    if (coalescedRequests.empty()) {
        // Coalescer has finished and all requests were put into buffers
        // Now we can accept the next warp request
        DPRINTF(ShaderLSQ, "Finished sending all coalesced requests\n");
        coalescingRegister = NULL;
    }
}

vector<ShaderLSQ::CoalescedRequest*> &
ShaderLSQ::getRequestBuffer(CoalescedRequest *request)
{
    Addr addr = request->req->getVaddr();

    // NOTE: This assumes 128 byte lines and address striping at the L1
    int bufferNum = (addr / 128) % requestBuffers.size();
    request->bufferNum = bufferNum;
    return requestBuffers[bufferNum];
}

bool
ShaderLSQ::insertRequestIntoBuffer(CoalescedRequest *request)
{
    vector<CoalescedRequest*> &buffer = getRequestBuffer(request);

    // If no room in the buffer
    if (buffer.size() >= requestBufferDepth) {
        return false;
    }

    buffer.push_back(request);
    return true;
}

void
ShaderLSQ::removeRequestFromBuffer(CoalescedRequest *request)
{
    vector<CoalescedRequest*> &buffer = getRequestBuffer(request);

    vector<CoalescedRequest*>::iterator it;
    bool found = false;
    for (it = buffer.begin(); it != buffer.end(); it++) {
        if (*it == request) {
            found  = true;
            buffer.erase(it);
            break;
        }
    }
    // It better have been in the buffer!!
    assert(found);

    if (!coalescedRequests.empty() && !coalesceEvent.scheduled()) {
        // This may have unblocked the coalescer
        schedule(coalesceEvent, curTick());
    }
}

void
ShaderLSQ::beginTranslation(CoalescedRequest *request)
{
    BaseTLB::Mode mode;
    if (request->read) {
        mode = BaseTLB::Read;
    } else if (request->write) {
        mode = BaseTLB::Write;
    } else {
        panic("ShaderLSQ::beginTranslation unknown request type\n");
    }
    DPRINTF(ShaderLSQ, "Translating vaddr: 0x%llx\n", request->req->getVaddr());

    RequestPtr req = request->req;
    req->setExtraData((uint64_t)request);

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
    DataTranslation<ShaderLSQ*> *translation
            = new DataTranslation<ShaderLSQ*>(this, state);

    tlb->beginTranslateTiming(req, translation, mode);
}

void
ShaderLSQ::finishTranslation(WholeTranslationState *state)
{
    if (state->getFault() != NoFault) {
        panic("Translation encountered fault (%s) for address 0x%x\n",
              state->getFault()->name(), state->mainReq->getVaddr());
    }

    DPRINTF(ShaderLSQ, "Finished translation on 0x%llx paddr: 0x%llx\n",
            state->mainReq->getVaddr(), state->mainReq->getPaddr());

    CoalescedRequest *request =
        (CoalescedRequest*)state->mainReq->getExtraData();

    delete state;

    request->translated = true;

    Event *e = sendRubyRequestEvents[request->bufferNum];
    if (!e->scheduled()) {
        // if true, This means two translations that hash to the same ld/st
        // buffer finished in the same cycle
        schedule(e, curTick());
    }
}

void
ShaderLSQ::rescheduleBlockedRequestBuffers()
{
    list<int>::iterator iter = blockedBufferNums.begin();
    while (iter != blockedBufferNums.end()) {
        Event *e = sendRubyRequestEvents[*iter];
        if (!e->scheduled()) {
            // if true, This means two translations that hash to the same ld/st
            // buffer finished in the same cycle
            schedule(e, curTick());
        }
        blockedBufferNums.erase(iter++);
    }
}

void
ShaderLSQ::sendRubyRequest(int requestBufferNum)
{
    vector<CoalescedRequest*> &buffer = requestBuffers[requestBufferNum];

    // We must iterate through the buffer because if two requests both finish
    // translation in the same cycle then there will be only one event sched.

    vector<CoalescedRequest*>::iterator iter;
    for (iter = buffer.begin(); iter != buffer.end(); iter++) {
        CoalescedRequest *request = *iter;
        if (request->sent || !request->translated) {
            continue;
        }
        DPRINTF(ShaderLSQ, "[RB: %d] About to send req to ruby for vaddr: 0x%llx\n",
                requestBufferNum, request->req->getVaddr());

        PacketPtr pkt;
        if (request->read) {
            pkt = new Packet(request->req, MemCmd::ReadReq);
            pkt->allocate();
        } else if (request->write) {
            pkt = new Packet(request->req, MemCmd::WriteReq);
            pkt->allocate();
            pkt->setData(request->data);
        } else {
            panic("ShaderLSQ::sendRubyRequest bad request type\n");
        }
        pkt->senderState = request;
        if (!cachePort.sendTimingReq(pkt)) {
            delete pkt;
            DPRINTF(ShaderLSQ, "Sending to Ruby failed for buffer num %d\n", requestBufferNum);
            blockedBufferNums.push_back(requestBufferNum);
        } else {
            request->sent = true;
        }
    }
}

void
ShaderLSQ::coalesce(WarpRequest *warpRequest)
{
    // see the CUDA manual where it discusses coalescing rules before reading this
    // Copied from GPGPU-Sim
    int data_size = warpRequest->size;
    unsigned segment_size = 0;
    unsigned warp_parts = 2;
    switch( data_size ) {
    case 1: segment_size = 32; break;
    case 2: segment_size = 64; break;
    case 4: case 8: case 16: segment_size = 128; break;
    }
    unsigned subwarp_size = warpSize / warp_parts;

    for( unsigned subwarp=0; subwarp <  warp_parts; subwarp++ ) {
        std::map<Addr,transaction_info> subwarp_transactions;

        // step 1: find all transactions generated by this subwarp
        for( unsigned thread=subwarp*subwarp_size; thread<subwarp_size*(subwarp+1); thread++ ) {
            if( !warpRequest->isValid(thread) )
                continue;

            // unsigned data_size_coales = data_size;
            unsigned num_accesses = 1;

            // if( space.get_type() == local_space || space.get_type() == param_space_local ) {
            //    // Local memory accesses >4B were split into 4B chunks
            //    if(data_size >= 4) {
            //       data_size_coales = 4;
            //       num_accesses = data_size/4;
            //    }
            //    // Otherwise keep the same data_size for sub-4B access to local memory
            // }

            // assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);

            for(unsigned access=0; access<num_accesses; access++) {
                Addr addr = warpRequest->getAddr(thread);
                Addr block_address = addr & ~((Addr)(segment_size-1));
                unsigned chunk = (addr&127)/32; // which 32-byte chunk within in a 128-byte chunk does this thread access?
                transaction_info &info = subwarp_transactions[block_address];

                // can only write to one segment
                assert(block_address == ((addr+data_size-1) & ~((Addr)(segment_size-1))));

                info.chunks.set(chunk);
                info.activeLanes.push_back(thread);
                // unsigned idx = (addr&127);
                // for( unsigned i=0; i < data_size_coales; i++ )
                //     info.bytes.set(idx+i);
            }
        }

        // step 2: reduce each transaction size, if possible
        std::map< Addr, transaction_info >::iterator t;
        for( t=subwarp_transactions.begin(); t !=subwarp_transactions.end(); t++ ) {
            Addr addr = t->first;
            const transaction_info &info = t->second;

            // memory_coalescing_arch_13_reduce_and_send(is_write, access_type, info, addr, segment_size);
            // Copied below.
            assert( (addr & (segment_size-1)) == 0 );

            const std::bitset<4> &q = info.chunks;
            assert( q.count() >= 1 );
            std::bitset<2> h; // halves (used to check if 64 byte segment can be compressed into a single 32 byte segment)

            unsigned size=segment_size;
            if( segment_size == 128 ) {
                bool lower_half_used = q[0] || q[1];
                bool upper_half_used = q[2] || q[3];
                if( lower_half_used && !upper_half_used ) {
                    // only lower 64 bytes used
                    size = 64;
                    if(q[0]) h.set(0);
                    if(q[1]) h.set(1);
                } else if ( (!lower_half_used) && upper_half_used ) {
                    // only upper 64 bytes used
                    addr = addr+64;
                    size = 64;
                    if(q[2]) h.set(0);
                    if(q[3]) h.set(1);
                } else {
                    assert(lower_half_used && upper_half_used);
                }
            } else if( segment_size == 64 ) {
                // need to set halves
                if( (addr % 128) == 0 ) {
                    if(q[0]) h.set(0);
                    if(q[1]) h.set(1);
                } else {
                    assert( (addr % 128) == 64 );
                    if(q[2]) h.set(0);
                    if(q[3]) h.set(1);
                }
            }
            if( size == 64 ) {
                bool lower_half_used = h[0];
                bool upper_half_used = h[1];
                if( lower_half_used && !upper_half_used ) {
                    size = 32;
                } else if ( (!lower_half_used) && upper_half_used ) {
                    addr = addr+32;
                    size = 32;
                } else {
                    assert(lower_half_used && upper_half_used);
                }
            }
            // m_accessq.push_back( mem_access_t(access_type,addr,size,is_write,info.active,info.bytes) );

            // A map from the word of the block to the lane id
            // Needs to be multimap since two lanes could have same addr
            multimap<int,int> validWords;
            vector<int>::const_iterator iter = info.activeLanes.begin();
            for ( ; iter != info.activeLanes.end(); iter++) {
                Addr a = warpRequest->laneRequests[*iter]->req->getVaddr();
                int offset = a & (size-1);
                validWords.insert(pair<int,int>(offset, *iter));
            }

            multimap<int,int>::iterator it = validWords.begin();
            while (it != validWords.end()) {
                Addr base = addr + it->first;
                vector<int> lanes;
                int chunkSize = warpRequest->size;
                // While the next offset is the current offset + size of word
                // Use >= because could have two requests with same offset
                // incr the current offset
                multimap<int,int>::iterator next(it);
                next++;
                do {
                    lanes.push_back(it->second);
                    if (next == validWords.end()) {
                        // This was the last thread
                        it++;
                        break;
                    }
                    if (it->first + warpRequest->size == next->first) {
                        // Only add to the chunk if the address is the next
                        chunkSize += warpRequest->size;
                    } else if (it->first != next->first) {
                        // if next offset is not cur+size or cur, end of chunk
                        it++;
                        break;
                    }
                    it++;
                    next++;
                } while (it != validWords.end());
                DPRINTF(ShaderLSQ, "Base 0x%llx, chunk %d\n", base, chunkSize);
                // This is a new chunk that we need to send off
                generateMemoryAccess(base, chunkSize, warpRequest, lanes);
            }
        }
    }
}

void
ShaderLSQ::generateMemoryAccess(Addr addr, size_t size, WarpRequest *warpRequest, vector<int> &activeLanes)
{
    DPRINTF(ShaderLSQ, "Generating mem access at 0x%llx for %d bytes for %d threads\n", addr, size, activeLanes.size());

    assert(activeLanes.size() * warpRequest->size >= size);
    // Insert into coalesced request buffer

    Request::Flags flags;
    const int asid = 0;
    RequestPtr req = new Request(asid, addr, size, flags,
                                 warpRequest->masterId,
                                 warpRequest->pc, warpRequest->cid,
                                 warpRequest->warpId);

    CoalescedRequest *request = new CoalescedRequest();
    request->warpRequest = warpRequest;
    request->req = req;
    request->done = false;
    request->sent = false;
    request->translated = false;
    request->activeLanes = activeLanes;
    if (warpRequest->read) {
        request->read = true;
    } else if (warpRequest->write) {
        request->write = true;
        request->data = new uint8_t[size];
        vector<int>::iterator iter = activeLanes.begin();
        for ( ; iter != activeLanes.end(); iter++) {
            int offset = warpRequest->getAddr(*iter) - addr;
            assert(offset >= 0);
            assert(offset < size);
            memcpy(request->data+offset, warpRequest->getData(*iter),
                   warpRequest->size);
        }
    } else {
        panic("Coalescer only supports reads and writes\n");
    }
    warpRequest->coalescedRequests.push_back(request);
    coalescedRequests.push_back(request);
    DPRINTF(ShaderLSQ, "Now have %d coalesced requests\n", coalescedRequests.size());
}


ShaderLSQ *ShaderLSQParams::create() {
    return new ShaderLSQ(this);
}
