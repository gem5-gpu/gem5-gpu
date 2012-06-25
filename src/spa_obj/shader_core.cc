/*
 * Copyright (c) 2011 Mark D. Hill and David A. Wood
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

#include <cmath>
#include <iostream>
#include <map>

#include "arch/tlb.hh"
#include "arch/utility.hh"
#include "base/chunk_generator.hh"
#include "config/the_isa.hh"
#include "cpu/thread_context.hh"
#include "cpu/translation.hh"
#include "debug/ShaderCore.hh"
#include "debug/ShaderCoreAccess.hh"
#include "debug/ShaderCoreTick.hh"
#include "mem/page_table.hh"
#include "params/ShaderCore.hh"
#include "shader_core.hh"
#include "sp_array.hh"

extern void shader_update_mshr_wrapper(size_t addr, mem_fetch *mf, int sid);

using namespace TheISA;
using namespace std;

ShaderCore::ShaderCore(const Params *p) :
        MemObject(p), port("scPort", this), _params(p), tickEvent(this),
        scheduledTickEvent(0), masterId(p->sys->getMasterId(name())),
        id(p->id), dtb(p->dtb), itb(p->itb)
{
    spa = StreamProcessorArray::getStreamProcessorArray();
    int _id = spa->registerShaderCore(this);
    if (_id != id) {
        assert(0);
    }

    stallOnRetry = 0;

    numRetry = 0;
    maxOutstanding = 0;

    DPRINTF(ShaderCore, "[SC:%d] Created shader core\n", id);
}


Port *ShaderCore::getPort(const std::string &if_name, int idx)
{
    return &port;
}


Tick ShaderCore::SCPort::recvAtomic(PacketPtr pkt)
{
    panic("[SC:%d] ShaderCore::SPPort::recvAtomic() not implemented!\n", proc->id);
    return 0;
}

void ShaderCore::initialize(ThreadContext *_tc)
{
    tc = _tc;

    shaderImpl = spa->theGPU->get_shader(id);
}

bool ShaderCore::SCPort::recvTiming(PacketPtr pkt)
{
    map<Addr,mem_fetch *>::iterator iter = proc->outstandingAddrs.find(proc->addrToLine(pkt->req->getVaddr()));

    if(pkt->isRead()) {
        DPRINTF(ShaderCoreAccess, "[SC:%d] Finished read on vaddr 0x%x\n", proc->id, pkt->req->getVaddr());
    } else {
        DPRINTF(ShaderCoreAccess, "[SC:%d] Finished write on vaddr 0x%x\n", proc->id, pkt->req->getVaddr());
    }

    if (iter == proc->outstandingAddrs.end()) {
        panic("We should always find the address!!\n");
    }

    if (pkt->req->isLocked()) {
        if (pkt->isRead()) {
            RequestPtr wreq = new Request();
            Request::Flags flags = Request::LOCKED;
            Addr pc = 0;
            const int asid = 0;
            wreq->setVirt(asid, pkt->req->getVaddr(), pkt->req->getSize(), flags, proc->masterId, pc);

            assert(iter->second != NULL);
            // NOTE: We shouldn't go through translation again!
            if (proc->stallOnRetry) {
                proc->stallOnAtomicQueue = true;
                DPRINTF(ShaderCoreAccess, "[SC:%d] Pending write part of RMW to addr 0x%x\n", proc->id, pkt->req->getVaddr());
                PendingReq *req = new PendingReq(wreq, BaseTLB::Write, iter->second);
                proc->atomicQueue.push(req);
            } else {
                DPRINTF(ShaderCoreAccess, "[SC:%d] Sending write part of RMW to addr 0x%x\n", proc->id, pkt->req->getVaddr());
                proc->accessVirtMem(wreq, iter->second, BaseTLB::Write);
            }
            return true;
        } else {
            // the write part of the RMW finished
            DPRINTF(ShaderCoreAccess, "[SC:%d] Finished RMW to addr 0x%x\n", proc->id, pkt->req->getVaddr());
            iter->second->do_atomic();
        }
    }
    // need to clear mshr so this can commit
    DPRINTF(ShaderCoreAccess, "[SC:%d] removing vaddr 0x%x from outstanding\n", proc->id, pkt->req->getVaddr());
    iter->second->set_reply();
    proc->shaderImpl->accept_ldst_unit_response(iter->second);

    proc->outstandingAddrs.erase(iter);

    return true;
}


void ShaderCore::SCPort::recvRetry() {
    assert(outstandingPkt != NULL);

    proc->numRetry++;

    DPRINTF(ShaderCoreAccess, "recvRetry got called, pkt=%p, (pkt->req->getVaddr() = 0x%x)\n", outstandingPkt, (unsigned int)outstandingPkt->req->getVaddr());

    if(sendTiming(outstandingPkt)) {
        outstandingPkt = NULL;
        proc->stallOnRetry = 0;

        //DPRINTF(ShaderCoreAccess, "Clearing stallOnRetry flag\n");
        if (proc->stallOnAtomicQueue && !proc->scheduledTickEvent) {
            DPRINTF(ShaderCoreAccess, "[SC:%d] Scheduling tick in recvRetry\n", proc->id);
            proc->schedule(proc->tickEvent, curTick());
            proc->scheduledTickEvent = 1;
        }
    } else {
            DPRINTF(ShaderCoreAccess, "pkt(%p) failed more than once with addr 0x%x\n", outstandingPkt, (unsigned int)outstandingPkt->req->getVaddr());
    }
}


int ShaderCore::resourceAvailable(Addr a)
{
    map<Addr,mem_fetch *>::iterator iter = outstandingAddrs.find(addrToLine(a));
    return !stallOnRetry && !stallOnAtomicQueue && (iter == outstandingAddrs.end());
}


void ShaderCore::tick()
{
    DPRINTF(ShaderCoreTick, "[SC:%d] tick\n", id);
    scheduledTickEvent = 0;

    if (atomicQueue.empty()) {
        panic("Why is there nothing in the atomicQueue???\n");
    }

    if (stallOnRetry) {
        DPRINTF(ShaderCoreAccess, "[SC:%d] Stalled on retry, trying again later\n", id);
        return;
    }

    DPRINTF(ShaderCoreAccess, "Emptying atomicQueue. %d total\n", atomicQueue.size());
    assert(!atomicQueue.empty());
    PendingReq *req = atomicQueue.front();
    atomicQueue.pop();
    if (!atomicQueue.empty()) {
        schedule(tickEvent, curTick());
        scheduledTickEvent = 1;
    } else {
        DPRINTF(ShaderCoreAccess, "Finally done emptying atomic queue\n");
        stallOnAtomicQueue = false;
    }

    DPRINTF(ShaderCoreAccess, "Checking req->Vaddr=0x%x\n", req->req->getVaddr());

    if (req->mode == BaseTLB::Write) {
        DPRINTF(ShaderCoreAccess, "[SC:%d] pendingReq is write\n", id);
        accessVirtMem(req->req, req->mf, BaseTLB::Write);
    } else if (req->req->isLocked()) {
        DPRINTF(ShaderCoreAccess, "[SC:%d] pendingReq is rmw\n", id);
        accessVirtMem(req->req, req->mf, BaseTLB::Read);
    } else {
        DPRINTF(ShaderCoreAccess, "[SC:%d] pendingReq is read\n", id);
        accessVirtMem(req->req, req->mf, BaseTLB::Read);
    }

}

void ShaderCore::finishTranslation(WholeTranslationState *state)
{
    //DPRINTF(ShaderCoreAccess, "[SC:%d] finished translation! It was a %d\n", id, state->mode);
    //DPRINTF(ShaderCoreAccess, "[SC:%d] virtual: 0x%x -> 0x%x physical\n", id, state->mainReq->getVaddr(), state->mainReq->getPaddr());
    PacketPtr pkt;
    if (state->mode == BaseTLB::Read) {
        pkt = new Packet(state->mainReq, MemCmd::ReadReq, Packet::Broadcast);
        pkt->dataDynamicArray(new uint8_t[state->mainReq->getSize()]);
    } else {
        pkt = new Packet(state->mainReq, MemCmd::WriteReq, Packet::Broadcast);
    }

    sendPkt(pkt);
}


int ShaderCore::readTiming (Addr a, size_t size, mem_fetch *mf)
{
    if (!resourceAvailable(a)) {
        return 1;
    }
    DPRINTF(ShaderCoreAccess, "[SC:%d] Reading %d bytes virtual address 0x%x\n", id, size, a);

    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;

    req->setVirt(asid, a, size, flags, masterId, pc);

    accessVirtMem(req, mf, BaseTLB::Read);

    return 0;
}


int ShaderCore::writeTiming(Addr a, size_t size, mem_fetch *mf)
{
    if (!resourceAvailable(a)) {
        return 1;
    }
    DPRINTF(ShaderCoreAccess, "[SC:%d] Writing %d bytes virtual address 0x%x\n", id, size, a);

    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;

    req->setVirt(asid, a, size, flags, masterId, pc);

    accessVirtMem(req, mf, BaseTLB::Write);
    return 0;
}

int ShaderCore::atomicRMW(Addr a, size_t size,  mem_fetch *mf)
{
    if (!resourceAvailable(a)) {
        return 1;
    }
    DPRINTF(ShaderCoreAccess, "[SC:%d] AtomicRMW %d bytes virtual address 0x%x\n", id, size, a);

    RequestPtr req = new Request();
    Request::Flags flags = Request::LOCKED;
    Addr pc = 0;
    const int asid = 0;

    req->setVirt(asid, a, size, flags, masterId, pc);

    accessVirtMem(req, mf, BaseTLB::Read);

    return 0;
}

inline Addr ShaderCore::addrToLine(Addr a)
{
    unsigned int maskBits = spa->ruby->getBlockSizeBits();
    return a & (((uint64_t)-1) << maskBits);
}

void ShaderCore::accessVirtMem(RequestPtr req, mem_fetch *mf, BaseTLB::Mode mode)
{
    assert(mf != NULL);

    if (req->isLocked() && mode == BaseTLB::Write) {
        // skip below
    } else {
        map<Addr,mem_fetch *>::iterator iter = outstandingAddrs.find(addrToLine(req->getVaddr()));
        assert(iter == outstandingAddrs.end());
        outstandingAddrs[addrToLine(req->getVaddr())] = mf;
    }

    WholeTranslationState *state =
            new WholeTranslationState(req, new uint8_t[req->getSize()], NULL, mode);
    DataTranslation<ShaderCore*> *translation
            = new DataTranslation<ShaderCore*>(this, state);

    dtb->translateTiming(req, tc, translation, mode);
}


bool ShaderCore::sendPkt(PacketPtr pkt) {

    if (!port.sendTiming(pkt)) {
        DPRINTF(ShaderCoreAccess, "sendTiming failed in sendPkt (pkt->req->getVaddr()=0x%x)\n", (unsigned int)pkt->req->getVaddr());
        stallOnRetry = 1;
        port.outstandingPkt = pkt;
        return false;
    }
    return true;
}


ShaderCore *ShaderCoreParams::create() {
    return new ShaderCore(this);
}
