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
#include "debug/ShaderCoreFetch.hh"
#include "debug/ShaderCoreTick.hh"
#include "debug/ShaderMemTrace.hh"
#include "gpu/gpgpu-sim/cuda_core.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "mem/page_table.hh"
#include "mem/ruby/fusion_profiler/fusion_profiler.hh"
#include "params/ShaderCore.hh"

using namespace TheISA;
using namespace std;

ShaderCore::ShaderCore(const Params *p) :
    MemObject(p), dataPort(name() + ".data_port", this),
    instPort(name() + ".inst_port", this), _params(p), tickEvent(this),
    scheduledTickEvent(false), masterId(p->sys->getMasterId(name())), id(p->id),
    dtb(p->dtb), itb(p->itb), spa(p->spa)
{
    writebackBlocked = -1; // Writeback is not blocked
    stallOnDCacheRetry = false;
    stallOnICacheRetry = false;

    spa->registerShaderCore(this);

    warpSize = spa->getWarpSize();

    if (p->port_lsq_port_connection_count != warpSize) {
        panic("Shader core lsq_port size != to warp size\n");
    }

    // create the ports
    for (int i = 0; i < p->port_lsq_port_connection_count; ++i) {
        lsqPorts.push_back(new LSQPort(csprintf("%s-lsqPort%d", name(), i),
                                    this, i));
    }

    DPRINTF(ShaderCore, "[SC:%d] Created shader core\n", id);
}

BaseMasterPort&
ShaderCore::getMasterPort(const std::string &if_name, PortID idx)
{
    if (if_name == "data_port") {
        return dataPort;
    } else if (if_name == "inst_port") {
        return instPort;
    } else if (if_name == "lsq_port") {
        if (idx >= static_cast<PortID>(lsqPorts.size())) {
            panic("ShaderCore::getMasterPort: unknown index %d\n", idx);
        }
        return *lsqPorts[idx];
    } else {
        return MemObject::getMasterPort(if_name, idx);
    }
}

void
ShaderCore::unserialize(Checkpoint *cp, const std::string &section)
{
    // Intentionally left blank to keep from trying to read shader header from
    // checkpoint files. Allows for restore into any number of shader cores.
    // NOTE: Cannot checkpoint during kernels
}

Tick ShaderCore::SCDataPort::recvAtomic(PacketPtr pkt)
{
    panic("[SC:%d] ShaderCore::SPPort::recvAtomic() not implemented!\n", proc->id);
    return 0;
}

void ShaderCore::SCDataPort::recvFunctional(PacketPtr pkt)
{
    panic("[SC:%d] ShaderCore::SPPort::recvFunctional() not implemented!\n", proc->id);
}

void ShaderCore::initialize()
{
    shaderImpl = spa->getTheGPU()->get_shader(id);
}

bool ShaderCore::SCDataPort::recvTimingResp(PacketPtr pkt)
{
    map<Addr,mem_fetch *>::iterator iter = proc->busyDataCacheLineAddrs.find(proc->addrToLine(pkt->req->getVaddr()));

    DPRINTF(ShaderCoreAccess, "[SC:%d] Finished %s on vaddr 0x%x\n", proc->id, (pkt->isWrite()) ? "write" : "read", pkt->req->getVaddr());

    if (iter == proc->busyDataCacheLineAddrs.end()) {
        panic("We should always find the address!!\n");
    }

    // profile the warp memory latency
    mem_fetch *mf = iter->second;
    DPRINTF(ShaderCoreAccess, "Looking for (%llu, %u)\n", (uint64_t)mf->get_pc(), mf->get_wid());
    map<pair<uint64_t, unsigned>, WarpMemRequest>::iterator wIter =
        proc->warpMemRequests.find(make_pair((uint64_t)mf->get_pc(), mf->get_wid()));
    assert(wIter != proc->warpMemRequests.end());
    bool done = wIter->second.requestFinish(curTick(), pkt->isRead());
    if (done) {
        proc->warpMemRequests.erase(wIter);
    }

    if (pkt->req->isLocked()) {
        panic("This code is very stale!");
//        if (pkt->isRead()) {
//            RequestPtr wreq = new Request();
//            Request::Flags flags = Request::LOCKED;
//            Addr pc = 0;
//            const int asid = 0;
//            wreq->setVirt(asid, pkt->req->getVaddr(), pkt->req->getSize(), flags, proc->masterId, pc);
//
//            assert(iter->second != NULL);
//            // NOTE: We shouldn't go through translation again!
//            if (proc->stallOnDCacheRetry) {
//                proc->stallOnAtomicQueue = true;
//                DPRINTF(ShaderCoreAccess, "[SC:%d] Pending write part of RMW to addr 0x%x\n", proc->id, pkt->req->getVaddr());
//                PendingReq *req = new PendingReq(wreq, BaseTLB::Write, iter->second);
//                proc->atomicQueue.push(req);
//            } else {
//                DPRINTF(ShaderCoreAccess, "[SC:%d] Sending write part of RMW to addr 0x%x\n", proc->id, pkt->req->getVaddr());
//                proc->accessVirtMem(wreq, iter->second, BaseTLB::Write);
//            }
//            return true;
//        } else {
//            // the write part of the RMW finished
//            DPRINTF(ShaderCoreAccess, "[SC:%d] Finished RMW to addr 0x%x\n", proc->id, pkt->req->getVaddr());
//            iter->second->do_atomic();
//        }
    } else if (pkt->isRead()) {
        // @TODO: If necessary, update operands/registers with data received
        // Information about this should be included in the pkt->senderState
        // @TODO: Use the memRequestHints to get this information
        ReadPacketBuffer* read_buffer = (ReadPacketBuffer*)pkt->senderState;
        DPRINTF(ShaderCoreAccess, "Got read_buffer %x with %d buffered reads\n", read_buffer, read_buffer->numBufferedReads());
        std::list<MemRequestHint*> coalesced_reads = read_buffer->getBufferedReads();
        std::list<MemRequestHint*>::iterator it;
        std::map<int,int> vectorReg;
        for (it = coalesced_reads.begin(); it != coalesced_reads.end();) {
            MemRequestHint* curr_hint = (*it);
            DPRINTF(ShaderCoreAccess, "Completed read of addr %x for thread ID:%d:%d\n", curr_hint->getAddr(), curr_hint->getWID(), curr_hint->getTID());
            const ptx_instruction* pI = curr_hint->getInst();
            ptx_thread_info* thread = curr_hint->getThread();

            unsigned vector_spec = pI->get_vector();
            // @TODO: We might need to get the destination when the read
            // is buffered rather than here
            const operand_info &dst = pI->dst();
            unsigned type = pI->get_type();
            unsigned offset_in_line = curr_hint->getAddr() - pkt->req->getVaddr();
            DPRINTF(ShaderCoreAccess, "offset is %d\n", offset_in_line);
            DPRINTF(ShaderCoreAccess, "Data is %d\n", *(int*)(pkt->getPtr<uint8_t>()+offset_in_line));
            // @TODO: Update register data
            // *Note: This replicates code in ld_impl (instructions.cc)
            ptx_reg_t register_data;
            memcpy(&register_data, (void*)(pkt->getPtr<uint8_t>() + offset_in_line), curr_hint->getSize());
            if (type == S16_TYPE || type == S32_TYPE) {
//                sign_extend(register_data, curr_hint->getSize()*8, dst);
            }
            if (!vector_spec) {
                thread->set_operand_value(dst, register_data, type, thread, pI);
            } else {
                // NOTE: The code below may be buggy. Does std::map[] always initialize int to 0?
                thread->set_reg(dst.vec_symbol(vectorReg[curr_hint->getTID()]), register_data);
                vectorReg[curr_hint->getTID()]++;
            }
            coalesced_reads.erase(it++);
            delete curr_hint;
        }
        assert(coalesced_reads.begin() == coalesced_reads.end() && coalesced_reads.size() == 0);
        delete read_buffer;
    } else if (pkt->isWrite()) {
        Addr line_addr = proc->addrToLine(pkt->req->getVaddr());
        writePackets[line_addr].remove(pkt);
        // If the write consists of multiple packets and there are more
        // to send, then the next packet in the series needs to be issued
        if (writePackets[line_addr].size()) {
            if (pkt->req) delete pkt->req;
            delete pkt;
            pkt = writePackets[line_addr].front();
            sendPkt(pkt);
            return true;
        } else {
            writePackets.erase(line_addr);
        }
    }
    // need to clear mshr so this can commit
    proc->busyDataCacheLineAddrs.erase(iter);
    DPRINTF(ShaderCoreAccess, "[SC:%d] Removing vaddr 0x%x from busy. Curr busy: %d\n", proc->id, proc->addrToLine(pkt->req->getVaddr()), proc->busyDataCacheLineAddrs.size());
    iter->second->set_reply();
    proc->shaderImpl->accept_ldst_unit_response(iter->second);

    if (pkt->req) delete pkt->req;
    delete pkt;
    return true;
}

void ShaderCore::SCDataPort::recvRetry() {
    assert(outDataPkts.size());

    proc->numDataCacheRetry++;

    PacketPtr pktToRetry = outDataPkts.front();
    DPRINTF(ShaderCoreAccess, "recvRetry got called, pkt: %p, vaddr: 0x%x\n", pktToRetry, pktToRetry->req->getVaddr());

    if (sendPkt(pktToRetry)) {
        outDataPkts.remove(pktToRetry);
        // If there are still packets on the retry list, signal to Ruby that
        // there should be a retry call for the next packet when possible
        proc->stallOnDCacheRetry = (outDataPkts.size() > 0);
        if (proc->stallOnDCacheRetry) {
            pktToRetry = outDataPkts.front();
            sendPkt(pktToRetry);
        }

        // THIS IS EXTREMELY STALE CODE
//        if (proc->stallOnAtomicQueue && !proc->scheduledTickEvent) {
//            DPRINTF(ShaderCoreAccess, "[SC:%d] Scheduling tick in recvRetry\n", proc->id);
//            proc->schedule(proc->tickEvent, curTick());
//            proc->scheduledTickEvent = true;
//        }
    } else {
        // Don't yet know how to handle this situation
//        panic("RecvRetry failed!");
    }
}

int ShaderCore::dataCacheResourceAvailable(Addr addr)
{
    map<Addr,mem_fetch *>::iterator iter = busyDataCacheLineAddrs.find(addrToLine(addr));
    return !stallOnDCacheRetry && (iter == busyDataCacheLineAddrs.end());
}

int ShaderCore::instCacheResourceAvailable(Addr addr)
{
    map<Addr,mem_fetch *>::iterator iter = busyInstCacheLineAddrs.find(addrToLine(addr));
    return iter == busyInstCacheLineAddrs.end();
}

void ShaderCore::tick()
{
    DPRINTF(ShaderCoreTick, "[SC:%d] tick\n", id);
    scheduledTickEvent = false;

    if (atomicQueue.empty()) {
        panic("Why is there nothing in the atomicQueue???\n");
    }

    if (stallOnDCacheRetry) {
        DPRINTF(ShaderCoreAccess, "[SC:%d] Stalled on retry, trying again later\n", id);
        return;
    }

    // THIS IS EXTREMELY STALE CODE...
    panic("THIS IS EXTREMELY STALE CODE");

//    DPRINTF(ShaderCoreAccess, "Emptying atomicQueue. %d total\n", atomicQueue.size());
//    assert(!atomicQueue.empty());
//    PendingReq *req = atomicQueue.front();
//    atomicQueue.pop();
//    if (!atomicQueue.empty()) {
//        schedule(tickEvent, curTick());
//        scheduledTickEvent = true;
//    } else {
//        DPRINTF(ShaderCoreAccess, "Finally done emptying atomic queue\n");
//        stallOnAtomicQueue = false;
//    }
//
//    DPRINTF(ShaderCoreAccess, "Checking req->Vaddr=0x%x\n", req->req->getVaddr());
//
//    if (req->mode == BaseTLB::Write) {
//        DPRINTF(ShaderCoreAccess, "[SC:%d] pendingReq is write\n", id);
//        accessVirtMem(req->req, req->mf, BaseTLB::Write);
//    } else if (req->req->isLocked()) {
//        DPRINTF(ShaderCoreAccess, "[SC:%d] pendingReq is rmw\n", id);
//        accessVirtMem(req->req, req->mf, BaseTLB::Read);
//    } else {
//        DPRINTF(ShaderCoreAccess, "[SC:%d] pendingReq is read\n", id);
//        accessVirtMem(req->req, req->mf, BaseTLB::Read);
//    }

}

void ShaderCore::finishTranslation(WholeTranslationState *state)
{
    if (state->getFault() != NoFault) {
//        state->getFault()->invoke(tc, NULL);
//        return;
        panic("Translation encountered fault (%s) for address 0x%x", state->getFault()->name(), state->mainReq->getVaddr());
    }
    PacketPtr pkt;
    if (state->mode == BaseTLB::Read) {
        pkt = new Packet(state->mainReq, MemCmd::ReadReq);
        pkt->allocate();
        if (pkt->req->isInstFetch()) {
            instPort.sendPkt(pkt);
        } else {
            pkt->senderState = (ReadPacketBuffer*)state->mainReq->getExtraData();
            dataPort.sendPkt(pkt);
        }
    } else if (state->mode == BaseTLB::Write) {
        // For each write packet hint for this cache line, we need to create
        // a new request and a new packet that will actually be sent to Ruby
        Addr block_vaddr = state->mainReq->getVaddr();
        std::list<MemRequestHint*> packet_hints = writePacketHints[block_vaddr];
        assert(packet_hints.size());
        std::list<MemRequestHint*>::iterator it;
        for (it = packet_hints.begin(); it != packet_hints.end(); ) {
            Addr offset = (*it)->getAddr() - block_vaddr;
            RequestPtr pkt_req = new Request();
            pkt_req->setVirt(state->mainReq->getAsid(), block_vaddr + offset, (*it)->getSize(), state->mainReq->getFlags(), masterId, state->mainReq->getPC());
            pkt_req->setPaddr(state->mainReq->getPaddr() + offset);
            pkt = new Packet(pkt_req, MemCmd::WriteReq);
            pkt->allocate();
            pkt->setData((*it)->getData());
            packet_hints.erase(it++);
            dataPort.writePackets[addrToLine(block_vaddr)].push_back(pkt);
        }
        assert(!packet_hints.size());
        writePacketHints.erase(block_vaddr);
        pkt = dataPort.writePackets[addrToLine(block_vaddr)].front();
        dataPort.sendPkt(pkt);
        if (state->mainReq) delete state->mainReq;
    } else {
        panic("Finished translation of unknown mode: %d\n", state->mode);
    }
    delete state;
}

int ShaderCore::readTiming (Addr addr, size_t size, mem_fetch *mf)
{
    if (!dataCacheResourceAvailable(addr)) {
        DPRINTF(ShaderCoreAccess, "Access of %lld bytes failed for addr 0x%llx\n", size, addr);
        return 1;
    }

    // According to the CUDA spec: coalesced reads are sized 32, 64 or 128B
    // and they must be address-aligned on their size
    assert((size == 32) || (size == 64) || (size == 128));
    assert(!(addr % size));

    // Currently, we don't handle reads that stride across multiple lines
    // @TODO: This will require issuing multiple translation requests to the DTB
    assert(addr + size <= addrToLine(addr) + spa->getRubySystem()->getBlockSizeBytes());

    // Mark this as the begin time for the memory request in the mf warp inst object
    // This is just for profiling and should probably be put after the translation
    DPRINTF(ShaderCoreAccess, "Adding (%llu, %u)\n", (uint64_t)mf->get_pc(), mf->get_wid());
    WarpMemRequest& warpReq = warpMemRequests[make_pair(mf->get_pc(), mf->get_wid())];
    warpReq.addRequest(curTick());

    // For each buffered read landing in this block, check the warp ID to verify
    // that this coalesced read should include the buffered read
    ReadPacketBuffer* read_buffer = new ReadPacketBuffer(mf);
    Addr base_addr = addr;
    for (Addr offset = 0; offset < size;) {
        if (memReadHints.find(base_addr + offset) != memReadHints.end()) {
            std::list<MemRequestHint*> addr_hints = memReadHints[base_addr + offset];
            std::list<MemRequestHint*>::iterator it = addr_hints.begin();
            for (; it != addr_hints.end(); ++it) {
                // If this buffered read is the correct warp/thread ID, then
                // add it to the buffered, coalesced request (*Note: This is
                // how coalescing is controlled)
                MemRequestHint* curr_hint = (*it);
                if (mf->get_wid() == curr_hint->getWID()) {
                    DPRINTF(ShaderCoreAccess, "[TH:%d] Found hint: %x\n", curr_hint->getTID(), base_addr + offset);
                    read_buffer->addBufferedRead(curr_hint);
                    memReadHints[base_addr + offset].remove(curr_hint);
                }
            }
            if (memReadHints[base_addr + offset].size() == 0) {
                memReadHints.erase(base_addr + offset);
            }
        }
        offset++;
    }
    if (!read_buffer->numBufferedReads()) {
        warn_once("No buffered reads found for this read! Sending anyway");
        // @TODO: Decide if we should do something else here
        //assert(read_buffer->numBufferedReads());
    }

    DPRINTF(ShaderCoreAccess, "[SC:%d] Reading %d bytes at virtual address 0x%x for %d bufferedRead(s), pktBuf: %x\n", id, size, addr, read_buffer->numBufferedReads(), read_buffer);
    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;

    req->setVirt(asid, addr, size, flags, masterId, pc);
    req->setExtraData((uint64_t)read_buffer);
    accessVirtMem(req, mf, BaseTLB::Read);

    return 0;
}

int ShaderCore::writeTiming(Addr addr, size_t size, mem_fetch *mf)
{
    if (!dataCacheResourceAvailable(addr)) {
        return 1;
    }

    // According to the CUDA spec: coalesced writes are sized 32, 64 or 128B
    // and they must be address-aligned on their size
    assert((size == 32) || (size == 64) || (size == 128));
    assert(!(addr % size));

    // Currently, we don't handle writes that stride across multiple lines
    // @TODO: This will require issuing multiple translation requests to the DTB
    assert(addr + size <= addrToLine(addr) + spa->getRubySystem()->getBlockSizeBytes());

    // Mark this as the begin time for the memory request in the mf warp inst object
    // This is just for profiling and should probably be put after the translation
    DPRINTF(ShaderCoreAccess, "Adding (%llu, %u)\n", (uint64_t)mf->get_pc(), mf->get_wid());
    WarpMemRequest& warpReq = warpMemRequests[make_pair(mf->get_pc(), mf->get_wid())];
    warpReq.addRequest(curTick());

    // @TODO: This should be extracted as a separate function: coalesceBlockHints()
    uint8_t* block_write_data = new uint8_t[size];
    Addr base_addr = addr;
    Addr write_start_addr = base_addr - 1;
    Addr write_end_addr = base_addr - 1;
    int chunks_to_write = 0;
    for (Addr offset = 0; offset < size; ) {
        std::list<MemRequestHint*> addr_hints = memWriteHints[base_addr + offset];
        Addr offset_advance = 1;
        if (addr_hints.size()) {
            std::list<MemRequestHint*>::iterator it = addr_hints.begin();
            assert(addr_hints.size() <= 1);
            for (; it != addr_hints.end(); ++it) {
                MemRequestHint* curr_hint = (*it);
                assert(curr_hint->isWrite());
                memcpy(&block_write_data[offset], curr_hint->getData(), curr_hint->getSize());
                if (write_start_addr < base_addr) {
                    write_start_addr = base_addr + offset;
                } else {
                    if (write_end_addr >= base_addr) {
                        size_t packet_size = write_end_addr - write_start_addr + 1;
                        MemRequestHint* packet_hint = new MemRequestHint(write_start_addr, packet_size, BaseTLB::Write);
                        packet_hint->addData(packet_size, &block_write_data[write_start_addr - base_addr]);
                        writePacketHints[addr].push_back(packet_hint);
                        write_start_addr = base_addr + offset;
                        write_end_addr = base_addr - 1;
                        chunks_to_write++;

                        DPRINTF(ShaderCoreAccess, "Adding (%llu, %u)\n", (uint64_t)mf->get_pc(), mf->get_wid());
                        WarpMemRequest& warpReq = warpMemRequests[make_pair(mf->get_pc(), mf->get_wid())];
                        warpReq.addRequest(curTick());
                    }
                }
                memWriteHints[base_addr + offset].remove(curr_hint);
                // Check that other writes were not partially overlapped with
                // this chunk
                for (Addr within_chunk = offset + 1; within_chunk < curr_hint->getSize(); ++within_chunk) {
                    assert(memWriteHints[base_addr + within_chunk].size() == 0);
                }
                offset_advance = curr_hint->getSize();
                delete curr_hint;
            }
        } else {
            if (write_start_addr >= base_addr) {
                if (write_end_addr < base_addr) {
                    write_end_addr = base_addr + offset - 1;
                }
            }
        }
        if (memWriteHints[base_addr + offset].size() == 0) {
            memWriteHints.erase(base_addr + offset);
        }
        offset += offset_advance;
    }
    size_t packet_size;
    if (write_start_addr < base_addr) {
        // If no data found for this write: Send an empty request to later
        // signal back to the LDST unit that the request is complete
        write_start_addr = base_addr;
        write_end_addr = base_addr;
        packet_size = 0;
    } else {
        // If the data runs to the end of the line, setup the end address
        if (write_end_addr < base_addr) {
            write_end_addr = base_addr + size - 1;
        }
        packet_size = write_end_addr - write_start_addr + 1;
    }
    MemRequestHint* packet_hint = new MemRequestHint(write_start_addr, packet_size, BaseTLB::Write);
    packet_hint->addData(packet_size, &block_write_data[write_start_addr - base_addr]);
    writePacketHints[addr].push_back(packet_hint);
    chunks_to_write++;
    DPRINTF(ShaderCoreAccess, "[SC:%d] Writing %d bytes virtual address 0x%x in %d chunk(s)\n", id, size, addr, chunks_to_write);
    delete[] block_write_data;

    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;

    req->setVirt(asid, addr, size, flags, masterId, pc);

    accessVirtMem(req, mf, BaseTLB::Write);
    return 0;
}

int ShaderCore::atomicRMW(Addr addr, size_t size,  mem_fetch *mf)
{
    panic("ATOMIC RMW NEEDS WORK!");
//    if (!dataCacheResourceAvailable(addr)) {
//        return 1;
//    }
//    DPRINTF(ShaderCoreAccess, "[SC:%d] AtomicRMW %d bytes virtual address 0x%x\n", id, size, addr);
//
//    RequestPtr req = new Request();
//    Request::Flags flags = Request::LOCKED;
//    Addr pc = 0;
//    const int asid = 0;
//
//    req->setVirt(asid, addr, size, flags, masterId, pc);
//
//    accessVirtMem(req, mf, BaseTLB::Read);
//
    return 0;
}

inline Addr ShaderCore::addrToLine(Addr a)
{
    unsigned int maskBits = spa->getRubySystem()->getBlockSizeBits();
    return a & (((uint64_t)-1) << maskBits);
}

void ShaderCore::accessVirtMem(RequestPtr req, mem_fetch *mf, BaseTLB::Mode mode)
{
    assert(mf != NULL);

    if (req->isLocked() && mode == BaseTLB::Write) {
        // skip below
        panic("WHY IS THIS BEING EXECUTED?");
    }

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
    DataTranslation<ShaderCore*> *translation
            = new DataTranslation<ShaderCore*>(this, state);

    if (req->isInstFetch()) {
        assert(busyInstCacheLineAddrs.find(addrToLine(req->getVaddr())) == busyInstCacheLineAddrs.end());
        busyInstCacheLineAddrs[addrToLine(req->getVaddr())] = mf;
        itb->beginTranslateTiming(req, translation, mode);
    } else {
        assert(busyDataCacheLineAddrs.find(addrToLine(req->getVaddr())) == busyDataCacheLineAddrs.end());
        busyDataCacheLineAddrs[addrToLine(req->getVaddr())] = mf;
        dtb->beginTranslateTiming(req, translation, mode);
    }
}

bool ShaderCore::SCDataPort::sendPkt(PacketPtr pkt)
{
    // @TODO: Throttle the number of packets that can be sent per cycle here?
    DPRINTF(ShaderCoreAccess, "[SC:%d] Sending %s of %d bytes to vaddr: 0x%x, paddr: 0x%x, busy: %d\n", proc->id, (pkt->isWrite()) ? "write" : "read", pkt->getSize(), pkt->req->getVaddr(), pkt->getAddr(), proc->busyDataCacheLineAddrs.size());
    if (!sendTimingReq(pkt)) {
        DPRINTF(ShaderCoreAccess, "dataPort.sendPkt failed. pkt: %p vaddr: 0x%x\n", pkt, pkt->req->getVaddr());
        proc->stallOnDCacheRetry = true;
        if (pkt != outDataPkts.front()) {
            outDataPkts.push_back(pkt);
        }
        DPRINTF(ShaderCoreAccess, "Busy waiting requests: %d\n", outDataPkts.size());
        return false;
    }
    proc->numDataCacheRequests++;
    return true;
}

void
ShaderCore::MemRequestHint::addData(size_t _size, const void* _data, unsigned offset)
{
    assert(offset + _size <= size);
    memcpy(&data[offset], _data, _size);
}

void
ShaderCore::addWriteHint(Addr addr, size_t size, const void* data)
{
    DPRINTF(ShaderCoreAccess, "[SC:%d] Received write hint, addr: 0x%x, size: %d\n", id, addr, size);
    DPRINTFR(ShaderMemTrace, ">>%llu [%d] W 0x%x %d\n", curTick(), id, addr, size);
    std::list<MemRequestHint*>::iterator it = memWriteHints[addr].begin();
    if (it == memWriteHints[addr].end()) {
        MemRequestHint* hint = new MemRequestHint(addr, size, BaseTLB::Write);
        hint->addData(size, data);
        hint->tick = curTick();
        memWriteHints[addr].push_back(hint);
    } else {
        assert(memWriteHints[addr].size() == 1);
        MemRequestHint* hint = *it;
        if (size != hint->getSize()) {
            panic("Need to implement partial write overlap handling");
        }
        if (curTick() >= hint->tick) {
            hint->addData(size, data);
            hint->tick = curTick();
        }
    }
}

void
ShaderCore::addReadHint(Addr addr, size_t size, const void* data, ptx_thread_info *thd, const ptx_instruction *pI)
{
    DPRINTF(ShaderCoreAccess, "[SC:%d] Received read hint, src_line: %d, pc: %d, addr: 0x%x, size: %d, ID:%d:%d\n", id, pI->source_line(), pI->get_PC(), addr, size, thd->get_hw_wid(), thd->get_hw_tid());
    DPRINTFR(ShaderMemTrace, ">>%llu [%d] R 0x%x %d\n", curTick(), id, addr, size);
    MemRequestHint* hint = new MemRequestHint(addr, size, BaseTLB::Read, thd->get_hw_wid(), thd->get_hw_tid());
    hint->tick = curTick();
    hint->setThread(thd);
    hint->setInst(pI);

    memReadHints[addr].push_back(hint);
}

void
ShaderCore::icacheFetch(Addr addr, mem_fetch *mf)
{
    Addr line_addr = addrToLine(addr);
    DPRINTF(ShaderCoreFetch, "[SC:%d] Received fetch request, addr: 0x%x, size: %d, line: 0x%x\n", id, addr, mf->size(), line_addr);
    if (!instCacheResourceAvailable(addr)) {
        // Executed when there is a duplicate inst fetch request outstanding
        panic("This code shouldn't be executed?");
        return;
    }

    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = (Addr)mf->get_pc();
    const int asid = 0;

    req->setVirt(asid, line_addr, mf->size(), flags, masterId, pc);
    req->setFlags(Request::INST_FETCH);

    accessVirtMem(req, mf, BaseTLB::Read);
}

bool
ShaderCore::SCInstPort::sendPkt(PacketPtr pkt)
{
    DPRINTF(ShaderCoreFetch, "[SC:%d] Sending %s of %d bytes to vaddr: 0x%x, paddr: 0x%x, busy: %d\n", proc->id, (pkt->isWrite()) ? "write" : "read", pkt->getSize(), pkt->req->getVaddr(), pkt->getAddr(), proc->busyInstCacheLineAddrs.size());
    if (!sendTimingReq(pkt)) {
        DPRINTF(ShaderCoreFetch, "instPort.sendPkt failed. pkt: %p vaddr: 0x%x\n", pkt, pkt->req->getVaddr());
        proc->stallOnICacheRetry = true;
        if (pkt != outInstPkts.front()) {
            outInstPkts.push_back(pkt);
        }
        DPRINTF(ShaderCoreFetch, "Busy waiting requests: %d\n", outInstPkts.size());
        return false;
    }
    proc->numInstCacheRequests++;
    return true;
}

bool
ShaderCore::SCInstPort::recvTimingResp(PacketPtr pkt)
{
    assert(pkt->req->isInstFetch());
    map<Addr,mem_fetch *>::iterator iter = proc->busyInstCacheLineAddrs.find(proc->addrToLine(pkt->req->getVaddr()));

    DPRINTF(ShaderCoreFetch, "[SC:%d] Finished fetch on vaddr 0x%x\n", proc->id, pkt->req->getVaddr());

    if (iter == proc->busyInstCacheLineAddrs.end()) {
        panic("We should always find the address!!\n");
    }

    proc->shaderImpl->accept_fetch_response(iter->second);

    proc->busyInstCacheLineAddrs.erase(iter);

    if (pkt->req) delete pkt->req;
    delete pkt;
    return true;
}

void
ShaderCore::SCInstPort::recvRetry()
{
    assert(outInstPkts.size());

    proc->numInstCacheRetry++;

    PacketPtr pktToRetry = outInstPkts.front();
    DPRINTF(ShaderCoreFetch, "recvRetry got called, pkt: %p, vaddr: 0x%x\n", pktToRetry, pktToRetry->req->getVaddr());

    if (sendPkt(pktToRetry)) {
        outInstPkts.remove(pktToRetry);
        // If there are still packets on the retry list, signal to Ruby that
        // there should be a retry call for the next packet when possible
        proc->stallOnICacheRetry = (outInstPkts.size() > 0);
        if (proc->stallOnICacheRetry) {
            pktToRetry = outInstPkts.front();
            sendPkt(pktToRetry);
        }

        // THIS IS EXTREMELY STALE CODE
//        if (proc->stallOnAtomicQueue && !proc->scheduledTickEvent) {
//            DPRINTF(ShaderCoreAccess, "[SC:%d] Scheduling tick in recvRetry\n", proc->id);
//            proc->schedule(proc->tickEvent, curTick());
//            proc->scheduledTickEvent = true;
//        }
    } else {
        // Don't yet know how to handle this situation
//        panic("RecvRetry failed!");
    }
}


bool
ShaderCore::executeMemOp(const warp_inst_t &inst)
{
    assert(inst.space.get_type() == global_space ||
           inst.space.get_type() == const_space);
    assert(inst.valid());

    // for debugging
    bool completed = false;

    for (int lane=0; lane<warpSize; lane++) {
        if (inst.active(lane)) {
            Addr addr = inst.get_addr(lane);
            Addr pc = (Addr)inst.pc;
            int size = inst.data_size;
            assert(size >= 1 && size <= 8);
            size *= inst.vectorLength;
            assert(size <= 16);


            DPRINTF(ShaderCoreAccess, "Got addr 0x%llx\n", addr);
            if (inst.space.get_type() == const_space) {
                DPRINTF(ShaderCoreAccess, "Is const!!\n");
            }

            Request::Flags flags;
            const int asid = 0;
            RequestPtr req = new Request(asid, addr, size, flags, masterId, pc,
                                         id, inst.warp_id());

            PacketPtr pkt;
            if (inst.is_load()) {
                pkt = new Packet(req, MemCmd::ReadReq);
                pkt->allocate();
                // Since only loads return to the ShaderCore
                pkt->senderState = new SenderState(inst);
            } else if (inst.is_store()) {
                pkt = new Packet(req, MemCmd::WriteReq);
                pkt->allocate();
                uint8_t data[16]; // Can't have more than 16 bytes
                shaderImpl->readRegister(inst, warpSize, lane, (char*)data);
                // assert(inst.vectorLength == regs);
                DPRINTF(ShaderCoreAccess, "Storing %d\n", *(int*)data);
                pkt->setData((uint8_t*)data);
            } else {
                panic("Unsupported instruction type\n");
            }

            if (!lsqPorts[lane]->sendTimingReq(pkt)) {
                // NOTE: This should fail early. If executeMemOp fails after
                // some, but not all, of the requests have been sent the
                // behavior is undefined.
                assert(!completed);
                return true;
            } else {
                completed = true;
            }
        }
    }

    // Return that there should not be a pipeline stall
    return false;
}

bool
ShaderCore::LSQPort::recvTimingResp(PacketPtr pkt)
{
    DPRINTF(ShaderCoreAccess, "Got a response for lane %d address 0x%llx\n",
            idx, pkt->req->getVaddr());

    assert(pkt->isRead());

    uint8_t data[16];
    assert(pkt->getSize() <= sizeof(data));

    ShaderCore &sc = dynamic_cast<ShaderCore&>(owner);
    warp_inst_t &inst = ((SenderState*)pkt->senderState)->inst;

    if (!sc.shaderImpl->ldst_unit_wb_inst(inst)) {
        // Writeback register is occupied, stall
        assert(sc.writebackBlocked < 0);
        sc.writebackBlocked = idx;
        return false;
    }

    pkt->writeData(data);
    DPRINTF(ShaderCoreAccess, "Loaded data %d\n", *(int*)data);
    sc.shaderImpl->writeRegister(inst, sc.warpSize, idx, (char*)data);

    delete pkt->senderState;
    delete pkt->req;
    delete pkt;

    return true;
}

void
ShaderCore::LSQPort::recvRetry()
{
    panic("Not sure how to respond to a recvRetry...");
}

Tick
ShaderCore::SCInstPort::recvAtomic(PacketPtr pkt)
{
    panic("Not sure how to recvAtomic");
    return 0;
}

void
ShaderCore::SCInstPort::recvFunctional(PacketPtr pkt)
{
    panic("Not sure how to recvFunctional");
}

void
ShaderCore::writebackClear()
{
    if (writebackBlocked >= 0) lsqPorts[writebackBlocked]->sendRetry();
    writebackBlocked = -1;
}

ShaderCore *ShaderCoreParams::create() {
    return new ShaderCore(this);
}

void
ShaderCore::regStats()
{
    numLocalLoads
        .name(name() + ".local_loads")
        .desc("Number of loads from local space")
        ;
    numLocalStores
        .name(name() + ".local_stores")
        .desc("Number of stores to local space")
        ;
    numSharedLoads
        .name(name() + ".shared_loads")
        .desc("Number of loads from shared space")
        ;
    numSharedStores
        .name(name() + ".shared_stores")
        .desc("Number of stores to shared space")
        ;
    numParamKernelLoads
        .name(name() + ".param_kernel_loads")
        .desc("Number of loads from kernel parameter space")
        ;
    numParamLocalLoads
        .name(name() + ".param_local_loads")
        .desc("Number of loads from local parameter space")
        ;
    numParamLocalStores
        .name(name() + ".param_local_stores")
        .desc("Number of stores to local parameter space")
        ;
    numConstLoads
        .name(name() + ".const_loads")
        .desc("Number of loads from constant space")
        ;
    numTexLoads
        .name(name() + ".tex_loads")
        .desc("Number of loads from texture space")
        ;
    numGlobalLoads
        .name(name() + ".global_loads")
        .desc("Number of loads from global space")
        ;
    numGlobalStores
        .name(name() + ".global_stores")
        .desc("Number of stores to global space")
        ;
    numSurfLoads
        .name(name() + ".surf_loads")
        .desc("Number of loads from surface space")
        ;
    numGenericLoads
        .name(name() + ".generic_loads")
        .desc("Number of loads from generic spaces (global, shared, local)")
        ;
    numGenericStores
        .name(name() + ".generic_stores")
        .desc("Number of stores to generic spaces (global, shared, local)")
        ;
    numDataCacheRequests
        .name(name() + ".coalesced_data_cache_requests")
        .desc("Number of coalesced data cache requests sent")
        ;
    numDataCacheRetry
        .name(name() + ".coalesced_data_cache_retries")
        .desc("Number of coalesced data cache retries")
        ;
    numInstCacheRequests
        .name(name() + ".inst_cache_requests")
        .desc("Number of instruction cache requests sent")
        ;
    numInstCacheRetry
        .name(name() + ".inst_cache_retries")
        .desc("Number of instruction cache retries")
        ;
    instCounts
        .init(8)
        .name(name() + ".inst_counts")
        .desc("Inst counts: 1: ALU, 2: MAD, 3: CTRL, 4: SFU, 5: MEM, 6: TEX, 7: NOP")
        ;
}

void
ShaderCore::record_ld(memory_space_t space)
{
    switch(space.get_type()) {
    case local_space: numLocalLoads++; break;
    case shared_space: numSharedLoads++; break;
    case param_space_kernel: numParamKernelLoads++; break;
    case param_space_local: numParamLocalLoads++; break;
    case const_space: numConstLoads++; break;
    case tex_space: numTexLoads++; break;
    case surf_space: numSurfLoads++; break;
    case global_space: numGlobalLoads++; break;
    case generic_space: numGenericLoads++; break;
    case param_space_unclassified:
    case undefined_space:
    case reg_space:
    case instruction_space:
    default:
        panic("Load from invalid space: %d!", space.get_type());
        break;
    }
}

void
ShaderCore::record_st(memory_space_t space)
{
    switch(space.get_type()) {
    case local_space: numLocalStores++; break;
    case shared_space: numSharedStores++; break;
    case param_space_local: numParamLocalStores++; break;
    case global_space: numGlobalStores++; break;
    case generic_space: numGenericStores++; break;

    case param_space_kernel:
    case const_space:
    case tex_space:
    case surf_space:
    case param_space_unclassified:
    case undefined_space:
    case reg_space:
    case instruction_space:
    default:
        panic("Store to invalid space: %d!", space.get_type());
        break;
    }
}

void
ShaderCore::record_inst(int inst_type)
{
    instCounts[inst_type]++;
}

void
ShaderCore::record_block_issue(unsigned hw_cta_id)
{
    assert(!shaderCTAActive[hw_cta_id]);
    shaderCTAActive[hw_cta_id] = true;
    shaderCTAActiveStats[hw_cta_id].push_back(curTick());
}

void
ShaderCore::record_block_commit(unsigned hw_cta_id)
{
    assert(shaderCTAActive[hw_cta_id]);
    shaderCTAActive[hw_cta_id] = false;
    shaderCTAActiveStats[hw_cta_id].push_back(curTick());
}

void ShaderCore::printCTAStats(std::ostream& out)
{
    std::map<unsigned, std::vector<Tick> >::iterator iter;
    std::vector<Tick>::iterator times;
    for (iter = shaderCTAActiveStats.begin(); iter != shaderCTAActiveStats.end(); iter++) {
        unsigned cta_id = iter->first;
        out << id << ", " << cta_id << ", ";
        for (times = shaderCTAActiveStats[cta_id].begin(); times != shaderCTAActiveStats[cta_id].end(); times++) {
            out << *times << ", ";
        }
        out << curTick() << "\n";
    }
}
