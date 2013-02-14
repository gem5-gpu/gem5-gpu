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
    MemObject(p),
    instPort(name() + ".inst_port", this), _params(p),
    masterId(p->sys->getMasterId(name())), id(p->id),
    itb(p->itb), spa(p->spa)
{
    writebackBlocked = -1; // Writeback is not blocked

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

    activeCTAs = 0;

    DPRINTF(ShaderCore, "[SC:%d] Created shader core\n", id);
}

BaseMasterPort&
ShaderCore::getMasterPort(const std::string &if_name, PortID idx)
{
    if (if_name == "inst_port") {
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

void ShaderCore::initialize()
{
    shaderImpl = spa->getTheGPU()->get_shader(id);
}

int ShaderCore::instCacheResourceAvailable(Addr addr)
{
    map<Addr,mem_fetch *>::iterator iter = busyInstCacheLineAddrs.find(addrToLine(addr));
    return iter == busyInstCacheLineAddrs.end();
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
        assert(pkt->req->isInstFetch());
        instPort.sendPkt(pkt);
    } else {
        panic("Finished translation of unknown mode: %d\n", state->mode);
    }
    delete state;
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

    assert(req->isInstFetch());

    assert(busyInstCacheLineAddrs.find(addrToLine(req->getVaddr())) == busyInstCacheLineAddrs.end());
    busyInstCacheLineAddrs[addrToLine(req->getVaddr())] = mf;
    itb->beginTranslateTiming(req, translation, mode);
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

                if (inst.is_load()) {
                    delete pkt->senderState;
                }
                delete pkt->req;
                delete pkt;

                // Return that there is a pipeline stall
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

    activeCycles
        .name(name() + ".activeCycles")
        .desc("Number of cycles this shader was executing a CTA")
        ;
    notStalledCycles
        .name(name() + ".notStalledCycles")
        .desc("Number of cycles this shader was actually executing at least one instance")
        ;
    instInstances
        .name(name() + ".instInstances")
        .desc("Total instructions executed by all PEs in the core")
        ;
    instPerCycle
        .name(name() + ".instPerCycle")
        .desc("Instruction instances per cycle")
        ;

    instPerCycle = instInstances / activeCycles;
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

    // if not nop
    if (inst_type != 7) {
        instInstances++;
        if (curCycle() != lastActiveCycle) {
            lastActiveCycle = curCycle();
            notStalledCycles++;
        }
    }
}

void
ShaderCore::record_block_issue(unsigned hw_cta_id)
{
    assert(!shaderCTAActive[hw_cta_id]);
    shaderCTAActive[hw_cta_id] = true;
    shaderCTAActiveStats[hw_cta_id].push_back(curTick());

    if (activeCTAs == 0) {
        beginActiveCycle = curCycle();
    }
    activeCTAs++;
}

void
ShaderCore::record_block_commit(unsigned hw_cta_id)
{
    assert(shaderCTAActive[hw_cta_id]);
    shaderCTAActive[hw_cta_id] = false;
    shaderCTAActiveStats[hw_cta_id].push_back(curTick());

    activeCTAs--;
    if (activeCTAs == 0) {
        activeCycles += curCycle() - beginActiveCycle;
    }
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
