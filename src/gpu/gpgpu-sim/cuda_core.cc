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

#include "cpu/translation.hh"
#include "debug/CudaCore.hh"
#include "debug/CudaCoreAccess.hh"
#include "debug/CudaCoreFetch.hh"
#include "debug/CudaCoreMemTrace.hh"
#include "debug/CudaCoreTick.hh"
#include "gpu/gpgpu-sim/cuda_core.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "mem/page_table.hh"
#include "params/CudaCore.hh"
#include "sim/system.hh"

using namespace std;

CudaCore::CudaCore(const Params *p) :
    MemObject(p), instPort(name() + ".inst_port", this), _params(p),
    dataMasterId(p->sys->getMasterId(name() + ".data")),
    instMasterId(p->sys->getMasterId(name() + ".inst")), id(p->id),
    itb(p->itb), cudaGPU(p->gpu)
{
    writebackBlocked = -1; // Writeback is not blocked

    stallOnICacheRetry = false;

    cudaGPU->registerCudaCore(this);

    warpSize = cudaGPU->getWarpSize();

    signalKernelFinish = false;

    if (p->port_lsq_port_connection_count != warpSize) {
        panic("Shader core lsq_port size != to warp size\n");
    }

    // create the ports
    for (int i = 0; i < warpSize; ++i) {
        lsqPorts.push_back(new LSQPort(csprintf("%s-lsqPort%d", name(), i),
                                    this, i));
    }

    activeCTAs = 0;

    DPRINTF(CudaCore, "[SC:%d] Created CUDA core\n", id);
}

CudaCore::~CudaCore()
{
    for (int i = 0; i < warpSize; ++i) {
        delete lsqPorts[i];
    }
    lsqPorts.clear();
}

BaseMasterPort&
CudaCore::getMasterPort(const std::string &if_name, PortID idx)
{
    if (if_name == "inst_port") {
        return instPort;
    } else if (if_name == "lsq_port") {
        if (idx >= static_cast<PortID>(lsqPorts.size())) {
            panic("CudaCore::getMasterPort: unknown index %d\n", idx);
        }
        return *lsqPorts[idx];
    } else {
        return MemObject::getMasterPort(if_name, idx);
    }
}

void
CudaCore::unserialize(Checkpoint *cp, const std::string &section)
{
    // Intentionally left blank to keep from trying to read shader header from
    // checkpoint files. Allows for restore into any number of shader cores.
    // NOTE: Cannot checkpoint during kernels
}

void CudaCore::initialize()
{
    shaderImpl = cudaGPU->getTheGPU()->get_shader(id);
}

int CudaCore::instCacheResourceAvailable(Addr addr)
{
    map<Addr,mem_fetch *>::iterator iter = busyInstCacheLineAddrs.find(addrToLine(addr));
    return iter == busyInstCacheLineAddrs.end();
}

void CudaCore::finishTranslation(WholeTranslationState *state)
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

inline Addr CudaCore::addrToLine(Addr a)
{
    unsigned int maskBits = cudaGPU->getRubySystem()->getBlockSizeBits();
    return a & (((uint64_t)-1) << maskBits);
}

void CudaCore::accessVirtMem(RequestPtr req, mem_fetch *mf, BaseTLB::Mode mode)
{
    assert(mf != NULL);

    if (req->isLocked() && mode == BaseTLB::Write) {
        // skip below
        panic("WHY IS THIS BEING EXECUTED?");
    }

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
    DataTranslation<CudaCore*> *translation
            = new DataTranslation<CudaCore*>(this, state);

    assert(req->isInstFetch());

    assert(busyInstCacheLineAddrs.find(addrToLine(req->getVaddr())) == busyInstCacheLineAddrs.end());
    busyInstCacheLineAddrs[addrToLine(req->getVaddr())] = mf;
    itb->beginTranslateTiming(req, translation, mode);
}

void
CudaCore::icacheFetch(Addr addr, mem_fetch *mf)
{
    Addr line_addr = addrToLine(addr);
    DPRINTF(CudaCoreFetch, "[SC:%d] Received fetch request, addr: 0x%x, size: %d, line: 0x%x\n", id, addr, mf->size(), line_addr);
    if (!instCacheResourceAvailable(addr)) {
        // Executed when there is a duplicate inst fetch request outstanding
        panic("This code shouldn't be executed?");
        return;
    }

    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = (Addr)mf->get_pc();
    const int asid = 0;

    req->setVirt(asid, line_addr, mf->size(), flags, instMasterId, pc);
    req->setFlags(Request::INST_FETCH);

    accessVirtMem(req, mf, BaseTLB::Read);
}

bool
CudaCore::SCInstPort::sendPkt(PacketPtr pkt)
{
    DPRINTF(CudaCoreFetch, "[SC:%d] Sending %s of %d bytes to vaddr: 0x%x, paddr: 0x%x, busy: %d\n", core->id, (pkt->isWrite()) ? "write" : "read", pkt->getSize(), pkt->req->getVaddr(), pkt->getAddr(), core->busyInstCacheLineAddrs.size());
    if (!sendTimingReq(pkt)) {
        DPRINTF(CudaCoreFetch, "instPort.sendPkt failed. pkt: %p vaddr: 0x%x\n", pkt, pkt->req->getVaddr());
        core->stallOnICacheRetry = true;
        if (pkt != outInstPkts.front()) {
            outInstPkts.push_back(pkt);
        }
        DPRINTF(CudaCoreFetch, "Busy waiting requests: %d\n", outInstPkts.size());
        return false;
    }
    core->numInstCacheRequests++;
    return true;
}

bool
CudaCore::SCInstPort::recvTimingResp(PacketPtr pkt)
{
    assert(pkt->req->isInstFetch());
    map<Addr,mem_fetch *>::iterator iter = core->busyInstCacheLineAddrs.find(core->addrToLine(pkt->req->getVaddr()));

    DPRINTF(CudaCoreFetch, "[SC:%d] Finished fetch on vaddr 0x%x\n", core->id, pkt->req->getVaddr());

    if (iter == core->busyInstCacheLineAddrs.end()) {
        panic("We should always find the address!!\n");
    }

    core->shaderImpl->accept_fetch_response(iter->second);

    core->busyInstCacheLineAddrs.erase(iter);

    if (pkt->req) delete pkt->req;
    delete pkt;
    return true;
}

void
CudaCore::SCInstPort::recvRetry()
{
    assert(outInstPkts.size());

    core->numInstCacheRetry++;

    PacketPtr pktToRetry = outInstPkts.front();
    DPRINTF(CudaCoreFetch, "recvRetry got called, pkt: %p, vaddr: 0x%x\n", pktToRetry, pktToRetry->req->getVaddr());

    if (sendPkt(pktToRetry)) {
        outInstPkts.remove(pktToRetry);
        // If there are still packets on the retry list, signal to Ruby that
        // there should be a retry call for the next packet when possible
        core->stallOnICacheRetry = (outInstPkts.size() > 0);
        if (core->stallOnICacheRetry) {
            pktToRetry = outInstPkts.front();
            sendPkt(pktToRetry);
        }
    } else {
        // Don't yet know how to handle this situation
//        panic("RecvRetry failed!");
    }
}

bool
CudaCore::executeMemOp(const warp_inst_t &inst)
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


            DPRINTF(CudaCoreAccess, "Got addr 0x%llx\n", addr);
            if (inst.space.get_type() == const_space) {
                DPRINTF(CudaCoreAccess, "Is const!!\n");
            }

            Request::Flags flags;
            const int asid = 0;
            RequestPtr req = new Request(asid, addr, size, flags, dataMasterId,
                                         pc, id, inst.warp_id());

            PacketPtr pkt;
            if (inst.is_load()) {
                pkt = new Packet(req, MemCmd::ReadReq);
                pkt->allocate();
                // Since only loads return to the CudaCore
                pkt->senderState = new SenderState(inst);
            } else if (inst.is_store()) {
                pkt = new Packet(req, MemCmd::WriteReq);
                pkt->allocate();
                pkt->setData(inst.get_data(lane));
            } else {
                panic("Unsupported instruction type\n");
            }

            if (!lsqPorts[lane]->sendTimingReq(pkt)) {
                // NOTE: This should fail early. If executeMemOp fails after
                // some, but not all, of the requests have been sent the
                // behavior is undefined.
                if (completed) {
                    panic("Should never fail after first accepted lane");
                }

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
CudaCore::LSQPort::recvTimingResp(PacketPtr pkt)
{
    DPRINTF(CudaCoreAccess, "Got a response for lane %d address 0x%llx\n",
            idx, pkt->req->getVaddr());

    if (pkt->isFlush()) {
        DPRINTF(CudaCoreAccess, "Got flush response\n");
        if (core->signalKernelFinish) {
            core->shaderImpl->finish_kernel();
            core->signalKernelFinish = false;
        }
        delete pkt->req;
        delete pkt;
        return true;
    }

    assert(pkt->isRead());

    uint8_t data[16];
    assert(pkt->getSize() <= sizeof(data));

    warp_inst_t &inst = ((SenderState*)pkt->senderState)->inst;

    if (!core->shaderImpl->ldst_unit_wb_inst(inst)) {
        // Writeback register is occupied, stall
        assert(core->writebackBlocked < 0);
        core->writebackBlocked = idx;
        return false;
    }

    pkt->writeData(data);
    DPRINTF(CudaCoreAccess, "Loaded data %d\n", *(int*)data);
    core->shaderImpl->writeRegister(inst, core->warpSize, idx, (char*)data);

    delete pkt->senderState;
    delete pkt->req;
    delete pkt;

    return true;
}

void
CudaCore::LSQPort::recvRetry()
{
    panic("Not sure how to respond to a recvRetry...");
}

Tick
CudaCore::SCInstPort::recvAtomic(PacketPtr pkt)
{
    panic("Not sure how to recvAtomic");
    return 0;
}

void
CudaCore::SCInstPort::recvFunctional(PacketPtr pkt)
{
    panic("Not sure how to recvFunctional");
}

void
CudaCore::writebackClear()
{
    if (writebackBlocked >= 0) lsqPorts[writebackBlocked]->sendRetry();
    writebackBlocked = -1;
}

void
CudaCore::flush()
{
    int asid = 0;
    Addr addr(0);
    Request::Flags flags;
    RequestPtr req = new Request(asid, addr, flags, dataMasterId);
    PacketPtr pkt = new Packet(req, MemCmd::FlushReq);

    DPRINTF(CudaCoreAccess, "Sending flush request\n");
    // It doesn't matter what lane to send this request as it is a control
    // message. We'll assume all control messges go on lane 0.
    if (!lsqPorts[0]->sendTimingReq(pkt)){
        panic("Flush requests should never fail");
    }
}

void
CudaCore::finishKernel()
{
    numKernelsCompleted++;
    signalKernelFinish = true;
    flush();
}

CudaCore *CudaCoreParams::create() {
    return new CudaCore(this);
}

void
CudaCore::regStats()
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
    numKernelsCompleted
        .name(name() + ".kernels_completed")
        .desc("Number of kernels completed")
        ;
}

void
CudaCore::record_ld(memory_space_t space)
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
CudaCore::record_st(memory_space_t space)
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
CudaCore::record_inst(int inst_type)
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
CudaCore::record_block_issue(unsigned hw_cta_id)
{
    assert(!coreCTAActive[hw_cta_id]);
    coreCTAActive[hw_cta_id] = true;
    coreCTAActiveStats[hw_cta_id].push_back(curTick());

    if (activeCTAs == 0) {
        beginActiveCycle = curCycle();
    }
    activeCTAs++;
}

void
CudaCore::record_block_commit(unsigned hw_cta_id)
{
    assert(coreCTAActive[hw_cta_id]);
    coreCTAActive[hw_cta_id] = false;
    coreCTAActiveStats[hw_cta_id].push_back(curTick());

    activeCTAs--;
    if (activeCTAs == 0) {
        activeCycles += curCycle() - beginActiveCycle;
    }
}

void CudaCore::printCTAStats(std::ostream& out)
{
    std::map<unsigned, std::vector<Tick> >::iterator iter;
    std::vector<Tick>::iterator times;
    for (iter = coreCTAActiveStats.begin(); iter != coreCTAActiveStats.end(); iter++) {
        unsigned cta_id = iter->first;
        out << id << ", " << cta_id << ", ";
        for (times = coreCTAActiveStats[cta_id].begin(); times != coreCTAActiveStats[cta_id].end(); times++) {
            out << *times << ", ";
        }
        out << curTick() << "\n";
    }
}
