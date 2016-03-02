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
#include "gpu/gpgpu-sim/cuda_core.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "mem/page_table.hh"
#include "params/CudaCore.hh"
#include "sim/system.hh"

using namespace std;

CudaCore::CudaCore(const Params *p) :
    MemObject(p), instPort(name() + ".inst_port", this),
    lsqControlPort(name() + ".lsq_ctrl_port", this), _params(p),
    dataMasterId(p->sys->getMasterId(name() + ".data")),
    instMasterId(p->sys->getMasterId(name() + ".inst")), id(p->id),
    itb(p->itb), cudaGPU(p->gpu), maxNumWarpsPerCore(p->warp_contexts)
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

    needsFenceUnblock.resize(maxNumWarpsPerCore);
    for (int i = 0; i < maxNumWarpsPerCore; i++) {
        needsFenceUnblock[i] = false;
    }
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
    } else if (if_name == "lsq_ctrl_port") {
        return lsqControlPort;
    } else {
        return MemObject::getMasterPort(if_name, idx);
    }
}

void
CudaCore::unserialize(CheckpointIn &cp)
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
    map<Addr,mem_fetch *>::iterator iter =
            busyInstCacheLineAddrs.find(addrToLine(addr));
    return iter == busyInstCacheLineAddrs.end();
}

inline Addr CudaCore::addrToLine(Addr a)
{
    unsigned int maskBits = cudaGPU->getRubySystem()->getBlockSizeBits();
    return a & (((uint64_t)-1) << maskBits);
}

void
CudaCore::icacheFetch(Addr addr, mem_fetch *mf)
{
    assert(instCacheResourceAvailable(addr));

    Addr line_addr = addrToLine(addr);
    DPRINTF(CudaCoreFetch,
            "Fetch request, addr: 0x%x, size: %d, line: 0x%x\n",
            addr, mf->size(), line_addr);

    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = (Addr)mf->get_pc();
    const int asid = 0;

    BaseTLB::Mode mode = BaseTLB::Read;
    req->setVirt(asid, line_addr, mf->size(), flags, instMasterId, pc);
    req->setFlags(Request::INST_FETCH);

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
    DataTranslation<CudaCore*> *translation
            = new DataTranslation<CudaCore*>(this, state);

    busyInstCacheLineAddrs[addrToLine(req->getVaddr())] = mf;
    itb->beginTranslateTiming(req, translation, mode);
}

void CudaCore::finishTranslation(WholeTranslationState *state)
{
    if (state->getFault() != NoFault) {
        panic("Instruction translation encountered fault (%s) for address 0x%x",
              state->getFault()->name(), state->mainReq->getVaddr());
    }
    assert(state->mode == BaseTLB::Read);
    PacketPtr pkt = new Packet(state->mainReq, MemCmd::ReadReq);
    pkt->allocate();
    assert(pkt->req->isInstFetch());
    if (!stallOnICacheRetry) {
        sendInstAccess(pkt);
    } else {
        DPRINTF(CudaCoreFetch, "Port blocked, add vaddr: 0x%x to retry list\n",
                pkt->req->getVaddr());
        retryInstPkts.push_back(pkt);
    }
    delete state;
}

void
CudaCore::sendInstAccess(PacketPtr pkt)
{
    assert(!stallOnICacheRetry);

    DPRINTF(CudaCoreFetch,
            "Sending inst read of %d bytes to vaddr: 0x%x\n",
            pkt->getSize(), pkt->req->getVaddr());

    if (!instPort.sendTimingReq(pkt)) {
        stallOnICacheRetry = true;
        if (pkt != retryInstPkts.front()) {
            retryInstPkts.push_back(pkt);
        }
        DPRINTF(CudaCoreFetch, "Send failed vaddr: 0x%x. Waiting: %d\n",
                pkt->req->getVaddr(), retryInstPkts.size());
    }
    numInstCacheRequests++;
}

void
CudaCore::handleRetry()
{
    assert(stallOnICacheRetry);
    assert(retryInstPkts.size());

    numInstCacheRetry++;

    PacketPtr retry_pkt = retryInstPkts.front();
    DPRINTF(CudaCoreFetch, "Received retry, vaddr: 0x%x\n",
            retry_pkt->req->getVaddr());

    if (instPort.sendTimingReq(retry_pkt)) {
        retryInstPkts.remove(retry_pkt);

        // If there are still packets on the retry list, signal to Ruby that
        // there should be a retry call for the next packet when possible
        stallOnICacheRetry = (retryInstPkts.size() > 0);
        if (stallOnICacheRetry) {
            retry_pkt = retryInstPkts.front();
            instPort.sendTimingReq(retry_pkt);
        }
    } else {
        panic("Access should never fail on a retry!");
    }
}

void
CudaCore::recvInstResp(PacketPtr pkt)
{
    assert(pkt->req->isInstFetch());
    map<Addr,mem_fetch *>::iterator iter =
            busyInstCacheLineAddrs.find(addrToLine(pkt->req->getVaddr()));
    assert(iter != busyInstCacheLineAddrs.end());

    DPRINTF(CudaCoreFetch, "Finished fetch on vaddr 0x%x\n",
            pkt->req->getVaddr());

    shaderImpl->accept_fetch_response(iter->second);

    busyInstCacheLineAddrs.erase(iter);

    if (pkt->req) delete pkt->req;
    delete pkt;
}

bool
CudaCore::executeMemOp(const warp_inst_t &inst)
{
    assert(inst.space.get_type() == global_space ||
           inst.space.get_type() == const_space ||
           inst.space.get_type() == local_space ||
           inst.op == BARRIER_OP ||
           inst.op == MEMORY_BARRIER_OP);
    assert(inst.valid());

    // for debugging
    bool completed = false;

    int size = inst.data_size;
    if (inst.is_load() || inst.is_store()) {
        assert(size >= 1 && size <= 8);
    }
    size *= inst.vectorLength;
    assert(size <= 16);
    if (inst.op == BARRIER_OP || inst.op == MEMORY_BARRIER_OP) {
        if (inst.active_count() != inst.warp_size()) {
            warn_once("ShaderLSQ received partial-warp fence: Assuming you know what you're doing");
        }
    }
    const int asid = 0;
    Request::Flags flags;
    if (inst.isatomic()) {
        assert(inst.memory_op == memory_store);
        // Assert that gem5-gpu knows how to handle the requested atomic type.
        // TODO: When all atomic types and data sizes are implemented, remove
        assert(inst.get_atomic() == ATOMIC_INC ||
               inst.get_atomic() == ATOMIC_MAX ||
               inst.get_atomic() == ATOMIC_MIN ||
               inst.get_atomic() == ATOMIC_ADD ||
               inst.get_atomic() == ATOMIC_CAS);
        assert(inst.data_type == S32_TYPE ||
               inst.data_type == U32_TYPE ||
               inst.data_type == F32_TYPE ||
               inst.data_type == B32_TYPE);
        // GPU atomics will use the MEM_SWAP flag to indicate to Ruby that the
        // request should be passed to the cache hierarchy as secondary
        // RubyRequest_Atomic.
        // NOTE: Most GPU atomics have conditional writes and most perform some
        //       operation on loaded data before writing it back into cache.
        //       This makes them read-modify-conditional-write operations, but
        //       for ease of development, use the MEM_SWAP designator for now.
        flags.set(Request::MEM_SWAP);
    }

    if (inst.space.get_type() == const_space) {
        DPRINTF(CudaCoreAccess, "Const space: %p\n", inst.pc);
    } else if (inst.space.get_type() == local_space) {
        DPRINTF(CudaCoreAccess, "Local space: %p\n", inst.pc);
    } else if (inst.space.get_type() == param_space_local) {
        DPRINTF(CudaCoreAccess, "Param local space: %p\n", inst.pc);
    } else {
        DPRINTF(CudaCoreAccess, "Global space: %p\n", inst.pc);
    }

    for (int lane = 0; lane < warpSize; lane++) {
        if (inst.active(lane)) {
            Addr addr = inst.get_addr(lane);

            PacketPtr pkt;
            if (inst.is_load()) {
                // Not all cache operators are currently supported in gem5-gpu.
                // Verify that a supported cache operator is specified for this
                // load instruction.
                if (!inst.isatomic() && inst.cache_op == CACHE_GLOBAL) {
                    // If this is a load instruction that must access coherent
                    // global memory, bypass the L1 cache to avoid stale hits
                    flags.set(Request::BYPASS_L1);
                } else if (inst.cache_op != CACHE_ALL &&
                    !(inst.isatomic() && inst.cache_op == CACHE_GLOBAL)) {
                    panic("Unhandled cache operator (%d) on load\n",
                          inst.cache_op);
                }
                RequestPtr req = new Request(asid, addr, size, flags,
                        dataMasterId, inst.pc, id, inst.warp_id());
                pkt = new Packet(req, MemCmd::ReadReq);
                if (inst.isatomic()) {
                    assert(flags.isSet(Request::MEM_SWAP));
                    AtomicOpRequest *pkt_data = new AtomicOpRequest();
                    pkt_data->lastAccess = true;
                    pkt_data->uniqueId = lane;
                    pkt_data->dataType = getDataType(inst.data_type);
                    pkt_data->atomicOp = getAtomOpType(inst.get_atomic());
                    pkt_data->lineOffset = 0;
                    pkt_data->setData((uint8_t*)inst.get_data(lane));

                    // TODO: If supporting atomics that require more operands,
                    // will need to copy that data here also

                    // Create packet data to include the atomic type and
                    // the register data to be used (e.g. atomicInc requires
                    // the saturating value up to which to count)
                    pkt->dataDynamic(pkt_data);
                } else {
                    pkt->allocate();
                }
                // Since only loads return to the CudaCore
                pkt->senderState = new SenderState(inst);
            } else if (inst.is_store()) {
                assert(!inst.isatomic());
                // Not all cache operators are currently supported in gem5-gpu.
                // Verify that a supported cache operator is specified for this
                // load instruction.
                if (inst.cache_op == CACHE_GLOBAL) {
                    flags.set(Request::BYPASS_L1);
                } else if (inst.cache_op != CACHE_ALL &&
                           inst.cache_op != CACHE_WRITE_BACK) {
                    panic("Unhandled cache operator (%d) on store\n",
                          inst.cache_op);
                }
                RequestPtr req = new Request(asid, addr, size, flags,
                        dataMasterId, inst.pc, id, inst.warp_id());
                pkt = new Packet(req, MemCmd::WriteReq);
                pkt->allocate();
                pkt->setData((uint8_t*)inst.get_data(lane));
                DPRINTF(CudaCoreAccess, "Send store from lane %d address 0x%llx: data = %d\n",
                        lane, pkt->req->getVaddr(), *(int*)inst.get_data(lane));
            } else if (inst.op == BARRIER_OP || inst.op == MEMORY_BARRIER_OP) {
                assert(!inst.isatomic());
                // Setup Fence packet
                // TODO: If adding fencing functionality, specify control data
                // in packet or request
                RequestPtr req = new Request(asid, 0x0, 0, flags, dataMasterId,
                        inst.pc, id, inst.warp_id());
                pkt = new Packet(req, MemCmd::FenceReq);
                pkt->senderState = new SenderState(inst);
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

                if (inst.is_load() || inst.op == BARRIER_OP ||
                    inst.op == MEMORY_BARRIER_OP) {
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

    if (inst.op == BARRIER_OP || inst.op == MEMORY_BARRIER_OP) {
        needsFenceUnblock[inst.warp_id()] = true;
    }

    // Return that there should not be a pipeline stall
    return false;
}

bool
CudaCore::recvLSQDataResp(PacketPtr pkt, int lane_id)
{
    assert(pkt->isRead() || pkt->cmd == MemCmd::FenceResp);

    DPRINTF(CudaCoreAccess, "Got a response for lane %d address 0x%llx\n",
            lane_id, pkt->req->getVaddr());

    warp_inst_t &inst = ((SenderState*)pkt->senderState)->inst;
    assert(!inst.empty() && inst.valid());

    if (pkt->isRead()) {
        if (!shaderImpl->ldst_unit_wb_inst(inst)) {
            // Writeback register is occupied, stall
            assert(writebackBlocked < 0);
            writebackBlocked = lane_id;
            return false;
        }

        uint8_t data[16];
        assert(pkt->getSize() <= sizeof(data));

        if (inst.isatomic()) {
            assert(pkt->req->isSwap());
            AtomicOpRequest *lane_req = pkt->getPtr<AtomicOpRequest>();
            lane_req->writeData(data);
        } else {
            pkt->writeData(data);
        }
        DPRINTF(CudaCoreAccess, "Loaded data %d\n", *(int*)data);
        shaderImpl->writeRegister(inst, warpSize, lane_id, (char*)data);
    } else if (pkt->cmd == MemCmd::FenceResp) {
        if (needsFenceUnblock[inst.warp_id()]) {
            if (inst.op == BARRIER_OP) {
                // Signal that warp has reached barrier
                assert(!shaderImpl->warp_waiting_at_barrier(inst.warp_id()));
                shaderImpl->warp_reaches_barrier(inst);
                DPRINTF(CudaCoreAccess, "Warp %d reaches barrier\n",
                        pkt->req->threadId());
            }

            // Signal that fence has been cleared
            assert(shaderImpl->fence_unblock_needed(inst.warp_id()));
            shaderImpl->complete_fence(pkt->req->threadId());
            DPRINTF(CudaCoreAccess, "Cleared fence, unblocking warp %d\n",
                    pkt->req->threadId());

            needsFenceUnblock[inst.warp_id()] = false;
        }
    }

    delete pkt->senderState;
    delete pkt->req;
    delete pkt;

    return true;
}

void
CudaCore::recvLSQControlResp(PacketPtr pkt)
{
    if (pkt->isFlush()) {
        DPRINTF(CudaCoreAccess, "Got flush response\n");
        if (signalKernelFinish) {
            shaderImpl->finish_kernel();
            signalKernelFinish = false;
        }
    } else {
        panic("Received unhandled packet type in control port");
    }
    delete pkt->req;
    delete pkt;
}

void
CudaCore::writebackClear()
{
    if (writebackBlocked >= 0) lsqPorts[writebackBlocked]->sendRetryResp();
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
    if (!lsqControlPort.sendTimingReq(pkt)){
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

bool
CudaCore::LSQPort::recvTimingResp(PacketPtr pkt)
{
    return core->recvLSQDataResp(pkt, idx);
}

void
CudaCore::LSQPort::recvReqRetry()
{
    panic("Not sure how to respond to a recvReqRetry...");
}

bool
CudaCore::LSQControlPort::recvTimingResp(PacketPtr pkt)
{
    core->recvLSQControlResp(pkt);
    return true;
}

void
CudaCore::LSQControlPort::recvReqRetry()
{
    panic("CudaCore::LSQControlPort::recvReqRetry() not implemented!");
}

bool
CudaCore::InstPort::recvTimingResp(PacketPtr pkt)
{
    core->recvInstResp(pkt);
    return true;
}

void
CudaCore::InstPort::recvReqRetry()
{
    core->handleRetry();
}

Tick
CudaCore::InstPort::recvAtomic(PacketPtr pkt)
{
    panic("Not sure how to recvAtomic");
    return 0;
}

void
CudaCore::InstPort::recvFunctional(PacketPtr pkt)
{
    panic("Not sure how to recvFunctional");
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
    std::map<unsigned, std::vector<Tick> >::iterator iter =
            coreCTAActiveStats.begin();
    std::vector<Tick>::iterator times;
    for (; iter != coreCTAActiveStats.end(); iter++) {
        unsigned cta_id = iter->first;
        out << id << ", " << cta_id << ", ";
        times = coreCTAActiveStats[cta_id].begin();
        for (; times != coreCTAActiveStats[cta_id].end(); times++) {
            out << *times << ", ";
        }
        out << curTick() << "\n";
    }
}
