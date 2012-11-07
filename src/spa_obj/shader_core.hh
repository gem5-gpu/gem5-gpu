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

#ifndef __GPGPU_SHADER_CORE_HH__
#define __GPGPU_SHADER_CORE_HH__

#include <map>
#include <queue>
#include <set>

#include "../gpgpu-sim/cuda-sim/ptx_ir.h"
#include "../gpgpu-sim/cuda-sim/ptx_sim.h"
#include "../gpgpu-sim/gpgpu-sim/mem_fetch.h"
#include "../gpgpu-sim/gpgpu-sim/shader.h"
#include "arch/types.hh"
#include "config/the_isa.hh"
#include "cpu/translation.hh"
#include "mem/ruby/system/RubyPort.hh"
#include "mem/mem_object.hh"
#include "params/ShaderCore.hh"
#include "shader_tlb.hh"
#include "sim/process.hh"

class StreamProcessorArray;

/**
 *  Wrapper class for the shader cores in GPGPU-Sim
 *  A shader core is equivalent to an NVIDIA streaming multiprocessor (SM)
 *
 *  GPGPU-Sim shader *timing* memory references are routed through this class.
 *
 */
class ShaderCore : public MemObject
{
protected:
    typedef ShaderCoreParams Params;

    /**
     * Port for sending a receiving data memory requests
     * Required for implementing MemObject
     */
    class SCDataPort : public MasterPort
    {
        friend class ShaderCore;

    private:
        // Pointer back to shader core for callbacks
        ShaderCore *proc;

        // Holds packets that failed to send for retry
        std::list<PacketPtr> outDataPkts;

        // List of write packets to be sent
        std::map<Addr,std::list<PacketPtr> > writePackets;
    public:
        SCDataPort(const std::string &_name, ShaderCore *_proc)
        : MasterPort(_name, _proc), proc(_proc) {}
        // Sends a request into the gem5 memory system (Ruby)
        bool sendPkt(PacketPtr pkt);

    protected:
        virtual bool recvTimingResp(PacketPtr pkt);
        virtual void recvRetry();
        virtual Tick recvAtomic(PacketPtr pkt);
        virtual void recvFunctional(PacketPtr pkt);
    };
    /// Instantiation of above port
    SCDataPort dataPort;

    /**
     * Port for sending a receiving data memory requests
     * Required for implementing MemObject
     */
    class SCInstPort : public MasterPort
    {
        friend class ShaderCore;

    private:
        // Pointer back to shader core for callbacks
        ShaderCore *proc;

        // Holds packets that failed to send for retry
        std::list<PacketPtr> outInstPkts;

    public:
        SCInstPort(const std::string &_name, ShaderCore *_proc)
        : MasterPort(_name, _proc), proc(_proc) {}
        // Sends a request into the gem5 memory system (Ruby)
        bool sendPkt(PacketPtr pkt);

    protected:
        virtual bool recvTimingResp(PacketPtr pkt);
        virtual void recvRetry();
        virtual Tick recvAtomic(PacketPtr pkt);
        virtual void recvFunctional(PacketPtr pkt);
    };
    /// Instantiation of above port
    SCInstPort instPort;

    /**
     *  Helper class for tick events
     */
    class TickEvent : public Event
    {
        friend class ShaderCore;

    private:
        ShaderCore *sc;
    public:
        TickEvent(ShaderCore *c) : Event(CPU_Tick_Pri), sc(c) {}
        void process() { sc->tick(); }
        virtual const char *description() const { return "ShaderCore tick"; }
    };

    const Params * params() const { return dynamic_cast<const Params *>(_params);	}
    const ShaderCoreParams *_params;

    TickEvent tickEvent;
    bool scheduledTickEvent;

    MasterID masterId;

private:
    /// Called on tick events, and re-schedules a previously failed access
    void tick();

    /// Called to begin a virtual memory access
    void accessVirtMem(RequestPtr req, mem_fetch *mf, BaseTLB::Mode mode);

    /// Id for this shader core, should match the id in GPGPU-Sim
    /// To convert gem5Id = cluster_num*shader_per_cluster+num_in_cluster
    int id;

    /// Stalled because a memory request called recvRetry, usually because a queue
    /// filled up
    bool stallOnDCacheRetry;
    bool stallOnICacheRetry;

    /// Stalled because we have some atomics finish the read part
    /// and we still need to issue the writes, but they are beinging blocked by
    /// something else
//    bool stallOnAtomicQueue;

    /// Holds requests that came in after a failed packet, but
    /// before a call to resourceAvailable
    class PendingReq {
    public:
        PendingReq(RequestPtr _req, BaseTLB::Mode _mode, mem_fetch *_mf=NULL) :
            req(_req), mode(_mode), mf(_mf) {}
        RequestPtr req;
        BaseTLB::Mode mode;
        mem_fetch *mf;
    };

    /// holds all outstanding addresses, maps from address to mf object (from gpgpu-sim)
    /// used mostly for acking GPGPU-Sim
    std::map<Addr,mem_fetch *> busyDataCacheLineAddrs;
    std::map<Addr,mem_fetch *> busyInstCacheLineAddrs;

    /// For profiling warp specific latencies pair is (pc, warpID)
    std::map<std::pair<uint64_t, unsigned>, class WarpMemRequest> warpMemRequests;

    /// Queue for outstanding atomic writes (see stallOnAtomicQueue)
    std::queue<PendingReq*> atomicQueue;

    /// TLB's. These do NOT perform any timing right now
    ShaderTLB *dtb;
    ShaderTLB *itb;

    /// Point to SPA this shader core is part of
    StreamProcessorArray *spa;

    /// Pointer to GPGPU-Sim shader this shader core is a proxy for
    shader_core_ctx *shaderImpl;

    /// Returns the line of the address, a
    Addr addrToLine(Addr a);

    /// Can we issue a data cache request this cycle?
    int dataCacheResourceAvailable(Addr a);

    /// Can we issue a data cache request this cycle?
    int instCacheResourceAvailable(Addr a);

    // A class to store actions that need to be taken by the shader core
    // when memory accesses use gem5 memory
    class MemRequestHint {
        Addr addr;
        size_t size;
        BaseTLB::Mode reqType;
        uint8_t* data;
        unsigned int wID;
        unsigned int tID;
        ptx_thread_info* thread;
        const ptx_instruction* pI;
    public:
        MemRequestHint(Addr _addr, size_t _size, BaseTLB::Mode _reqType) :
            addr(_addr), size(_size), reqType(_reqType), data(NULL)
            { data = new uint8_t[size]; }
        MemRequestHint(Addr _addr, size_t _size, BaseTLB::Mode _reqType, unsigned int _wID, unsigned int _tID) :
            addr(_addr), size(_size), reqType(_reqType), data(NULL), wID(_wID), tID(_tID)
            {}
        ~MemRequestHint() { if (data) delete[] data; }
        void addData(size_t _size, const void* _data, unsigned offset = 0);
        bool isRead() {return (reqType == BaseTLB::Read);}
        bool isWrite() {return (reqType == BaseTLB::Write);}
        Addr getAddr() {return addr;}
        size_t getSize() {return size;}
        uint8_t* getData() {return data;}
        unsigned int getWID() {return wID;}
        unsigned int getTID() {return tID;}
        ptx_thread_info * getThread() {return thread;}
        const ptx_instruction* getInst() {return pI;}
        void setThread(ptx_thread_info *_thd) {thread = _thd;}
        void setInst(const ptx_instruction *_pI) {pI = _pI;}
        Tick tick;
    };

    // Map of memory write hints from ST instructions executed on this core
    std::map<Addr,std::list<MemRequestHint*> > memWriteHints;
    // Map of memory write packets to be sent by this core
    std::map<Addr,std::list<MemRequestHint*> > writePacketHints;

    class ReadPacketBuffer : public Packet::SenderState {
        mem_fetch* mf;
        std::list<MemRequestHint*> bufferedReads;
    public:
        ReadPacketBuffer(mem_fetch* _mf) : mf(_mf) {}
        ~ReadPacketBuffer() { bufferedReads.clear(); }
        unsigned int numBufferedReads() {return bufferedReads.size();}
        std::list<MemRequestHint*> getBufferedReads() {return bufferedReads;}
        void addBufferedRead(MemRequestHint* hint) {bufferedReads.push_back(hint);}
    };

    // Map of memory read hints from LD instructions executed on this core
    std::map<Addr,std::list<MemRequestHint*> > memReadHints;

public:
    /// Constructor
    ShaderCore(const Params *p);

    /// Required for implementing MemObject
    virtual BaseMasterPort& getMasterPort(const std::string &if_name, PortID idx = -1);

    /// For checkpoint restore (empty unserialize)
    virtual void unserialize(Checkpoint *cp, const std::string &section);

    /// Perform initialization. Called from SPA
    void initialize();

    /// Required for translation. Calls sendPkt with physical address
    void finishTranslation(WholeTranslationState *state);

    /** This function is used by the page table walker to determine if it could
    * translate the a pending request or if the underlying request has been
    * squashed. This always returns false for the GPU as it never
    * executes any instructions speculatively.
    * @ return Is the current instruction squashed?
    */
    bool isSquashed() const { return false; }

    /// Wrapper functions for GPGPU-Sim to call on reads, writes, and atomics
    int readTiming (Addr a, size_t size, mem_fetch *mf);
    int writeTiming(Addr a, size_t size, mem_fetch *mf);
    int atomicRMW(Addr a, size_t size, mem_fetch *mf);
    void addWriteHint(Addr addr, size_t size, const void* data);
    void addReadHint(Addr addr, size_t size, const void* data, ptx_thread_info *thd, const ptx_instruction *pI);

    // Wrapper functions for GPGPU-Sim instruction cache accesses
    void icacheFetch(Addr a, mem_fetch *mf);

    // For counting statistics
    Stats::Scalar numLocalLoads;
    Stats::Scalar numLocalStores;
    Stats::Scalar numSharedLoads;
    Stats::Scalar numSharedStores;
    Stats::Scalar numParamKernelLoads;
    Stats::Scalar numParamLocalLoads;
    Stats::Scalar numParamLocalStores;
    Stats::Scalar numConstLoads;
    Stats::Scalar numTexLoads;
    Stats::Scalar numGlobalLoads;
    Stats::Scalar numGlobalStores;
    Stats::Scalar numSurfLoads;
    Stats::Scalar numGenericLoads;
    Stats::Scalar numGenericStores;
    Stats::Scalar numDataCacheRequests;
    Stats::Scalar numDataCacheRetry;
    Stats::Scalar numInstCacheRequests;
    Stats::Scalar numInstCacheRetry;
    Stats::Vector instCounts;
    void regStats();

    void record_ld(memory_space_t space);
    void record_st(memory_space_t space);
    void record_inst(int inst_type);
    std::map<unsigned, bool> shaderCTAActive;
    std::map<unsigned, std::vector<unsigned long long> > shaderCTAActiveStats;
    void record_block_issue(unsigned hw_cta_id);
    void record_block_commit(unsigned hw_cta_id);
    void printCTAStats(std::ostream& out);
};


#endif

