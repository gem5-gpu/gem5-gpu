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

#include "../gpgpu-sim/gpgpu-sim/mem_fetch.h"
#include "../gpgpu-sim/gpgpu-sim/shader.h"
#include "arch/types.hh"
#include "config/the_isa.hh"
#include "cpu/translation.hh"
#include "mem/ruby/system/RubyPort.hh"
#include "mem/mem_object.hh"
#include "params/ShaderCore.hh"
#include "sim/process.hh"

class StreamProcessorArray;

/**
 *  Wrapper class for the shader cores in GPGPU-Sim
 *  A shader core is equivelent to an NVIDIA streaming multiprocessor (SM)
 *
 *  GPGPU-Sim shader *timing* memory references are routed through this class.
 *
 */
class ShaderCore : public MemObject
{
protected:
    typedef ShaderCoreParams Params;

    /**
     * Port for sending a receiving memeory requests
     * Required for implementing MemObject
     */
    class SCPort : public SimpleTimingPort
    {
        friend class ShaderCore;

    private:
        /// Pointer back to shader core for callbacks
        ShaderCore *proc;

        /// holds packets that failed to send for retry
        PacketPtr outstandingPkt;

    public:
        SCPort(const std::string &_name, ShaderCore *_proc)
        : SimpleTimingPort(_name, _proc), proc(_proc)
        { outstandingPkt = NULL; }

    protected:
        virtual bool recvTiming(PacketPtr pkt);
        virtual void recvRetry();
        virtual Tick recvAtomic(PacketPtr pkt);
    };
    /// Instatiation of above port
    SCPort port;

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
    int scheduledTickEvent;

    MasterID masterId;

private:
    friend class StreamProcessorArray;

    /// Called on tick events, and re-schedules a previously failed access
    void tick();

    /// Sends a request into the gem5 memory system (Ruby)
    bool sendPkt(PacketPtr pkt);

    /// Called to begin a virtual memory access
    void accessVirtMem(RequestPtr req, mem_fetch *mf, BaseTLB::Mode mode);

    /// Id for this shader core, should match the id in GPGPU-Sim
    /// To convert gem5Id = cluster_num*shader_per_cluster+num_in_cluster
    int id;

    /// Stalled because a memory request called recvRetry, usually because a queue
    /// filled up
    int stallOnRetry;

    /// Stalled because we have some atomics finish the read part
    /// and we still need to issue the writes, but they are beinging blocked by
    /// something else
    bool stallOnAtomicQueue;

    unsigned int numRetry;
    unsigned int maxOutstanding;

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
    std::map<Addr,mem_fetch *> outstandingAddrs;

    /// Queue for outstanding atomic writes (see stallOnAtomicQueue)
    std::queue<PendingReq*> atomicQueue;

    /// TLB's. These do NOT perform any timing right now
    TheISA::TLB *dtb;
    TheISA::TLB *itb;

    /// Pointer to thread context for translation
    ThreadContext *tc;

    /// Point to SPA this shader core is part of
    StreamProcessorArray *spa;

    /// Perform initialization. Called from SPA
    void initialize(ThreadContext *_tc);

    /// Pointer to GPGPU-Sim shader this shader core is a proxy for
    shader_core_ctx *shaderImpl;

    /// Returns the line of the address, a
    Addr addrToLine(Addr a);

    /// Can we issue a request this cycle?
    int resourceAvailable(Addr a);

public:
    /// Constructor
    ShaderCore(const Params *p);

    /// Required for implementing MemObject
    virtual Port *getPort(const std::string &if_name, int idx = -1);

    /// Required for translation. Calls sendPkt with physical address
    void finishTranslation(WholeTranslationState *state);

    /// Wrapper functions for GPGPU-Sim to call on reads, writes, and atomics
    int readTiming (Addr a, size_t size, mem_fetch *mf);
    int writeTiming(Addr a, size_t size, mem_fetch *mf);
    int atomicRMW(Addr a, size_t size, mem_fetch *mf);
};


#endif

