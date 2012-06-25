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

#ifndef __GPGPU_STREAM_PROCESSOR_HH__
#define __GPGPU_STREAM_PROCESSOR_HH__

#include <map>
#include <queue>
#include <set>
#include <vector>

#include "../gpgpu-sim/gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/stream_manager.h"
#include "arch/types.hh"
#include "config/the_isa.hh"
#include "cpu/translation.hh"
#include "mem/ruby/system/RubyPort.hh"
#include "mem/mem_object.hh"
#include "params/StreamProcessorArray.hh"
#include "sim/process.hh"
#include "sim/system.hh"
#include "copy_engine.hh"
#include "shader_core.hh"

/**
 *  Main wrapper class for GPGPU-Sim
 *
 *  All functional accesses from GPGPU-Sim are routed through this class.
 *  This class also holds pointers to all of the shader cores and the copy engine.
 *  Statistics for kernel times are also kept in this class.
 *
 *  Currently this class only supports a single GPU device and does not support
 *  concurrent kernels.
 */
class StreamProcessorArray : public SimObject
{
public:
    /**
     *  In order to get around the fact that we can't modify any of gem5,
     *  we need a pointer to the SPA. You can get it by calling this.
     *  Note: We could easily make this implement multiple SPA's when we
     *        get to that point
     */
    static StreamProcessorArray *getStreamProcessorArray() {
        assert(singletonPointer != NULL);
        return singletonPointer;
    }

private:
    static StreamProcessorArray *singletonPointer;

protected:
    typedef StreamProcessorArrayParams Params;

    /**
     *  Helper class for both Stream and GPU tick events
     */
    class TickEvent : public Event
    {
        friend class StreamProcessorArray;

    private:
        StreamProcessorArray *cpu;
        bool streamTick;

    public:
        TickEvent(StreamProcessorArray *c, bool stream) : Event(CPU_Tick_Pri), cpu(c), streamTick(stream) {}
        void process() {
            if (streamTick) cpu->streamTick();
            else cpu->gpuTick();
        }
        virtual const char *description() const { return "StreamProcessorArray tick"; }
    };

    const StreamProcessorArrayParams *_params;
    const Params * params() const { return dynamic_cast<const Params *>(_params);	}

    /// Tick for when the GPU needs to run its next cycle
    TickEvent gpuTickEvent;

    /// Tick for when the stream manager needs execute
    TickEvent streamTickEvent;

private:
    friend class ShaderCore;

    /// Callback for the gpu tick
    void gpuTick();

    /// Callback for the stream manager tick
    void streamTick();

    /// When the shader core is created it calls this
    int registerShaderCore(ShaderCore *sc);

    /// Pointer to the copy engine for this device
    SPACopyEngine *copyEngine;

    /// Used to register this SPA with the system
    System *system;

    /// If true do global mem requests through gem5 otherwise do them through GPGPU-Sim
    int useGem5Mem;
    int sharedMemDelay;
    double launchDelay;
    double returnDelay;

    /// If true there is a kernel currently executing
    /// NOTE: Jason doesn't think we need this
    bool running;

    /// True if the running thread is currently blocked and needs to be activated
    bool unblockNeeded;

    /// Pointer to ruby system used to clear the Ruby stats
    /// NOTE: I think there is a more right way to do this
    RubySystem *ruby;

    /// Holds all of the shader cores in this stream processor array
    int numShaderCores;
    std::vector<ShaderCore*> shaderCores;

    /// Used for allocating memory that isn't global (constant, local, etc)
    std::map<Addr,size_t> allocatedMemory;
    Addr brk_point;
    Addr nextAddr;

    /// From the process that is using this SPA
    ThreadContext *tc;
    LiveProcess *process;

    /// Conversion from the requests that GPGPU-Sim makes to gem5 ticks
    /// Right now this should be seconds to picoseconds (10^12)
    float gpuTickConversion;

    /// For statistics
    unsigned long long kernelStartTime;
    std::vector<unsigned long long> kernelTimes;
    Tick clearTick;
    std::queue<kernelTermInfo> finished_kernels;

    /// Pointers to GPGPU-Sim objects
    gpgpu_sim *theGPU;
    stream_manager *streamManager;

    /// Flag to make sure we don't schedule twice in the same tick
    bool streamScheduled;

    /// Number of ticks to delay for each stream operation
    /// This is a function of the driver overheads
    int streamDelay;

public:
    /// Constructor
    StreamProcessorArray(const Params *p);

    /// Called during GPGPU-Sim initialization to initialize the SPA
    void start(ThreadContext *_tc, gpgpu_sim *the_gpu, stream_manager *_stream_manager);

    /// Getter for whether we are using Ruby or GPGPU-Sim memory modeling
    int getUseGem5Mem(){ return useGem5Mem; }
    int getSharedMemDelay(){ return sharedMemDelay; }

    /// called if the gpu is going to block the processor and should unblock it
    /// when it's done. Returns true if you should suspend the thread
    bool setUnblock();

    /// Used to unblock the thread when an event completes
    void unblock();

    /// Called at the beginning of each kernel launch to start the statistics
    void beginRunning(Tick launchTime);

    /// Called from GPGPU-Sim next_clock_domain and schedules cycle() to be run
    /// gpuTicks (in GPPGU-Sim tick (seconds)) from now
    void gpuRequestTick(float gpuTicks);

    /// Schedules the stream manager to be checked in 'ticks' ticks from now
    void streamRequestTick(int ticks);

    /// Reset statistics for the SPA and for all of Ruby
    void clearStats();

    /// used for GPGPU-Sim's functional simulation portion. (decode)
    void writeFunctional(Addr addr, size_t length, const uint8_t* data);
    void readFunctional(Addr addr, size_t length, uint8_t* data);

    /// used for gpgpu cuda api
    Addr allocMemory(size_t length);
    void freeMemory(Addr addr);

    /// Returns shader core with id coreId
    ShaderCore *getShaderCore(int coreId);

    /// Callback for GPGPU-Sim to get the current simulation time
    unsigned long long getCurTick(){ return curTick(); }

    /// Used to print stats at the end of simulation
    void gpuPrintStats(std::ostream& out);

    /// Begins a timing memory copy from src to dst
    void memcpy(void *src, void *dst, size_t count, struct CUstream_st *stream);

    /// Begins a timing memory copy from src to/from the symbol+offset
    void memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to, struct CUstream_st *stream);
};

/**
 *  Helper class to print out statistics at the end of simulation
 */
class GPUExitCallback : public Callback
{
private:
    std::string stats_filename;
    StreamProcessorArray *spa_obj;

public:
    virtual ~GPUExitCallback() {}

    GPUExitCallback(StreamProcessorArray *_spa_obj, const std::string& _stats_filename)
    {
        stats_filename = _stats_filename;
        spa_obj = _spa_obj;
    }

    virtual void process();
};


#endif

