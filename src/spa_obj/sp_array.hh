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
#include <string>
#include <vector>

#include "../gpgpu-sim/gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/stream_manager.h"
#include "arch/types.hh"
#include "config/the_isa.hh"
#include "cpu/translation.hh"
#include "debug/StreamProcessorArrayPageTable.hh"
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
     *  Only to be used in GPU system calls (gpu_syscalls) as a way to access
     *  the currently running SPA.
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
    /// Callback for the gpu tick
    void gpuTick();

    /// Callback for the stream manager tick
    void streamTick();

    /// Pointer to the copy engine for this device
    SPACopyEngine *copyEngine;

    /// Used to register this SPA with the system
    System *system;

    /// Frequency of GPU in Hz
    Tick frequency;

    /// Number of threads in each warp, also number of "cores" per shader core/SM
    int warpSize;

    /// Are we restoring from a checkpoint?
    bool restoring;

    /// If true do global mem requests through gem5 otherwise do them through GPGPU-Sim
    int sharedMemDelay;
    std::string gpgpusimConfigPath;
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
    std::vector<ShaderCore*> shaderCores;

    /// The thread context, stream and thread ID currently running on the SPA
    ThreadContext *runningTC;
    struct CUstream_st *runningStream;
    int runningTID;
    void beginStreamOperation(struct CUstream_st *_stream) {
        // We currently do not support multiple concurrent streams
        if (runningStream || runningTC) {
            panic("Already a stream operation running (only support one at a time)!");
        }
        runningStream = _stream;
        runningTC = runningStream->getThreadContext();
        runningTID = runningTC->threadId();
    }
    void endStreamOperation() {
        runningStream = NULL;
        runningTC = NULL;
        runningTID = -1;
    }

    /// For statistics
    std::vector<unsigned long long> kernelTimes;
    Tick clearTick;
    std::queue<kernelTermInfo> finishedKernels;
    bool dumpKernelStats;

    /// Pointers to GPGPU-Sim objects
    gpgpu_sim *theGPU;
    stream_manager *streamManager;

    /// Flag to make sure we don't schedule twice in the same tick
    bool streamScheduled;

    /// Number of ticks to delay for each stream operation
    /// This is a function of the driver overheads
    int streamDelay;

    /// For GPU syscalls
    /// This is what is required to save and restore on checkpoints
    std::map<unsigned,symbol_table*> m_code; // fat binary handle => global symbol table
    unsigned int m_last_fat_cubin_handle;
    std::map<const void*,function_info*> m_kernel_lookup; // unique id (CUDA app function address) => kernel entry point
    uint64_t instBaseVaddr;
    bool instBaseVaddrSet;

    /**
     * Helper class for checkpointing
     */
    class _FatBinary
    {
    public:
        int tid; // CPU thread ID
        unsigned int handle;
        Addr sim_fatCubin;
        size_t sim_binSize;
        addr_t sim_alloc_ptr;
        std::map<const void*,std::string> funcMap;
    };

    class _CudaVar
    {
    public:
        Addr sim_deviceAddress;
        std::string deviceName;
        int sim_size;
        int sim_constant;
        int sim_global;
        int sim_ext;
        Addr sim_hostVar;
    };

    std::vector<_FatBinary> fatBinaries;
    std::vector<_CudaVar> cudaVars;

    class SPAPageTable
    {
      private:
        std::map<Addr, Addr> pageMap;

      public:
        SPAPageTable() {};

        Addr addrToPage(Addr addr);
        void insert(Addr vaddr, Addr paddr) {
            if (pageMap.find(vaddr) == pageMap.end()) {
                pageMap[vaddr] = paddr;
            } else {
                assert(paddr == pageMap[vaddr]);
            }
        }
        bool lookup(Addr vaddr, Addr& paddr) {
            Addr page_vaddr = addrToPage(vaddr);
            Addr offset = vaddr - page_vaddr;
            if (pageMap.find(page_vaddr) != pageMap.end()) {
                paddr = pageMap[page_vaddr] + offset;
                return true;
            }
            return false;
        }
        /// For checkpointing
        void serialize(std::ostream &os);
        void unserialize(Checkpoint *cp, const std::string &section);
    };
    SPAPageTable pageTable;
    bool manageGPUMemory;
    Addr physicalGPUBaseAddr, physicalGPUBrkAddr;
    Addr virtualGPUBaseAddr, virtualGPUBrkAddr;
    Addr gpuMemorySize;
    std::map<Addr,size_t> allocatedGPUMemory;

public:
    /// Constructor
    StreamProcessorArray(const Params *p);

    /// For checkpointing
    virtual void serialize(std::ostream &os);
    virtual void unserialize(Checkpoint *cp, const std::string &section);

    /// Called after constructor, but before any real simulation
    virtual void startup();

    /// Called during GPGPU-Sim initialization to initialize the SPA
    void start(gpgpu_sim *the_gpu, stream_manager *_stream_manager);

    /// Register devices callbacks
    void registerShaderCore(ShaderCore *sc);
    void registerCopyEngine(SPACopyEngine *ce);

    /// Getter for whether we are using Ruby or GPGPU-Sim memory modeling
    int getSharedMemDelay() { return sharedMemDelay; }
    const char* getConfigPath() { return gpgpusimConfigPath.c_str(); }
    RubySystem* getRubySystem() { return ruby; }
    gpgpu_sim* getTheGPU() { return theGPU; }

    /// Called at the beginning of each kernel launch to start the statistics
    void beginRunning(Tick launchTime, struct CUstream_st *_stream);

    /// Called from GPGPU-Sim next_clock_domain and schedules cycle() to be run
    /// gpuTicks (in GPPGU-Sim tick (seconds)) from now
    void gpuRequestTick(float gpuTicks);

    /// Schedules the stream manager to be checked in 'ticks' ticks from now
    void streamRequestTick(int ticks);

    /// Reset statistics for the SPA and for all of Ruby
    void clearStats();

    /// Returns shader core with id coreId
    ShaderCore *getShaderCore(int coreId);

    /// Returns size of warp (same for all shader cores)
    int getWarpSize() { return warpSize; }

    /// Callback for GPGPU-Sim to get the current simulation time
    unsigned long long getCurTick(){ return curTick(); }

    Tick getFrequency() { return frequency; }

    /// Used to print stats at the end of simulation
    void gpuPrintStats(std::ostream& out);

    /// Begins a timing memory copy from src to dst
    void memcpy(void *src, void *dst, size_t count, struct CUstream_st *stream, stream_operation_type type);

    /// Begins a timing memory copy from src to/from the symbol+offset
    void memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to, struct CUstream_st *stream);

    /// Begins a timing memory set of value to dst
    void memset(Addr dst, int value, size_t count, struct CUstream_st *stream);

    /// Called by the copy engine when a memcpy or memset is complete
    void finishCopyOperation();

    /// Called from shader TLB to be used for X86TLB lookups
    // TODO: Remove this when we remove shader TLB access to X86TLB
    ThreadContext *getThreadContext() { return runningTC; }

    /// Used when blocking and signaling threads
    std::map<ThreadContext*, Addr> blockedThreads;
    bool needsToBlock();
    void blockThread(ThreadContext *tc, Addr signal_ptr);
    void signalThread(ThreadContext *tc, Addr signal_ptr);
    void unblockThread(ThreadContext *tc);

    void saveFatBinaryInfoTop(int tid, unsigned int handle, Addr sim_fatCubin, size_t sim_binSize) {
        _FatBinary bin;
        bin.tid = tid;
        bin.handle = handle;
        bin.sim_fatCubin = sim_fatCubin;
        bin.sim_binSize = sim_binSize;
        fatBinaries.push_back(bin);
    }
    void saveFatBinaryInfoBottom(addr_t sim_alloc_ptr) {
        _FatBinary& bin = fatBinaries.back();
        bin.sim_alloc_ptr = sim_alloc_ptr;
    }
    void saveFunctionNames(unsigned int handle, const char *host, const char *dev) {
        _FatBinary& bin = fatBinaries[handle-1];
        assert(bin.handle == handle);
        bin.funcMap[host] = std::string(dev);
    }
    void saveVar(Addr sim_deviceAddress, const char* deviceName, int sim_size, int sim_constant, int sim_global, int sim_ext, Addr sim_hostVar) {
        _CudaVar var;
        var.sim_deviceAddress = sim_deviceAddress;
        var.deviceName = std::string(deviceName);
        var.sim_size = sim_size;
        var.sim_constant = sim_constant;
        var.sim_global = sim_global;
        var.sim_ext = sim_ext;
        var.sim_hostVar = sim_hostVar;
        cudaVars.push_back(var);
    }

    /// From gpu syscalls (used to be CUctx_st)
    void add_binary( symbol_table *symtab, unsigned fat_cubin_handle );
    void add_ptxinfo( const char *deviceFun, const struct gpgpu_ptx_sim_kernel_info info );
    void register_function( unsigned fat_cubin_handle, const char *hostFun, const char *deviceFun );
    function_info *get_kernel(const char *hostFun);
    void setInstBaseVaddr(uint64_t addr);
    uint64_t getInstBaseVaddr();

    /// For handling GPU memory mapping table
    SPAPageTable* getGPUPageTable() { return &pageTable; };
    void registerDeviceMemory(ThreadContext *tc, Addr vaddr, size_t size);
    void registerDeviceInstText(ThreadContext *tc, Addr vaddr, size_t size);
    bool isManagingGPUMemory() { return manageGPUMemory; }
    Addr allocateGPUMemory(size_t size);
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

