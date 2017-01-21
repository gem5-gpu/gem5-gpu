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

#ifndef __CUDA_GPU_HH__
#define __CUDA_GPU_HH__

#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "base/callback.hh"
#include "debug/CudaGPU.hh"
#include "debug/CudaGPUPageTable.hh"
#include "gpgpu-sim/gpu-sim.h"
#include "gpu/gpgpu-sim/cuda_core.hh"
#include "gpu/copy_engine.hh"
#include "gpu/shader_mmu.hh"
#include "params/CudaGPU.hh"
#include "params/GPGPUSimComponentWrapper.hh"
#include "sim/clock_domain.hh"
#include "sim/eventq.hh"
#include "sim/process.hh"
#include "sim/system.hh"
#include "stream_manager.h"

/**
 * A wrapper class to manage the clocking of GPGPU-Sim-side components.
 * The CudaGPU must contain one of these wrappers for each clocked component or
 * else GPGPU-Sim simulation progress will stall. Currently, there are four
 * GPGPU-Sim components that are separately cycled: the shader cores, the
 * interconnect, the GPU L2 cache and the DRAM.
 *
 * TODO: Eventually, the L2 and DRAM events should be eliminated by migrating
 * all GPU parameter and local memory accesses over to gem5-gpu.
 */
class GPGPUSimComponentWrapper : public ClockedObject
{
  private:
    gpgpu_sim *theGPU;
    typedef void (gpgpu_sim::*CycleFunc)();
    CycleFunc startCycleFunction;
    CycleFunc endCycleFunction;

  public:
    GPGPUSimComponentWrapper(const GPGPUSimComponentWrapperParams *p) :
        ClockedObject(p), theGPU(NULL), startCycleFunction(NULL),
        endCycleFunction(NULL), componentCycleStartEvent(this),
        // End cycle events must happen after all other components are cycled
        componentCycleEndEvent(this, false, Event::Progress_Event_Pri) {}

    void setGPU(gpgpu_sim *_gpu) {
        assert(!theGPU);
        theGPU = _gpu;
    }

    void setStartCycleFunction(CycleFunc _cycle_func) {
        assert(!startCycleFunction);
        startCycleFunction = _cycle_func;
    }

    void setEndCycleFunction(CycleFunc _cycle_func) {
        assert(!endCycleFunction);
        endCycleFunction = _cycle_func;
    }

    void scheduleEvent(Tick ticks_in_future) {
        Tick start_time;
        if (ticks_in_future < clockPeriod()) {
            start_time = nextCycle();
        } else {
            start_time = clockEdge(ticksToCycles(ticks_in_future));
        }

        assert(startCycleFunction);
        assert(!componentCycleStartEvent.scheduled());
        schedule(componentCycleStartEvent, start_time);

        if (endCycleFunction) {
            assert(!componentCycleEndEvent.scheduled());
            schedule(componentCycleEndEvent, start_time);
        }
    }

  protected:

    void componentCycleStart() {
        assert(startCycleFunction);

        if (theGPU->active()) {
            (theGPU->*startCycleFunction)();
        }

        if (theGPU->active()) {
            // Reschedule the start cycle event
            schedule(componentCycleStartEvent, nextCycle());
        }
    }

    void componentCycleEnd() {
        assert(endCycleFunction);

        if (theGPU->active()) {
            (theGPU->*endCycleFunction)();
        }

        if (theGPU->active()) {
            // Reschedule the end cycle event
            schedule(componentCycleEndEvent, nextCycle());
        }
    }

    EventWrapper<GPGPUSimComponentWrapper, &GPGPUSimComponentWrapper::componentCycleStart>
                                           componentCycleStartEvent;
    EventWrapper<GPGPUSimComponentWrapper, &GPGPUSimComponentWrapper::componentCycleEnd>
                                           componentCycleEndEvent;
};

/**
 *  Main wrapper class for GPGPU-Sim
 *
 *  All global and const accesses from GPGPU-Sim are routed through this class.
 *  This class also holds pointers to all of the CUDA cores and the copy engine.
 *  Statistics for kernel times are also kept in this class.
 *
 *  Currently this class only supports a single GPU device and does not support
 *  concurrent kernels.
 */
class CudaGPU : public ClockedObject
{
  private:
    static std::vector<CudaGPU*> gpuArray;

  public:
    /**
     *  Only to be used in GPU system calls (gpu_syscalls) as a way to access
     *  the CUDA-enabled GPUs.
     */
    static CudaGPU *getCudaGPU(unsigned id) {
        if (id >= gpuArray.size()) {
            panic("CUDA GPU ID not found: %u. Only %u GPUs registered!\n", id, gpuArray.size());
        }
        return gpuArray[id];
    }

    static unsigned getNumCudaDevices() {
        return gpuArray.size();
    }

    static unsigned registerCudaDevice(CudaGPU *gpu) {
        unsigned new_id = getNumCudaDevices();
        gpuArray.push_back(gpu);
        return new_id;
    }

    struct CudaDeviceProperties
    {
        char   name[256];                 // ASCII string identifying device
        size_t totalGlobalMem;            // Global memory available on device in bytes
        size_t sharedMemPerBlock;         // Shared memory available per block in bytes
        int    regsPerBlock;              // 32-bit registers available per block
        int    warpSize;                  // Warp size in threads
        size_t memPitch;                  // Maximum pitch in bytes allowed by memory copies
        int    maxThreadsPerBlock;        // Maximum number of threads per block
        int    maxThreadsDim[3];          // Maximum size of each dimension of a block
        int    maxGridSize[3];            // Maximum size of each dimension of a grid
        int    clockRate;                 // Clock frequency in kilohertz
        size_t totalConstMem;             // Constant memory available on device in bytes
        int    major;                     // Major compute capability
        int    minor;                     // Minor compute capability
        size_t textureAlignment;          // Alignment requirement for textures
        int    deviceOverlap;             // Device can concurrently copy memory and execute a kernel
        int    multiProcessorCount;       // Number of multiprocessors on device
        int    kernelExecTimeoutEnabled;  // Specified whether there is a run time limit on kernels
        int    integrated;                // Device is integrated as opposed to discrete
        int    canMapHostMemory;          // Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
        int    computeMode;               // Compute mode (See ::cudaComputeMode)
        int    maxTexture1D;              // Maximum 1D texture size
        int    maxTexture2D[2];           // Maximum 2D texture dimensions
        int    maxTexture3D[3];           // Maximum 3D texture dimensions
        int    maxTexture2DArray[3];      // Maximum 2D texture array dimensions
        size_t surfaceAlignment;          // Alignment requirements for surfaces
        int    concurrentKernels;         // Device can possibly execute multiple kernels concurrently
        int    ECCEnabled;                // Device has ECC support enabled
        int    pciBusID;                  // PCI bus ID of the device
        int    pciDeviceID;               // PCI device ID of the device
        int    __cudaReserved[22];
    };

  protected:
    typedef CudaGPUParams Params;

    /**
     *  Helper class for both Stream and GPU tick events
     */
    class TickEvent : public Event
    {
        friend class CudaGPU;

      private:
        CudaGPU *cpu;

      public:
        TickEvent(CudaGPU *c) : Event(CPU_Tick_Pri), cpu(c) {}
        void process() {
            cpu->streamTick();
        }
        virtual const char *description() const { return "CudaGPU tick"; }
    };

    class FinishKernelEvent : public Event
    {
        friend class CudaGPU;

    private:
        CudaGPU *gpu;
        int gridId;
    public:
        FinishKernelEvent(CudaGPU *_gpu, int grid_id) :
            gpu(_gpu), gridId(grid_id)
        {
            setFlags(Event::AutoDelete);
        }
        void process() {
            gpu->processFinishKernelEvent(gridId);
        }
    };

    const CudaGPUParams *_params;
    const Params * params() const { return dynamic_cast<const Params *>(_params); }

    /// Tick for when the stream manager needs execute
    TickEvent streamTickEvent;

  private:
    // The CUDA device ID for this GPU
    unsigned cudaDeviceID;

    // Clock domain for the GPU: Used for changing frequency
    SrcClockDomain *clkDomain;

    // Wrappers to cycle components in GPGPU-Sim
    GPGPUSimComponentWrapper &coresWrapper;
    GPGPUSimComponentWrapper &icntWrapper;
    GPGPUSimComponentWrapper &l2Wrapper;
    GPGPUSimComponentWrapper &dramWrapper;

    /// Callback for the stream manager tick
    void streamTick();

    /// Pointer to the copy engine for this device
    GPUCopyEngine *copyEngine;

    /// Used to register this SPA with the system
    System *system;

    /// Number of threads in each warp, also number of lanes per CUDA core
    int warpSize;

    /// Are we restoring from a checkpoint?
    bool restoring;

    int sharedMemDelay;
    std::string gpgpusimConfigPath;
    Tick launchDelay;
    Tick returnDelay;

    /// If true there is a kernel currently executing
    /// NOTE: Jason doesn't think we need this
    bool running;

    /// True if the running thread is currently blocked and needs to be activated
    bool unblockNeeded;

    /// Pointer to ruby system used to clear the Ruby stats
    /// NOTE: I think there is a more right way to do this
    RubySystem *ruby;

    /// Holds all of the CUDA cores in this GPU
    std::vector<CudaCore*> cudaCores;

    /// The thread context, stream and thread ID currently running on the SPA
    ThreadContext *runningTC;
    struct CUstream_st *runningStream;
    int runningTID;
    Addr runningPTBase;
    void beginStreamOperation(struct CUstream_st *_stream) {
        // We currently do not support multiple concurrent streams
        if (runningStream || runningTC) {
            panic("Already a stream operation running (only support one at a time)!");
        }
        // NOTE: This may cause a race: The runningTC may have changed (i.e.
        // the thread was migrated) between when the thread queued the stream
        // operation and when that operation starts executing here. By reading
        // CR3 here, we could use this to double check that the correct thread
        // is running. On the other hand, we could move the CR3 read into the
        // operation queuing code to avoid the race, but we would not be able
        // to detect of the thread had migrated since it queued the operation.
        runningStream = _stream;
        runningTC = runningStream->getThreadContext();
        runningTID = runningTC->threadId();
#if THE_ISA == X86_ISA
        runningPTBase = runningTC->readMiscRegNoEffect(X86ISA::MISCREG_CR3);
#else
        // TODO: ARM ISA should use the TTBCR for user space (which appears
        // to be called the TTBR1 register). Further investigation required.
        warn_once("ISA's pagetable base register handling needs to be set up");
#endif
    }
    void endStreamOperation() {
        runningStream = NULL;
        runningTC = NULL;
        runningTID = -1;
        runningPTBase = 0;
    }

    /// For statistics
    std::vector<Tick> kernelTimes;
    Tick clearTick;
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
    Addr localBaseVaddr;

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

    class GPUPageTable
    {
      private:
        std::map<Addr, Addr> pageMap;

      public:
        GPUPageTable() {};

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
        void serialize(CheckpointOut &cp) const;
        void unserialize(CheckpointIn &cp);
    };
    GPUPageTable pageTable;
    bool manageGPUMemory;
    bool accessHostPageTable;
    AddrRange gpuMemoryRange;
    Addr physicalGPUBrkAddr;
    Addr virtualGPUBrkAddr;
    std::map<Addr,size_t> allocatedGPUMemory;

    ShaderMMU *shaderMMU;

    CudaDeviceProperties deviceProperties;

  public:
    /// Constructor
    CudaGPU(const Params *p);

    /// For checkpointing
    virtual void serialize(CheckpointOut &cp) const;
    virtual void unserialize(CheckpointIn &cp);

    /// Called after constructor, but before any real simulation
    virtual void startup();

    /// Register devices callbacks
    void registerCudaCore(CudaCore *sc);
    void registerCopyEngine(GPUCopyEngine *ce);

    /// Getter for whether we are using Ruby or GPGPU-Sim memory modeling
    CudaDeviceProperties *getDeviceProperties() { return &deviceProperties; }
    unsigned getMaxThreadsPerMultiprocessor() {
        if (deviceProperties.major == 2) {
            warn("Returning threads per multiprocessor from compute capability 2.x\n");
            return 1536;
        }
        panic("Have not configured threads per multiprocessor!\n");
        return 0;
    }
    int getSharedMemDelay() { return sharedMemDelay; }
    const char* getConfigPath() { return gpgpusimConfigPath.c_str(); }
    RubySystem* getRubySystem() { return ruby; }
    gpgpu_sim* getTheGPU() { return theGPU; }

    /// Called at the beginning of each kernel launch to start the statistics
    void beginRunning(Tick stream_queued_time, struct CUstream_st *_stream);

    /**
     * Marks the kernel as complete and signals the stream manager
     */
    void processFinishKernelEvent(int grid_id);

    /**
     * Called from GPGPU-Sim when the kernel completes on all shaders
     */
    void finishKernel(int grid_id);

    void handleFinishPageFault(ThreadContext *tc)
        { shaderMMU->handleFinishPageFault(tc); }

    ShaderMMU *getMMU() { return shaderMMU; }

    /// Schedules the stream manager to be checked in 'ticks' ticks from now
    void scheduleStreamEvent();

    /// Reset statistics for the SPA and for all of Ruby
    void clearStats();

    /// Returns CUDA core with id coreId
    CudaCore *getCudaCore(int coreId);

    /// Returns size of warp (same for all CUDA cores)
    int getWarpSize() { return warpSize; }

    /// Callback for GPGPU-Sim to get the current simulation time
    Tick getCurTick(){ return curTick(); }

    /// Used to print stats at the end of simulation
    void gpuPrintStats(std::ostream& out);

    void printPTXFileLineStats();

    /// Begins a timing memory copy from src to dst
    void memcpy(void *src, void *dst, size_t count, struct CUstream_st *stream, stream_operation_type type);

    /// Begins a timing memory copy from src to/from the symbol+offset
    void memcpy_to_symbol(const char *hostVar, const void *src, size_t count, size_t offset, struct CUstream_st *stream);
    void memcpy_from_symbol(void *dst, const char *hostVar, size_t count, size_t offset, struct CUstream_st *stream);

    /// Begins a timing memory set of value to dst
    void memset(Addr dst, int value, size_t count, struct CUstream_st *stream);

    /// Called by the copy engine when a memcpy or memset is complete
    void finishCopyOperation();

    /// Called from shader TLB to be used for TLB lookups
    /// TODO: Move the thread context handling to GPU context when we get there
    ThreadContext *getThreadContext() { return runningTC; }
    Addr getRunningPTBase() { return runningPTBase; }
    void checkUpdateThreadContext(ThreadContext *tc) {
        if (!runningTC) {
            // The GPU isn't running anything, so it won't try to access the
            // thread context for anything (e.g. address translations). Hence,
            // it is safe to ignore this check/update process
            return;
        }
        if (tc != runningTC) {
#if THE_ISA == X86_ISA
            Addr pagetable_base = tc->readMiscRegNoEffect(X86ISA::MISCREG_CR3);
#else
            warn_once("ISA's pagetable base needs to be read and checked!");
            Addr pagetable_base = 0;
#endif
            warn("Thread migrated! Old tc: %p, PT: %p, New tc: %p, PT: %p\n",
                 runningTC, runningPTBase, tc, pagetable_base);
            if (pagetable_base == runningPTBase) {
                // No problem, just change migrate the runningTC
                DPRINTF(CudaGPU, "Updating the thread context\n");
                runningTC = tc;
                runningTID = runningTC->threadId();
            } else {
                panic("New pagetable address doesn't match old!\n");
            }
        }
        // NOTE: If we can get away with live updating the thread context
        // pointer while the GPU is executing, we need to make sure that the
        // ShaderMMU and ShaderTLBs use the latest runningTC (i.e. we may need
        // to dynamically update the tc of in-flight translations, and squash
        // those that are in page-walks). This possibility seems unlikely.
    }

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
    void setLocalBaseVaddr(uint64_t addr);
    uint64_t getLocalBaseVaddr();

    /// For handling GPU memory mapping table
    GPUPageTable* getGPUPageTable() { return &pageTable; };
    void registerDeviceMemory(ThreadContext *tc, Addr vaddr, size_t size);
    void registerDeviceInstText(ThreadContext *tc, Addr vaddr, size_t size);
    bool isManagingGPUMemory() { return manageGPUMemory; }
    bool isAccessingHostPagetable() { return accessHostPageTable; }
    Addr allocateGPUMemory(size_t size);

    /// Statistics for this GPU
    Stats::Scalar numKernelsStarted;
    Stats::Scalar numKernelsCompleted;
    void regStats();
};

/**
 *  Helper class to print out statistics at the end of simulation
 */
class GPUExitCallback : public Callback
{
  private:
    std::string stats_filename;
    CudaGPU *gpu;

  public:
    virtual ~GPUExitCallback() {}

    GPUExitCallback(CudaGPU *_gpu, const std::string& _stats_filename)
    {
        stats_filename = _stats_filename;
        gpu = _gpu;
    }

    virtual void process();
};

#endif

