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
#include <sstream>
#include <string>
#include <map>

#include "arch/tlb.hh"
#include "arch/utility.hh"
#include "api/gpu_syscall_helper.hh"
#include "base/chunk_generator.hh"
#include "base/output.hh"
#include "config/the_isa.hh"
#include "cpu/thread_context.hh"
#include "cpu/translation.hh"
#include "debug/GpuTick.hh"
#include "debug/StreamProcessorArray.hh"
#include "debug/StreamProcessorArrayAccess.hh"
#include "debug/StreamProcessorArrayPageTable.hh"
#include "debug/StreamProcessorArrayTick.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "mem/ruby/system/System.hh"
#include "mem/page_table.hh"
#include "params/StreamProcessorArray.hh"
#include "sim/pseudo_inst.hh"

#include "../gpgpu-sim/cuda-sim/cuda-sim.h"
#include "../gpgpu-sim/gpgpusim_entrypoint.h"

using namespace TheISA;
using namespace std;

vector<StreamProcessorArray*> StreamProcessorArray::gpuArray;

// From GPU syscalls
void registerFatBinaryTop(GPUSyscallHelper *helper, Addr sim_fatCubin, size_t sim_binSize);
unsigned int registerFatBinaryBottom(GPUSyscallHelper *helper, Addr sim_alloc_ptr);
void register_var(Addr sim_deviceAddress, const char* deviceName, int sim_size, int sim_constant, int sim_global, int sim_ext, Addr sim_hostVar);

StreamProcessorArray::StreamProcessorArray(const Params *p) :
    SimObject(p), _params(p), gpuTickEvent(this, false), streamTickEvent(this, true),
    system(p->sys), frequency(p->frequency), warpSize(p->warp_size), sharedMemDelay(p->shared_mem_delay),
    gpgpusimConfigPath(p->config_path), launchDelay(p->kernel_launch_delay),
    returnDelay(p->kernel_return_delay), ruby(p->ruby), runningTC(NULL),
    runningStream(NULL), runningTID(-1), clearTick(0),
    dumpKernelStats(p->dump_kernel_stats), pageTable(),
    manageGPUMemory(p->manage_gpu_memory),
    gpuMemoryRange(p->gpu_memory_range)
{
    // Register this device as a CUDA-enabled GPU
    cudaDeviceID = registerCudaDevice(this);
    if (cudaDeviceID >= 1) {
        // TODO: Remove this when multiple GPUs can exist in system
        panic("GPGPU-Sim is not currently able to simulate more than 1 CUDA-enabled GPU\n");
    }

    streamDelay = 1;

    running = false;

    streamScheduled = false;

    restoring = false;

    // GPU memory handling
    instBaseVaddr = 0;
    instBaseVaddrSet = false;
    // Reserve the 0 virtual page for NULL pointers
    virtualGPUBrkAddr = TheISA::PageBytes;
    physicalGPUBrkAddr = gpuMemoryRange.start();

    // Initialize GPGPU-Sim
    theGPU = gem5_ptx_sim_init_perf(&streamManager, getSharedMemDelay(), getConfigPath());
    theGPU->init();
    theGPU->setSPA(this);

    // Setup the device properties for this GPU
    snprintf(deviceProperties.name, 256, "GPGPU-Sim_v%s", g_gpgpusim_version_string);
    deviceProperties.major = 2;
    deviceProperties.minor = 0;
    deviceProperties.totalGlobalMem = gpuMemoryRange.size();
    deviceProperties.memPitch = 0;
    deviceProperties.maxThreadsPerBlock = 512;
    deviceProperties.maxThreadsDim[0] = 512;
    deviceProperties.maxThreadsDim[1] = 512;
    deviceProperties.maxThreadsDim[2] = 512;
    deviceProperties.maxGridSize[0] = 0x40000000;
    deviceProperties.maxGridSize[1] = 0x40000000;
    deviceProperties.maxGridSize[2] = 0x40000000;
    deviceProperties.totalConstMem = gpuMemoryRange.size();
    deviceProperties.textureAlignment = 0;
    deviceProperties.sharedMemPerBlock = theGPU->shared_mem_size();
    deviceProperties.regsPerBlock = theGPU->num_registers_per_core();
    deviceProperties.warpSize = theGPU->wrp_size();
    deviceProperties.clockRate = theGPU->shader_clock();
#if (CUDART_VERSION >= 2010)
    deviceProperties.multiProcessorCount = theGPU->get_config().num_shader();
#endif

    // Print gpu configuration and stats at exit
    GPUExitCallback* gpuExitCB = new GPUExitCallback(this, p->stats_filename);
    registerExitCallback(gpuExitCB);

}

void StreamProcessorArray::serialize(std::ostream &os)
{
    DPRINTF(StreamProcessorArray, "Serializing\n");
    if (running) {
        panic("Checkpointing during GPU execution not supported\n");
    }

    SERIALIZE_SCALAR(m_last_fat_cubin_handle);
    SERIALIZE_SCALAR(instBaseVaddr);

    SERIALIZE_SCALAR(runningTID);

    int numBinaries = fatBinaries.size();
    SERIALIZE_SCALAR(numBinaries);
    for (int i=0; i<numBinaries; i++) {
        stringstream ss;
        ss << i;
        string num = ss.str();
        paramOut(os, num+"fatBinaries.tid", fatBinaries[i].tid);
        paramOut(os, num+"fatBinaries.handle", fatBinaries[i].handle);
        paramOut(os, num+"fatBinaries.sim_fatCubin", fatBinaries[i].sim_fatCubin);
        paramOut(os, num+"fatBinaries.sim_binSize", fatBinaries[i].sim_binSize);
        paramOut(os, num+"fatBinaries.sim_alloc_ptr", fatBinaries[i].sim_alloc_ptr);

        paramOut(os, num+"fatBinaries.funcMap.size", fatBinaries[i].funcMap.size());
        std::map<const void*,string>::iterator it;
        int j = 0;
        for (it=fatBinaries[i].funcMap.begin(); it!=fatBinaries[i].funcMap.end(); it++) {
            paramOut(os, csprintf("%dfatBinaries.funcMap[%d].first", i, j), (uint64_t)it->first);
            paramOut(os, csprintf("%dfatBinaries.funcMap[%d].second", i, j), it->second);
            j++;
        }
    }

    int numVars = cudaVars.size();
    SERIALIZE_SCALAR(numVars);
    for (int i=0; i<numVars; i++) {
        _CudaVar var = cudaVars[i];
        paramOut(os, csprintf("cudaVars[%d].sim_deviceAddress", i), var.sim_deviceAddress);
        paramOut(os, csprintf("cudaVars[%d].deviceName", i), var.deviceName);
        paramOut(os, csprintf("cudaVars[%d].sim_size", i), var.sim_size);
        paramOut(os, csprintf("cudaVars[%d].sim_constant", i), var.sim_constant);
        paramOut(os, csprintf("cudaVars[%d].sim_global", i), var.sim_global);
        paramOut(os, csprintf("cudaVars[%d].sim_ext", i), var.sim_ext);
        paramOut(os, csprintf("cudaVars[%d].sim_hostVar", i), var.sim_hostVar);
    }

    pageTable.serialize(os);
}

void StreamProcessorArray::unserialize(Checkpoint *cp, const std::string &section)
{
    DPRINTF(StreamProcessorArray, "UNserializing\n");

    restoring = true;

    UNSERIALIZE_SCALAR(m_last_fat_cubin_handle);
    UNSERIALIZE_SCALAR(instBaseVaddr);

    UNSERIALIZE_SCALAR(runningTID);

    DPRINTF(StreamProcessorArray, "UNSerializing %d, %d\n", m_last_fat_cubin_handle, instBaseVaddr);

    int numBinaries;
    UNSERIALIZE_SCALAR(numBinaries);
    DPRINTF(StreamProcessorArray, "UNserializing %d binaries\n", numBinaries);
    fatBinaries.resize(numBinaries);
    for (int i=0; i<numBinaries; i++) {
        stringstream ss;
        ss << i;
        string num = ss.str();
        paramIn(cp, section, num+"fatBinaries.tid", fatBinaries[i].tid);
        paramIn(cp, section, num+"fatBinaries.handle", fatBinaries[i].handle);
        paramIn(cp, section, num+"fatBinaries.sim_fatCubin", fatBinaries[i].sim_fatCubin);
        paramIn(cp, section, num+"fatBinaries.sim_binSize", fatBinaries[i].sim_binSize);
        paramIn(cp, section, num+"fatBinaries.sim_alloc_ptr", fatBinaries[i].sim_alloc_ptr);
        DPRINTF(StreamProcessorArray, "Got %d %d %d %d\n", fatBinaries[i].handle, fatBinaries[i].sim_fatCubin, fatBinaries[i].sim_binSize, fatBinaries[i].sim_alloc_ptr);

        int funcMapSize;
        paramIn(cp, section, num+"fatBinaries.funcMap.size", funcMapSize);
        for (int j=0; j<funcMapSize; j++) {
            uint64_t first;
            string second;
            paramIn(cp, section, csprintf("%dfatBinaries.funcMap[%d].first", i, j), first);
            paramIn(cp, section, csprintf("%dfatBinaries.funcMap[%d].second", i, j), second);
            fatBinaries[i].funcMap[(const void*)first] = second;
        }
    }

    int numVars;
    UNSERIALIZE_SCALAR(numVars);
    cudaVars.resize(numVars);
    for (int i=0; i<numVars; i++) {
        paramIn(cp, section, csprintf("cudaVars[%d].sim_deviceAddress", i), cudaVars[i].sim_deviceAddress);
        paramIn(cp, section, csprintf("cudaVars[%d].deviceName", i), cudaVars[i].deviceName);
        paramIn(cp, section, csprintf("cudaVars[%d].sim_size", i), cudaVars[i].sim_size);
        paramIn(cp, section, csprintf("cudaVars[%d].sim_constant", i), cudaVars[i].sim_constant);
        paramIn(cp, section, csprintf("cudaVars[%d].sim_global", i), cudaVars[i].sim_global);
        paramIn(cp, section, csprintf("cudaVars[%d].sim_ext", i), cudaVars[i].sim_ext);
        paramIn(cp, section, csprintf("cudaVars[%d].sim_hostVar", i), cudaVars[i].sim_hostVar);
    }

    pageTable.unserialize(cp, section);
}

void StreamProcessorArray::startup()
{
    // Initialize shader cores
    vector<ShaderCore*>::iterator iter;
    for (iter = shaderCores.begin(); iter != shaderCores.end(); ++iter) {
        (*iter)->initialize();
    }

    if (!restoring) {
        return;
    }

    if (runningTID >= 0) {
        runningTC = system->getThreadContext(runningTID);
        assert(runningTC);
    }

    // Setting everything up again!
    std::vector<_FatBinary>::iterator binaries;
    for (binaries = fatBinaries.begin(); binaries != fatBinaries.end(); ++binaries) {
        _FatBinary bin = *binaries;
        GPUSyscallHelper helper(system->getThreadContext(bin.tid));
        assert(helper.getThreadContext());
        registerFatBinaryTop(&helper, bin.sim_fatCubin, bin.sim_binSize);
        registerFatBinaryBottom(&helper, bin.sim_alloc_ptr);

        std::map<const void*, string>::iterator functions;
        for (functions = bin.funcMap.begin(); functions != bin.funcMap.end(); ++functions) {
            const char *host_fun = (const char*)functions->first;
            const char *device_fun = functions->second.c_str();
            register_function(bin.handle, host_fun, device_fun);
        }
    }

    std::vector<_CudaVar>::iterator variables;
    for (variables = cudaVars.begin(); variables != cudaVars.end(); ++variables) {
        _CudaVar var = *variables;
        register_var(var.sim_deviceAddress, var.deviceName.c_str(), var.sim_size, var.sim_constant, var.sim_global, var.sim_ext, var.sim_hostVar);
    }
}

void StreamProcessorArray::clearStats()
{
    ruby->resetStats();
    clearTick = curTick();
}

void StreamProcessorArray::registerShaderCore(ShaderCore *sc)
{
    shaderCores.push_back(sc);
}

void StreamProcessorArray::registerCopyEngine(SPACopyEngine *ce)
{
    copyEngine = ce;
}

void StreamProcessorArray::gpuTick()
{
    DPRINTF(GpuTick, "GPU Tick\n");

    // check if a kernel has completed
    // TODO: Cleanup this code to schedule an event for a completed kernel
    kernelTermInfo term_info = theGPU->finished_kernel();
    if( term_info.grid_uid ) {
        Tick delay = 1;
        Tick curTime = curTick();
        if (curTime - term_info.time < returnDelay * SimClock::Frequency ) {
            delay = (Tick)(returnDelay * SimClock::Frequency) - (curTime - term_info.time); //delay by whatever is left over
        }

        finishedKernels.push(kernelTermInfo(term_info.grid_uid, curTick()+delay));
        streamRequestTick(1);

        running = false;
    }

    while(!finishedKernels.empty() && finishedKernels.front().time < curTick()) {
        DPRINTF(StreamProcessorArrayTick, "GPU finished a kernel id %d\n", finishedKernels.front().grid_uid);

        streamManager->register_finished_kernel(finishedKernels.front().grid_uid);
        finishedKernels.pop();

        kernelTimes.push_back(curTick());
        if (dumpKernelStats) {
            PseudoInst::dumpresetstats(runningTC, 0, 0);
        }

        if (unblockNeeded && streamManager->empty() && finishedKernels.empty()) {
            DPRINTF(StreamProcessorArray, "Stream manager is empty, unblocking\n");
            unblockThread(runningTC);
        }

        endStreamOperation();
    }

    // simulate a clock cycle on the GPU
    if( theGPU->active() ) {
        theGPU->cycle();
    } else {
        if(!finishedKernels.empty()) {
            schedule(gpuTickEvent, finishedKernels.front().time + 1);
        }
    }
    theGPU->deadlock_check();

    if (streamManager->ready() && !streamScheduled) {
        schedule(streamTickEvent, curTick() + streamDelay);
        streamScheduled = true;
    }

}

void StreamProcessorArray::streamTick() {
    DPRINTF(StreamProcessorArrayTick, "Stream Tick\n");

    streamScheduled = false;

    // launch operation on device if one is pending and can be run
    stream_operation op = streamManager->front();
    op.do_operation(theGPU);

    if (streamManager->ready()) {
        schedule(streamTickEvent, curTick() + streamDelay);
        streamScheduled = true;
    }
}

void StreamProcessorArray::gpuRequestTick(float gpuTicks) {
    Tick gpuWakeupTick = (int)(gpuTicks * SimClock::Frequency) + curTick();

    schedule(gpuTickEvent, gpuWakeupTick);
}

void StreamProcessorArray::streamRequestTick(int ticks) {
    if (streamScheduled) {
        DPRINTF(StreamProcessorArrayTick, "Already scheduled a tick, ignoring\n");
        return;
    }
    Tick streamWakeupTick = ticks + curTick();

    schedule(streamTickEvent, streamWakeupTick);
    streamScheduled = true;
}

void StreamProcessorArray::beginRunning(Tick launchTime, struct CUstream_st *_stream)
{
    beginStreamOperation(_stream);

    DPRINTF(StreamProcessorArray, "Beginning kernel execution at %llu\n", curTick());
    kernelTimes.push_back(curTick());
    if (dumpKernelStats) {
        PseudoInst::dumpresetstats(runningTC, 0, 0);
    }
    if (running) {
        panic("Should not already be running if we are starting\n");
    }
    running = true;

    Tick delay = 1;
    Tick curTime = curTick();
    if (curTime - launchTime < launchDelay * SimClock::Frequency) {
        delay = (Tick)(launchDelay * SimClock::Frequency) - (curTime - launchTime); //delay by whatever is left over
    }

    schedule(gpuTickEvent, curTick() + delay);
}

ShaderCore *StreamProcessorArray::getShaderCore(int coreId)
{
    assert(coreId < shaderCores.size());
    return shaderCores[coreId];
}

StreamProcessorArray *StreamProcessorArrayParams::create() {
    return new StreamProcessorArray(this);
}

void StreamProcessorArray::gpuPrintStats(std::ostream& out) {
    // Print kernel statistics
    Tick total_kernel_ticks = 0;
    Tick last_kernel_time = 0;
    bool kernel_active = false;
    vector<Tick>::iterator it;

    out << "spa frequency: " << SimClock::Frequency/(frequency*1000000000.0) << " GHz\n";
    out << "spa period: " << frequency << " ticks\n";
    out << "kernel times (ticks):\n";
    out << "start, end, start, end, ..., exit\n";
    for (it = kernelTimes.begin(); it < kernelTimes.end(); it++) {
        out << *it << ", ";
        if (kernel_active) {
            total_kernel_ticks += (*it - last_kernel_time);
            kernel_active = false;
        } else {
            last_kernel_time = *it;
            kernel_active = true;
        }
    }
    out << curTick() << "\n";

    // Print Shader CTA statistics
    out << "\nshader CTA times (ticks):\n";
    out << "shader, CTA ID, start, end, start, end, ..., exit\n";
    std::vector<ShaderCore*>::iterator shaders;
    for (shaders = shaderCores.begin(); shaders != shaderCores.end(); shaders++) {
        (*shaders)->printCTAStats(out);
    }
    out << "\ntotal kernel time (ticks) = " << total_kernel_ticks << "\n";

    if (clearTick) {
        out << "Stats cleared at tick " << clearTick << "\n";
    }
}

void StreamProcessorArray::memcpy(void *src, void *dst, size_t count, struct CUstream_st *_stream, stream_operation_type type) {
    beginStreamOperation(_stream);
    copyEngine->memcpy((Addr)src, (Addr)dst, count, type);
}

void StreamProcessorArray::memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to, struct CUstream_st *_stream) {
    // First, initialize the stream operation
    beginStreamOperation(_stream);

    // Lookup destination address for transfer:
    std::string sym_name = gpgpu_ptx_sim_hostvar_to_sym_name(hostVar);
    std::map<std::string,symbol_table*>::iterator st = g_sym_name_to_symbol_table.find(sym_name.c_str());
    assert( st != g_sym_name_to_symbol_table.end() );
    symbol_table *symtab = st->second;

    symbol *sym = symtab->lookup(sym_name.c_str());
    assert(sym);
    unsigned dst = sym->get_address() + offset;
    printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %zu bytes %s symbol %s+%zu @0x%x ...\n",
           count, (to ? "to" : "from"), sym_name.c_str(), offset, dst);

    if (to) {
        copyEngine->memcpy((Addr)src, (Addr)dst, count, stream_memcpy_host_to_device);
    } else {
        copyEngine->memcpy((Addr)dst, (Addr)src, count, stream_memcpy_device_to_host);
    }
}

void StreamProcessorArray::memset(Addr dst, int value, size_t count, struct CUstream_st *_stream) {
    beginStreamOperation(_stream);
    copyEngine->memset(dst, value, count);
}

void StreamProcessorArray::finishCopyOperation()
{
    runningStream->record_next_done();
    streamRequestTick(1);
    unblockThread(runningTC);
    endStreamOperation();
}

// TODO: When we move the stream manager into libcuda, this will need to be
// eliminated, and libcuda will have to decide when to block the calling thread
bool StreamProcessorArray::needsToBlock()
{
    if (!streamManager->empty()) {
        DPRINTF(StreamProcessorArray, "Suspend request: Need to activate CPU later\n");
        unblockNeeded = true;
        streamManager->print(stdout);
        return true;
    } else {
        DPRINTF(StreamProcessorArray, "Suspend request: Already done.\n");
        return false;
    }
}

void StreamProcessorArray::blockThread(ThreadContext *tc, Addr signal_ptr)
{
    if (streamManager->empty()) {
        // It is common in small memcpys for the stream operation to be complete
        // by the time cudaMemcpy calls blockThread. In this case, just signal
        DPRINTF(StreamProcessorArray, "No stream operations to block thread %p. Continuing...\n", tc);
        signalThread(tc, signal_ptr);
        blockedThreads.erase(tc);
        unblockNeeded = false;
    } else {
        DPRINTF(StreamProcessorArray, "Blocking thread %p for GPU syscall\n", tc);
        blockedThreads[tc] = signal_ptr;
        tc->suspend();
    }
}

void StreamProcessorArray::signalThread(ThreadContext *tc, Addr signal_ptr)
{
    GPUSyscallHelper helper(tc);
    bool signal_val = true;

    // Read signal value and ensure that it is currently false
    // (i.e. thread should be currently blocked)
    helper.readBlob(signal_ptr, (uint8_t*)&signal_val, sizeof(bool));
    if (signal_val) {
        panic("Thread doesn't appear to be blocked!\n");
    }

    signal_val = true;
    helper.writeBlob(signal_ptr, (uint8_t*)&signal_val, sizeof(bool));
}

void StreamProcessorArray::unblockThread(ThreadContext *tc)
{
    if (!tc) tc = runningTC;
    if (tc->status() != ThreadContext::Suspended) return;
    assert(unblockNeeded);

    DPRINTF(StreamProcessorArray, "Unblocking thread %p for GPU syscall\n", tc);
    std::map<ThreadContext*, Addr>::iterator tc_iter = blockedThreads.find(tc);
    if (tc_iter == blockedThreads.end()) {
        panic("Cannot find blocked thread!\n");
    }

    Addr signal_ptr = blockedThreads[tc];
    signalThread(tc, signal_ptr);

    blockedThreads.erase(tc);
    unblockNeeded = false;
    tc->activate();
}

void StreamProcessorArray::add_binary( symbol_table *symtab, unsigned fat_cubin_handle )
{
    m_code[fat_cubin_handle] = symtab;
    m_last_fat_cubin_handle = fat_cubin_handle;
}

void StreamProcessorArray::add_ptxinfo( const char *deviceFun, const struct gpgpu_ptx_sim_kernel_info info )
{
    symbol *s = m_code[m_last_fat_cubin_handle]->lookup(deviceFun);
    assert( s != NULL );
    function_info *f = s->get_pc();
    assert( f != NULL );
    f->set_kernel_info(info);
}

void StreamProcessorArray::register_function( unsigned fat_cubin_handle, const char *hostFun, const char *deviceFun )
{
    if( m_code.find(fat_cubin_handle) != m_code.end() ) {
        symbol *s = m_code[fat_cubin_handle]->lookup(deviceFun);
        assert( s != NULL );
        function_info *f = s->get_pc();
        assert( f != NULL );
        m_kernel_lookup[hostFun] = f;
    } else {
        m_kernel_lookup[hostFun] = NULL;
    }
}

function_info *StreamProcessorArray::get_kernel(const char *hostFun)
{
    std::map<const void*,function_info*>::iterator i=m_kernel_lookup.find(hostFun);
    assert( i != m_kernel_lookup.end() );
    return i->second;
}

void StreamProcessorArray::setInstBaseVaddr(uint64_t addr)
{
    if (!instBaseVaddrSet) {
        instBaseVaddr = addr;
        instBaseVaddrSet = true;
    }
}

uint64_t StreamProcessorArray::getInstBaseVaddr()
{
    return instBaseVaddr;
}

Addr StreamProcessorArray::SPAPageTable::addrToPage(Addr addr)
{
    Addr offset = addr % TheISA::PageBytes;
    return addr - offset;
}

void StreamProcessorArray::SPAPageTable::serialize(std::ostream &os)
{
    unsigned int num_ptes = pageMap.size();
    unsigned int index = 0;
    Addr* pagetable_vaddrs = new Addr[num_ptes];
    Addr* pagetable_paddrs = new Addr[num_ptes];
    std::map<Addr, Addr>::iterator it;
    for (it = pageMap.begin(); it != pageMap.end(); ++it) {
        pagetable_vaddrs[index] = (*it).first;
        pagetable_paddrs[index++] = (*it).second;
    }
    SERIALIZE_SCALAR(num_ptes);
    SERIALIZE_ARRAY(pagetable_vaddrs, num_ptes);
    SERIALIZE_ARRAY(pagetable_paddrs, num_ptes);
    delete[] pagetable_vaddrs;
    delete[] pagetable_paddrs;
}

void StreamProcessorArray::SPAPageTable::unserialize(Checkpoint *cp, const std::string &section)
{
    unsigned int num_ptes = 0;
    UNSERIALIZE_SCALAR(num_ptes);
    Addr* pagetable_vaddrs = new Addr[num_ptes];
    Addr* pagetable_paddrs = new Addr[num_ptes];
    UNSERIALIZE_ARRAY(pagetable_vaddrs, num_ptes);
    UNSERIALIZE_ARRAY(pagetable_paddrs, num_ptes);
    for (unsigned int i = 0; i < num_ptes; ++i) {
        pageMap[pagetable_vaddrs[i]] = pagetable_paddrs[i];
    }
    delete[] pagetable_vaddrs;
    delete[] pagetable_paddrs;
}

void StreamProcessorArray::registerDeviceMemory(ThreadContext *tc, Addr vaddr, size_t size)
{
    if (manageGPUMemory) return;
    DPRINTF(StreamProcessorArrayPageTable, "Registering device memory vaddr: %x, size: %d\n", vaddr, size);
    // Get the physical address of full memory allocation (i.e. all pages)
    Addr page_vaddr, page_paddr;
    for (ChunkGenerator gen(vaddr, size, TheISA::PageBytes); !gen.done(); gen.next()) {
        page_vaddr = pageTable.addrToPage(gen.addr());
        if (FullSystem) {
            page_paddr = TheISA::vtophys(tc, page_vaddr);
        } else {
            tc->getProcessPtr()->pTable->translate(page_vaddr, page_paddr);
        }
        pageTable.insert(page_vaddr, page_paddr);
    }
}

void StreamProcessorArray::registerDeviceInstText(ThreadContext *tc, Addr vaddr, size_t size)
{
    if (manageGPUMemory) {
        // Allocate virtual and physical memory for the device text
        Addr gpu_vaddr = allocateGPUMemory(size);
        setInstBaseVaddr(gpu_vaddr);
    } else {
        setInstBaseVaddr(vaddr);
        registerDeviceMemory(tc, vaddr, size);
    }
}

Addr StreamProcessorArray::allocateGPUMemory(size_t size)
{
    assert(manageGPUMemory);
    DPRINTF(StreamProcessorArrayPageTable, "GPU allocating %d bytes\n", size);

    if (size == 0) return 0;

    // TODO: When we need to reclaim memory, this will need to be modified
    // heavily to actually track allocated and free physical and virtual memory

    // Cache block align the allocation size
    size_t block_part = size % ruby->getBlockSizeBytes();
    size_t aligned_size = size + (block_part ? (ruby->getBlockSizeBytes() - block_part) : 0);

    Addr base_vaddr = virtualGPUBrkAddr;
    virtualGPUBrkAddr += aligned_size;
    Addr base_paddr = physicalGPUBrkAddr;
    physicalGPUBrkAddr += aligned_size;

    if (virtualGPUBrkAddr > gpuMemoryRange.size()) {
        panic("Ran out of GPU memory!");
    }

    // Map pages to physical pages
    for (ChunkGenerator gen(base_vaddr, aligned_size, TheISA::PageBytes); !gen.done(); gen.next()) {
        Addr page_vaddr = pageTable.addrToPage(gen.addr());
        Addr page_paddr;
        if (page_vaddr <= base_vaddr) {
            page_paddr = base_paddr - (base_vaddr - page_vaddr);
        } else {
            page_paddr = base_paddr + (page_vaddr - base_vaddr);
        }
        DPRINTF(StreamProcessorArrayPageTable, "  Trying to allocate page at vaddr %x with addr %x\n", page_vaddr, gen.addr());
        pageTable.insert(page_vaddr, page_paddr);
    }

    DPRINTF(StreamProcessorArrayAccess, "Allocating %d bytes for GPU at address 0x%x\n", size, base_vaddr);

    return base_vaddr;
}

/**
* virtual process function that is invoked when the callback
* queue is executed.
*/
void GPUExitCallback::process()
{
    std::ostream *os = simout.find(stats_filename);
    if (!os) {
        os = simout.create(stats_filename);
    }
    spa_obj->gpuPrintStats(*os);
    *os << std::endl;
}

