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
#include "base/chunk_generator.hh"
#include "base/output.hh"
#include "config/the_isa.hh"
#include "cpu/thread_context.hh"
#include "cpu/translation.hh"
#include "debug/GpuTick.hh"
#include "debug/StreamProcessorArray.hh"
#include "debug/StreamProcessorArrayAccess.hh"
#include "debug/StreamProcessorArrayTick.hh"
#include "mem/ruby/system/System.hh"
#include "mem/page_table.hh"
#include "params/StreamProcessorArray.hh"
#include "sim/pseudo_inst.hh"

#include "sp_array.hh"

#include "../gpgpu-sim/cuda-sim/cuda-sim.h"

using namespace TheISA;
using namespace std;

// From GPU syscalls
void registerFatBinaryTop(Addr sim_fatCubin, size_t sim_binSize, ThreadContext *tc);
unsigned int registerFatBinaryBottom(addr_t sim_alloc_ptr);
void register_var(Addr sim_deviceAddress, const char* deviceName, int sim_size, int sim_constant, int sim_global, int sim_ext, Addr sim_hostVar);
class _cuda_device_id *GPGPUSim_Init(ThreadContext *tc);

StreamProcessorArray* StreamProcessorArray::singletonPointer = NULL;

StreamProcessorArray::StreamProcessorArray(const Params *p) :
        SimObject(p), _params(p), gpuTickEvent(this, false), streamTickEvent(this, true),
        copyEngine(p->ce), system(p->sys), useGem5Mem(p->useGem5Mem),
        sharedMemDelay(p->sharedMemDelay), gpgpusimConfigPath(p->config_path),
        launchDelay(p->launchDelay), returnDelay(p->returnDelay), ruby(p->ruby),
        gpuTickConversion(p->gpuTickConv), clearTick(0),
        dumpKernelStats(p->dump_kernel_stats)
{
    streamDelay = 1;
    assert(singletonPointer == NULL);
    singletonPointer = this;

    running = false;

    numShaderCores = 0;

    streamScheduled = false;

    // start our brk point at 2GB. Hopefully this won't clash with what the
    // OS is doing. See arch/x86/process.cc for what it's doing.
    // only used for design point 1 where the CPU and GPU have partitioned mem
    brk_point = 0x8000000;
    // Start giving constant addresses at offset of 0x100 to match GPGPU-Sim
    nextAddr = 0x8000100;

    restoring = false;

    //
    // Print gpu configuration and stats at exit
    //
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
    SERIALIZE_SCALAR(m_inst_base_vaddr);

    int tid = tc->threadId();
    SERIALIZE_SCALAR(tid);

    int numBinaries = fatBinaries.size();
    SERIALIZE_SCALAR(numBinaries);
    for (int i=0; i<numBinaries; i++) {
        stringstream ss;
        ss << i;
        string num = ss.str();
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
}

void StreamProcessorArray::unserialize(Checkpoint *cp, const std::string &section)
{
    DPRINTF(StreamProcessorArray, "UNserializing\n");

    restoring = true;

    UNSERIALIZE_SCALAR(m_last_fat_cubin_handle);
    UNSERIALIZE_SCALAR(m_inst_base_vaddr);

    int tid;
    UNSERIALIZE_SCALAR(tid);

    DPRINTF(StreamProcessorArray, "UNSerializing %d, %d\n", m_last_fat_cubin_handle, m_inst_base_vaddr);

    int numBinaries;
    UNSERIALIZE_SCALAR(numBinaries);
    DPRINTF(StreamProcessorArray, "UNserializing %d binaries\n", numBinaries);
    fatBinaries.resize(numBinaries);
    for (int i=0; i<numBinaries; i++) {
        stringstream ss;
        ss << i;
        string num = ss.str();
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

}

void StreamProcessorArray::startup()
{
    if (!restoring) {
        return;
    }

    tc = system->getThreadContext(tid);
    assert(tc != NULL);
    GPGPUSim_Init(tc);

    // Setting everything up again!
    std::vector<_FatBinary>::iterator it;
    for (it=fatBinaries.begin(); it!=fatBinaries.end(); it++) {
        registerFatBinaryTop((*it).sim_fatCubin, (*it).sim_binSize, tc);
        registerFatBinaryBottom((*it).sim_alloc_ptr);

        std::map<const void*,string>::iterator jt;
        for (jt=(*it).funcMap.begin(); jt!=(*it).funcMap.end(); jt++) {
            register_function((*it).handle, (const char*)jt->first, jt->second.c_str());
        }
    }


    std::vector<_CudaVar>::iterator ij;
    for (ij=cudaVars.begin(); ij!=cudaVars.end(); ij++) {
        register_var((*ij).sim_deviceAddress, (*ij).deviceName.c_str(), (*ij).sim_size, (*ij).sim_constant, (*ij).sim_global, (*ij).sim_ext, (*ij).sim_hostVar);
    }
}


void StreamProcessorArray::clearStats()
{
    ruby->clearStats();
    clearTick = curTick();
}


int StreamProcessorArray::registerShaderCore(ShaderCore *sc)
{
    // I don't think we need this function. I think it will work the way
    // the ruby system object works.
    shaderCores.push_back(sc);
    return numShaderCores++;
}


void StreamProcessorArray::gpuTick()
{
    DPRINTF(GpuTick, "GPU Tick\n");

    // check if a kernel has completed
    kernelTermInfo term_info = theGPU->finished_kernel();
    if( term_info.grid_uid ) {
        Tick delay = 1;
        Tick curTime = curTick();
        if (curTime - term_info.time < returnDelay*gpuTickConversion ) {
            delay = (Tick)(returnDelay*gpuTickConversion) - (curTime - term_info.time); //delay by whatever is left over
        }

        finished_kernels.push(kernelTermInfo(term_info.grid_uid, curTick()+delay));
        streamRequestTick(1);

        running = false;
    }

    while(!finished_kernels.empty() && finished_kernels.front().time < curTick()) {
        DPRINTF(StreamProcessorArrayTick, "GPU finished a kernel id %d\n", finished_kernels.front().grid_uid);

        DPRINTF(StreamProcessorArray, "GPGPU-sim done! Activating original thread context at %llu.\n", getCurTick());
        streamManager->register_finished_kernel(finished_kernels.front().grid_uid);
        finished_kernels.pop();

        kernelTimes.push_back(curTick());
        if (dumpKernelStats) {
            PseudoInst::dumpresetstats(tc, 0, 0);
        }

        if (unblockNeeded && streamManager->empty() && finished_kernels.empty()) {
            DPRINTF(StreamProcessorArray, "Stream manager is empty, unblocking\n");
            tc->activate();
            unblockNeeded = false;
        }
    }

    // simulate a clock cycle on the GPU
    if( theGPU->active() ) {
        theGPU->cycle();
    } else {
        if(!finished_kernels.empty()) {
            schedule(gpuTickEvent, finished_kernels.front().time + 1);
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


void StreamProcessorArray::unblock()
{
    DPRINTF(StreamProcessorArray, "Unblocking for an event\n");
    assert(tc->status() == ThreadContext::Suspended);
    tc->activate();
}


void StreamProcessorArray::gpuRequestTick(float gpuTicks) {
    Tick gpuWakeupTick = (int)(gpuTicks*gpuTickConversion) + curTick();

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


void StreamProcessorArray::start(ThreadContext *_tc, gpgpu_sim *the_gpu, stream_manager *_stream_manager)
{
    tc = _tc;
    process = (LiveProcess*)tc->getProcessPtr();
    theGPU = the_gpu;
    streamManager = _stream_manager;

    vector<ShaderCore*>::iterator iter;
    for (iter=shaderCores.begin(); iter!=shaderCores.end(); ++iter) {
        (*iter)->initialize(tc);
    }

    copyEngine->initialize(tc, this);

    DPRINTF(StreamProcessorArray, "Starting this stream processor from tc\n");
}


bool StreamProcessorArray::setUnblock()
{
    if (!streamManager->empty()) {
        DPRINTF(StreamProcessorArray, "Suspend request: Need to activate CPU later\n");
        unblockNeeded = true;
        streamManager->print(stdout);
        return true;
    }
    else {
        DPRINTF(StreamProcessorArray, "Suspend request: Already done.\n");
        return false;
    }
}


void StreamProcessorArray::beginRunning(Tick launchTime)
{
    DPRINTF(StreamProcessorArray, "Beginning kernel execution at %llu\n", curTick());
    kernelTimes.push_back(curTick());
    if (dumpKernelStats) {
        PseudoInst::dumpresetstats(tc, 0, 0);
    }
    if (running) {
        panic("Should not already be running if we are starting\n");
    }
    running = true;

    Tick delay = 1;
    Tick curTime = curTick();
    if (curTime - launchTime < launchDelay*gpuTickConversion) {
        delay = (Tick)(launchDelay*gpuTickConversion) - (curTime - launchTime); //delay by whatever is left over
    }

    schedule(gpuTickEvent, curTick()+delay);
}


void StreamProcessorArray::writeFunctional(Addr addr, size_t length, const uint8_t* data)
{
    DPRINTF(StreamProcessorArrayAccess, "Writing to addr 0x%x\n", addr);
    tc->getMemProxy().writeBlob(addr, const_cast<uint8_t*>(data), length);
}

void StreamProcessorArray::readFunctional(Addr addr, size_t length, uint8_t* data)
{
    DPRINTF(StreamProcessorArrayAccess, "Reading from addr 0x%x\n", addr);
    tc->getMemProxy().readBlob(addr, data, length);
}

ShaderCore *StreamProcessorArray::getShaderCore(int coreId)
{
    assert(coreId < numShaderCores);
    return shaderCores[coreId];
}


StreamProcessorArray *StreamProcessorArrayParams::create() {
    return new StreamProcessorArray(this);
}

void StreamProcessorArray::gpuPrintStats(std::ostream& out) {
    // Print kernel statistics
    unsigned long long total_kernel_ticks = 0;
    unsigned long long last_kernel_time = 0;
    bool kernel_active = false;
    vector<unsigned long long>::iterator it;
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
    out << "\ntotal kernel time = " << total_kernel_ticks << "\n";

    if (clearTick) {
        out << "Stats cleared at tick " << clearTick << "\n";
    }
}

void StreamProcessorArray::memcpy(void *src, void *dst, size_t count, struct CUstream_st *stream) {
    copyEngine->memcpy((Addr)src, (Addr)dst, count, stream);
}

void StreamProcessorArray::memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to, struct CUstream_st *stream) {
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
        copyEngine->memcpy((Addr)src, (Addr)dst, count, stream);
    } else {
        copyEngine->memcpy((Addr)dst, (Addr)src, count, stream);
    }
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

void StreamProcessorArray::set_inst_base_vaddr(uint64_t addr)
{
    m_inst_base_vaddr = addr;
}

uint64_t StreamProcessorArray::get_inst_base_vaddr()
{
    return m_inst_base_vaddr;
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

