// This file created from cuda_runtime_api.h distributed with CUDA 1.1
// Changes Copyright 2009,  Tor M. Aamodt, Ali Bakhoda and George L. Yuan
// University of British Columbia

/*
 * cuda_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda,
 * George L. Yuan and the University of British Columbia, Vancouver,
 * BC V6T 1Z4, All Rights Reserved.
 *
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and
 * benchmarks/template/ are derived from the CUDA SDK available from
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from
 * src/intersim/ are derived from Booksim (a simulator provided with the
 * textbook "Principles and Practices of Interconnection Networks" available
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by
 * the corresponding legal terms and conditions set forth separately (original
 * copyright notices are left in files from these sources and where we have
 * modified a file our copyright notice appears before the original copyright
 * notice).
 *
 * Using this version of GPGPU-Sim requires a complete installation of CUDA
 * which is distributed seperately by NVIDIA under separate terms and
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.
 *
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 *
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung,
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia,
 * Vancouver, BC V6T 1Z4
 */

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

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

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <iostream>
#include <string>

#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h> // Apple's version of GLUT is here
#else
#include <GL/gl.h>
#endif
#endif

#include "../gpgpu-sim/cuda-sim/cuda-sim.h"
#include "../gpgpu-sim/cuda-sim/ptx_ir.h"
#include "../gpgpu-sim/cuda-sim/ptx_loader.h"
#include "../gpgpu-sim/cuda-sim/ptx_parser.h"
#include "../gpgpu-sim/gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/gpgpusim_entrypoint.h"
#include "../gpgpu-sim/stream_manager.h"
#include "../spa_obj/sp_array.hh"
#include "cpu/thread_context.hh"
#include "debug/GPUSyscalls.hh"
#include "gpu_syscalls.hh"

#define MAX_STRING_LEN 1000

typedef struct CUstream_st *cudaStream_t;

extern void synchronize();
extern void exit_simulation();

gpusyscall_t *decode_package(ThreadContext *tc, gpusyscall_t *call_params);
char *unpack(char *bytes, int &bytes_off, int *lengths, int &lengths_off);

static int load_static_globals(symbol_table *symtab, gpgpu_t *gpu);
static int load_constants(symbol_table *symtab, gpgpu_t *gpu);

static kernel_info_t *gpgpu_cuda_ptx_sim_init_grid(gpgpu_ptx_sim_arg_list_t args,
        struct dim3 gridDim,
        struct dim3 blockDim,
        struct function_info* context );

/*DEVICE_BUILTIN*/

#if !defined(__dv)
#if defined(__cplusplus)
#define __dv(v) \
        = v
#else /* __cplusplus */
#define __dv(v)
#endif /* __cplusplus */
#endif /* !__dv */

cudaError_t g_last_cudaError = cudaSuccess;

extern stream_manager *g_stream_manager;

void register_ptx_function( const char *name, function_info *impl )
{
   // no longer need this
}

#if defined __APPLE__
#   define __my_func__    __PRETTY_FUNCTION__
#else
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define __my_func__    __PRETTY_FUNCTION__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __my_func__    __func__
#  else
#   define __my_func__    ((__const char *) 0)
#  endif
# endif
#endif

struct _cuda_device_id {
    _cuda_device_id(gpgpu_sim* gpu) {m_id = 0; m_next = NULL; m_gpgpu=gpu;}
    struct _cuda_device_id *next() { return m_next; }
    unsigned num_shader() const { return m_gpgpu->get_config().num_shader(); }
    int num_devices() const {
        if( m_next == NULL ) return 1;
        else return 1 + m_next->num_devices();
    }
    struct _cuda_device_id *get_device( unsigned n )
    {
        assert( n < (unsigned)num_devices() );
        struct _cuda_device_id *p=this;
        for(unsigned i=0; i<n; i++)
            p = p->m_next;
        return p;
    }
    const struct cudaDeviceProp *get_prop() const
    {
        return m_gpgpu->get_prop();
    }
    unsigned get_id() const { return m_id; }

    gpgpu_sim *get_gpgpu() { return m_gpgpu; }
    private:
        unsigned m_id;
        class gpgpu_sim *m_gpgpu;
        struct _cuda_device_id *m_next;
};

class kernel_config {
    public:
        kernel_config( dim3 GridDim, dim3 BlockDim, size_t sharedMem, struct CUstream_st *stream )
        {
            m_GridDim=GridDim;
            m_BlockDim=BlockDim;
            m_sharedMem=sharedMem;
            m_stream = stream;
        }
        void set_arg( const void *arg, size_t size, size_t offset )
        {
            m_args.push_front( gpgpu_ptx_sim_arg(arg,size,offset) );
        }
        dim3 grid_dim() const { return m_GridDim; }
        dim3 block_dim() const { return m_BlockDim; }
        gpgpu_ptx_sim_arg_list_t get_args() { return m_args; }
        struct CUstream_st *get_stream() { return m_stream; }

    private:
        dim3 m_GridDim;
        dim3 m_BlockDim;
        size_t m_sharedMem;
        struct CUstream_st *m_stream;
        gpgpu_ptx_sim_arg_list_t m_args;
};


class _cuda_device_id *GPGPUSim_Init(ThreadContext *tc)
{
    static _cuda_device_id *the_device = NULL;
    if( !the_device ) {
        assert(tc != NULL);
        stream_manager *p_stream_manager;
        StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
        gpgpu_sim *the_gpu = gem5_ptx_sim_init_perf(&p_stream_manager, spa->getSharedMemDelay(), spa->getConfigPath());

        cudaDeviceProp *prop = (cudaDeviceProp *) calloc(sizeof(cudaDeviceProp),1);
        snprintf(prop->name,256,"GPGPU-Sim_v%s", g_gpgpusim_version_string );
        prop->major = 2;
        prop->minor = 0;
        prop->totalGlobalMem = 0x40000000 /* 1 GB */;
        prop->memPitch = 0;
        prop->maxThreadsPerBlock = 512;
        prop->maxThreadsDim[0] = 512;
        prop->maxThreadsDim[1] = 512;
        prop->maxThreadsDim[2] = 512;
        prop->maxGridSize[0] = 0x40000000;
        prop->maxGridSize[1] = 0x40000000;
        prop->maxGridSize[2] = 0x40000000;
        prop->totalConstMem = 0x40000000;
        prop->textureAlignment = 0;
        prop->sharedMemPerBlock = the_gpu->shared_mem_size();
        prop->regsPerBlock = the_gpu->num_registers_per_core();
        prop->warpSize = the_gpu->wrp_size();
        prop->clockRate = the_gpu->shader_clock();
#if (CUDART_VERSION >= 2010)
        prop->multiProcessorCount = the_gpu->get_config().num_shader();
#endif
        the_gpu->set_prop(prop);
        the_device = new _cuda_device_id(the_gpu);


        //put stuff that was in gpgpu_sim_thread_concurrent here
        the_gpu->init();
        the_gpu->setSPA(spa);
        spa->start(tc, the_gpu, p_stream_manager);
    }
    //start_sim_thread(1);
    return the_device;
}

extern "C" void ptxinfo_addinfo()
{
    if( !strcmp("__cuda_dummy_entry__",get_ptxinfo_kname()) ) {
      // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
        clear_ptxinfo();
        return;
    }
    GPGPUSim_Init(NULL);
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    print_ptxinfo();
    spa->add_ptxinfo( get_ptxinfo_kname(), get_ptxinfo_kinfo() );
    clear_ptxinfo();
}

void cuda_not_implemented( const char* func, unsigned line )
{
    fflush(stdout);
    fflush(stderr);
    printf("\n\nGPGPU-Sim PTX: Execution error: CUDA API function \"%s()\" has not been implemented yet.\n"
            "                 [$GPGPUSIM_ROOT/libcuda/%s around line %u]\n\n\n",
    func,__FILE__, line );
    fflush(stdout);
    abort();
}

typedef std::map<unsigned,CUevent_st*> event_tracker_t;

int CUevent_st::m_next_event_uid;
event_tracker_t g_timer_events;
int g_active_device = 0; //active gpu that runs the code
std::list<kernel_config> g_cuda_launch_stack;

GPUSyscallHelper::GPUSyscallHelper(ThreadContext *_tc, gpusyscall_t* _call_params)
    : tc(_tc), sim_params_ptr((Addr)_call_params), arg_lengths(NULL),
      args(NULL), total_bytes(0)
{
    decode_package();
}

void
GPUSyscallHelper::readBlob(Addr addr, uint8_t* p, int size, ThreadContext *tc)
{
    if (FullSystem) {
        tc->getVirtProxy().readBlob(addr, p, size);
    } else {
        tc->getMemProxy().readBlob(addr, p, size);
    }
}

void
GPUSyscallHelper::readString(Addr addr, uint8_t* p, int size, gpgpu_t* the_gpu, ThreadContext *tc)
{
    // Ensure that the memory buffer is cleared
    memset(p, 0, size);

    // For each line in the read, grab the system's memory and check for
    // null-terminating character
    bool null_not_found = true;
    Addr curr_addr;
    int read_size;
    unsigned block_size = the_gpu->gem5_spa->getRubySystem()->getBlockSizeBytes();
    int bytes_read = 0;
    for (; bytes_read < size && null_not_found; bytes_read += read_size) {
        curr_addr = addr + bytes_read;
        read_size = block_size;
        if (bytes_read == 0) read_size -= curr_addr % block_size;
        if (bytes_read + read_size >= size) read_size = size - bytes_read;
        readBlob(curr_addr, &p[bytes_read], read_size, tc);
        for (int index = 0; index < read_size; ++index) {
            if (p[index] == 0) null_not_found = false;
        }
    }

    if (null_not_found) panic("Didn't find end of string at address %x (%s)!", addr, (char*)p);
}

void
GPUSyscallHelper::writeBlob(Addr addr, uint8_t* p, int size, ThreadContext *tc)
{
    if (FullSystem) {
        tc->getVirtProxy().writeBlob(addr, p, size);
    } else {
        tc->getMemProxy().writeBlob(addr, p, size);
    }
}

void
GPUSyscallHelper::decode_package()
{
    assert(sim_params_ptr);

    readBlob(sim_params_ptr, (unsigned char*)&sim_params, sizeof(gpusyscall_t));

    arg_lengths = new int[sim_params.num_args];
    readBlob((Addr)sim_params.arg_lengths, (unsigned char*)arg_lengths, sim_params.num_args * sizeof(int));

    args = new unsigned char[sim_params.total_bytes];
    readBlob((Addr)sim_params.args, args, sim_params.total_bytes);
}

GPUSyscallHelper::~GPUSyscallHelper()
{
    if (arg_lengths) {
        delete[] arg_lengths;
    }
    if (args) {
        delete[] args;
    }
}

char*
GPUSyscallHelper::getParam(int index)
{
    int arg_index = 0;
    for (int i = 0; i < index; i++) {
        arg_index += arg_lengths[i];
    }
    return (char*)&args[arg_index];
}

void
GPUSyscallHelper::setReturn(unsigned char* retValue, size_t size)
{
    writeBlob((uint64_t)sim_params.ret, retValue, size);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void
cudaMalloc(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_devPtr = *((Addr*)helper.getParam(0));
    size_t sim_size = *((size_t*)helper.getParam(1));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMalloc(devPtr = %x, size = %d)\n", sim_devPtr, sim_size);

    GPGPUSim_Init(tc);

    g_last_cudaError = cudaSuccess;
    // Tell CUDA runtime to allocate memory
    cudaError_t to_return = cudaErrorApiFailureBase;
    helper.setReturn((uint8_t*)&to_return, sizeof(cudaError_t));
    return;

//    uint64_t i = 0;
//    uint64_t *ip = &i;
//    void **devPtr = (void**)&ip;
//
//    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
//    *devPtr = spa->getTheGPU()->gpu_malloc(sim_size);
//    helper.writeBlob(sim_devPtr, (uint8_t*)(devPtr), sizeof(void *));
//
//    if (*devPtr) {
//        g_last_cudaError = cudaSuccess;
//    } else {
//        g_last_cudaError = cudaErrorMemoryAllocation;
//    }
//    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

void
cudaMallocHost(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_ptr = *((Addr*)helper.getParam(0));
    size_t sim_size = *((size_t*)helper.getParam(1));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMallocHost(ptr = %x, size = %d)\n", sim_ptr, sim_size);

    GPGPUSim_Init(tc);

    g_last_cudaError = cudaSuccess;
    // Tell CUDA runtime to allocate memory
    cudaError_t to_return = cudaErrorApiFailureBase;
    helper.setReturn((uint8_t*)&to_return, sizeof(cudaError_t));
}

void
cudaRegisterDeviceMemory(ThreadContext *tc, gpusyscall_t *call_params)
{
    // This GPU syscall is used to initialize tracking of GPU memory if the
    // simulation requires access credentials between CPU and GPU memory (e.g.
    // if the address space is segmented into CPU and device memory, or if
    // the CPU allocates GPU memory which it should not access)
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_devicePtr = *((Addr*)helper.getParam(0));
    size_t sim_size = *((size_t*)helper.getParam(1));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaRegisterDeviceMemory(devicePtr = %x, size = %d)\n", sim_devicePtr, sim_size);

    // TODO:
    // Get the physical address of full memory allocation (i.e. all pages)
    //   Separate function:
    //      if (FullSystem) {
    //          Addr paddr = TheISA::vtophys(tc, vaddr);
    //      } else {
    //          Addr paddr;
    //          tc->getProcessPtr()->pTable->translate(vaddr, paddr);
    //      }
    // Build struct to handle devicePtr and size (inside StreamProcessorArray?)
}

void
cudaMallocPitch(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
// 	uint64_t arg2 = process->getSyscallArg(tc, index);
// 	uint64_t arg3 = process->getSyscallArg(tc, index);
//
// 	cuda_not_implemented(__my_func__,__LINE__);
//
// 	void **devPtr = (void **)(arg0);
// 	size_t *pitch = (size_t *)(arg1);
// 	size_t width = (size_t)(arg2);
// 	size_t height = (size_t)(arg3);
//
// 	GPGPUSIM_INIT
// 	unsigned malloc_width_inbytes = width;
// 	DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: cudaMallocPitch (width = %d)\n", malloc_width_inbytes);
//
// 	if(useM5Mem) {
// 		*devPtr = (void*)m5_spa->allocMemory(malloc_width_inbytes*height);
// 	} else {
// 		*devPtr = gpgpu_ptx_sim_malloc(malloc_width_inbytes*height);
// 	}
//
// 	pitch[0] = malloc_width_inbytes;
// 	if ( *devPtr  ) {
// 		return  cudaSuccess;
// 	} else {
// 		return g_last_cudaError = cudaErrorMemoryAllocation;
// 	}
}

//__host__ cudaError_t CUDARTAPI cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(1)) {
void
cudaMallocArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);


// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
// 	uint64_t arg2 = process->getSyscallArg(tc, index);
// 	uint64_t arg3 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	struct cudaArray **array = (struct cudaArray **)(arg0);
// 	const struct cudaChannelFormatDesc *desc = (const struct cudaChannelFormatDesc *)(arg1);
// 	size_t width = (size_t)(arg2);
// 	size_t height =  (size_t)(arg3);	//__dv(1)
//
// 	unsigned size = width * height * ((desc->x + desc->y + desc->z + desc->w)/8);
// 	GPGPUSIM_INIT
// 			(*array) = (struct cudaArray*) malloc(sizeof(struct cudaArray));
// 	(*array)->desc = *desc;
// 	(*array)->width = width;
// 	(*array)->height = height;
// 	(*array)->size = size;
// 	(*array)->dimensions = 2;
//
// 	if(useM5Mem) {
// 		((*array)->devPtr) =  (void*)m5_spa->allocMemory(size);
// 	} else {
// 		// NOTE: in cuda-sim.cc mallocarray is exactly the same as malloc
// 		((*array)->devPtr) = gpgpu_ptx_sim_mallocarray(size);
// 	}
//
// 	DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: cudaMallocArray: devPtr32 = %d\n", ((*array)->devPtr32));
// 	((*array)->devPtr32) = (int) (long long) ((*array)->devPtr);
// 	if ( ((*array)->devPtr) ) {
// 		return g_last_cudaError = cudaSuccess;
// 	} else {
// 		return g_last_cudaError = cudaErrorMemoryAllocation;
// 	}
}

void
cudaFree(ThreadContext *tc, gpusyscall_t *call_params) {
    // TODO...  manage g_global_mem space?
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaFree() - Faked\n");
    g_last_cudaError = cudaSuccess;
}

void
cudaFreeHost(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	void *ptr = (void*)(arg0);
// 	free (ptr);  // this will crash the system if called twice
//
// 	return g_last_cudaError = cudaSuccess;
}

//__host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array){
void
cudaFreeArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	void *devPtr = (void *)(arg0);
//
// 	//struct cudaArray *array = (struct cudaArray *)(arg0);
//
// // TODO...  manage g_global_mem space?
//
// 	// We con implement this in m5!
// 	if(useM5Mem) {
// 		m5_spa->freeMemory((Addr)devPtr);
// 	}
//
// 	return g_last_cudaError = cudaSuccess;
};


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void
cudaMemcpy(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);

    void* sim_dst = *((void**)helper.getParam(0));
    const void* sim_src = *((void**)helper.getParam(1));
    size_t sim_count = *((size_t*)helper.getParam(2));
    enum cudaMemcpyKind sim_kind = *((enum cudaMemcpyKind*)helper.getParam(3));

    GPGPUSim_Init(tc);
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    gpgpu_t *gpu = spa->getTheGPU();

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMemcpy(dst = %x, src = %x, count = %d, kind = %s)\n",
            sim_dst, sim_src, sim_count, cudaMemcpyKindStrings[sim_kind]);

    if (sim_count == 0) {
        g_last_cudaError = cudaSuccess;
        helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
        return;
    }

    if( sim_kind == cudaMemcpyHostToDevice )
        g_stream_manager->push( stream_operation(sim_src, (size_t)sim_dst, sim_count, 0) );
    else if( sim_kind == cudaMemcpyDeviceToHost )
        g_stream_manager->push( stream_operation((size_t)sim_src, sim_dst, sim_count, 0) );
    else if( sim_kind == cudaMemcpyDeviceToDevice )
        g_stream_manager->push( stream_operation((size_t)sim_src, (size_t)sim_dst, sim_count, 0) );
    else {
        panic("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
    }

    bool suspend = gpu->gem5_spa->setUnblock();
    assert(suspend);
    if (suspend) {
        tc->suspend();
    }

    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

//__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
void
cudaMemcpyToArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
// 	uint64_t arg2 = process->getSyscallArg(tc, index);
// 	uint64_t arg3 = process->getSyscallArg(tc, index);
// 	uint64_t arg4 = process->getSyscallArg(tc, index);
// 	uint64_t arg5 = process->getSyscallArg(tc, index);
//
//
// 	cuda_not_implemented(__my_func__,__LINE__);
//
// 	struct cudaArray *dst = (struct cudaArray *)(arg0);
// 	size_t wOffset = (size_t )(arg1);
// 	size_t hOffset = (size_t)(arg2);
// 	const void *src = (const void *)(arg3);
// 	size_t count = (size_t)(arg4);
// 	enum cudaMemcpyKind kind = (enum cudaMemcpyKind)(arg5);
//
// 	size_t size = count;
// 	DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: cudaMemcpyToArray\n");
// 	gpgpu_ptx_sim_init_memory();
// 	if( kind == cudaMemcpyHostToDevice )
// 		gpgpu_ptx_sim_memcpy_to_gpu( (size_t)(dst->devPtr), src, size);
// 	else if( kind == cudaMemcpyDeviceToHost )
// 		gpgpu_ptx_sim_memcpy_from_gpu( dst->devPtr, (size_t)src, size);
// 	else if( kind == cudaMemcpyDeviceToDevice )
// 		gpgpu_ptx_sim_memcpy_gpu_to_gpu( (size_t)(dst->devPtr), (size_t)src, size);
// 	else {
// 		printf("GPGPU-Sim PTX: cudaMemcpyToArray - ERROR : unsupported cudaMemcpyKind\n");
// 		abort();
// 	}
// 	dst->devPtr32 = (unsigned) (size_t)(dst->devPtr);
// 	return g_last_cudaError = cudaSuccess;
}


//__host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
void
cudaMemcpyFromArray(ThreadContext *tc, gpusyscall_t *call_params) {
        //int index = 1;
        //uint64_t arg0 = process->getSyscallArg(tc, index);
        //uint64_t arg1 = process->getSyscallArg(tc, index);
        //uint64_t arg2 = process->getSyscallArg(tc, index);
        //uint64_t arg3 = process->getSyscallArg(tc, index);
        //uint64_t arg4 = process->getSyscallArg(tc, index);
        //uint64_t arg5 = process->getSyscallArg(tc, index);

        //struct cudaArray *dst = (struct cudaArray *)(arg0);
        //size_t wOffset = (size_t )(arg1);
        //size_t hOffset = (size_t)(arg2);
        //const void *src = (const void *)(arg3);
        //size_t count = (size_t)(arg4);
        //enum cudaMemcpyKind kind = (enum cudaMemcpyKind)(arg5);

        cuda_not_implemented(__my_func__,__LINE__);
}


//__host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
void
cudaMemcpyArrayToArray(ThreadContext *tc, gpusyscall_t *call_params) {
        //int index = 1;
        //uint64_t arg0 = process->getSyscallArg(tc, index);
        //uint64_t arg1 = process->getSyscallArg(tc, index);
        //uint64_t arg2 = process->getSyscallArg(tc, index);
        //uint64_t arg3 = process->getSyscallArg(tc, index);
        //uint64_t arg4 = process->getSyscallArg(tc, index);
        //uint64_t arg5 = process->getSyscallArg(tc, index);

        //struct cudaArray *dst = (struct cudaArray *)(arg0);
        //size_t wOffset = (size_t )(arg1);
        //size_t hOffset = (size_t)(arg2);
        //const void *src = (const void *)(arg4);
        //size_t count = (size_t)(arg5);
        //enum cudaMemcpyKind kind = (enum cudaMemcpyKind)(arg6); //__dv(cudaMemcpyDeviceToDevice)

        cuda_not_implemented(__my_func__,__LINE__);
}


//__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
void
cudaMemcpy2D(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
//        int index = 1;
//        uint64_t arg0 = process->getSyscallArg(tc, index);
//        uint64_t arg1 = process->getSyscallArg(tc, index);
//        uint64_t arg2 = process->getSyscallArg(tc, index);
//        uint64_t arg3 = process->getSyscallArg(tc, index);
//        uint64_t arg4 = process->getSyscallArg(tc, index);
//        uint64_t arg5 = process->getSyscallArg(tc, index);
//        uint64_t arg6 = process->getSyscallArg(tc, index);
//
//
//        cuda_not_implemented(__my_func__,__LINE__);

// 	void *dst = (void *)(arg0);
// 	size_t dpitch = (size_t )(arg1);
// 	const void *src = (const void *)(arg2);
// 	size_t spitch = (size_t)(arg3);
// 	size_t width = (size_t)(arg4);
// 	size_t height = (size_t)(arg5);
// 	enum cudaMemcpyKind kind = (enum cudaMemcpyKind)(arg6);
//
// 	gpgpu_ptx_sim_init_memory();
// 	struct cudaArray *cuArray_ptr;
// 	size_t size = spitch*height;
// 	cuArray_ptr = (cudaArray*)dst;
// 	gpgpusim_ptx_assert( (dpitch==spitch), "different src and dst pitch not supported yet" );
// 	if( kind == cudaMemcpyHostToDevice )
// 		gpgpu_ptx_sim_memcpy_to_gpu( (size_t)dst, src, size );
// 	else if( kind == cudaMemcpyDeviceToHost )
// 		gpgpu_ptx_sim_memcpy_from_gpu( dst, (size_t)src, size );
// 	else if( kind == cudaMemcpyDeviceToDevice )
// 		gpgpu_ptx_sim_memcpy_gpu_to_gpu( (size_t)dst, (size_t)src, size);
// 	else {
// 		printf("GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
// 		abort();
// 	}
}


//__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
void
cudaMemcpy2DToArray(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);
    //        int index = 1;
//        uint64_t arg0 = process->getSyscallArg(tc, index);
//        uint64_t arg1 = process->getSyscallArg(tc, index);
//        uint64_t arg2 = process->getSyscallArg(tc, index);
//        uint64_t arg3 = process->getSyscallArg(tc, index);
//        uint64_t arg4 = process->getSyscallArg(tc, index);
//        uint64_t arg5 = process->getSyscallArg(tc, index);
//        uint64_t arg6 = process->getSyscallArg(tc, index);
//        uint64_t arg7 = process->getSyscallArg(tc, index);
//
//        cuda_not_implemented(__my_func__,__LINE__);

// 	struct cudaArray *dst = (struct cudaArray *)(arg0);
// 	size_t wOffset = (size_t )(arg1);
// 	size_t hOffset = (size_t)(arg2);
// 	const void *src = (const void *)(arg3);
// 	size_t spitch = (size_t)(arg4);
// 	size_t width = (size_t)(arg5);
// 	size_t height = (size_t)(arg6);
// 	enum cudaMemcpyKind kind = (enum cudaMemcpyKind)(arg7);
//
// 	size_t size = spitch*height;
// 	gpgpu_ptx_sim_init_memory();
// 	size_t channel_size = dst->desc.w+dst->desc.x+dst->desc.y+dst->desc.z;
// 	gpgpusim_ptx_assert( ((channel_size%8) == 0), "none byte multiple destination channel size not supported (sz=%u)", channel_size );
// 	unsigned elem_size = channel_size/8;
// 	gpgpusim_ptx_assert( (dst->dimensions==2), "copy to none 2D array not supported" );
// 	gpgpusim_ptx_assert( (wOffset==0), "non-zero wOffset not yet supported" );
// 	gpgpusim_ptx_assert( (hOffset==0), "non-zero hOffset not yet supported" );
// 	gpgpusim_ptx_assert( (dst->height == (int)height), "partial copy not supported" );
// 	gpgpusim_ptx_assert( (elem_size*dst->width == width), "partial copy not supported" );
// 	gpgpusim_ptx_assert( (spitch == width), "spitch != width not supported" );
// 	if( kind == cudaMemcpyHostToDevice )
// 		gpgpu_ptx_sim_memcpy_to_gpu( (size_t)(dst->devPtr), src, size);
// 	else if( kind == cudaMemcpyDeviceToHost )
// 		gpgpu_ptx_sim_memcpy_from_gpu( dst->devPtr, (size_t)src, size);
// 	else if( kind == cudaMemcpyDeviceToDevice )
// 		gpgpu_ptx_sim_memcpy_gpu_to_gpu( (size_t)dst->devPtr, (size_t)src, size);
// 	else {
// 		printf("GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
// 		abort();
// 	}
// 	dst->devPtr32 = (unsigned) (size_t)(dst->devPtr);
}


//__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
void
cudaMemcpy2DFromArray(ThreadContext *tc, gpusyscall_t *call_params) {
        cuda_not_implemented(__my_func__,__LINE__);

}


//__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
void
cudaMemcpy2DArrayToArray(ThreadContext *tc, gpusyscall_t *call_params) {
        cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaMemcpyToSymbol(ThreadContext *tc, gpusyscall_t *call_params) {
    GPUSyscallHelper helper(tc, call_params);

    const char* sim_symbol = *((const char**)helper.getParam(0));
    const void* sim_src = *((const void**)helper.getParam(1));
    size_t sim_count = *((size_t*)helper.getParam(2));
    size_t sim_offset = *((size_t*)helper.getParam(3));
    enum cudaMemcpyKind sim_kind = *((enum cudaMemcpyKind*)helper.getParam(4));

    GPGPUSim_Init(tc);
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();


    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaMemcpyToSymbol(symbol = %x, src = %x, count = %d, offset = %d, kind = %s)\n",
            sim_symbol, sim_src, sim_count, sim_offset, cudaMemcpyKindStrings[sim_kind]);

    // Get the data to be copied
    gpgpu_t *gpu = spa->getTheGPU();

    assert(sim_kind == cudaMemcpyHostToDevice);
    g_stream_manager->push( stream_operation(sim_src, sim_symbol, sim_count, sim_offset, NULL) );

    bool suspend = gpu->gem5_spa->setUnblock();
    assert(suspend);
    if (suspend) {
        tc->suspend();
    }

    g_last_cudaError = cudaSuccess;
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}


//__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) {
void
cudaMemcpyFromSymbol(ThreadContext *tc, gpusyscall_t *call_params) {
    cuda_not_implemented(__my_func__,__LINE__);


// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
// 	uint64_t arg2 = process->getSyscallArg(tc, index);
// 	uint64_t arg3 = process->getSyscallArg(tc, index);
// 	uint64_t arg4 = process->getSyscallArg(tc, index);
//
// 	cuda_not_implemented(__my_func__,__LINE__);
//
// 	void *dst = (void *)(arg0);
// 	const char *symbol = (const char *)(arg1);
// 	size_t count = (size_t )(arg2);
// 	size_t offset = (size_t)(arg3); //__dv(0)
// 	enum cudaMemcpyKind kind = (enum cudaMemcpyKind)(arg4);
//
// 	assert(kind == cudaMemcpyDeviceToHost);
// 	DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: cudaMemcpyFromSymbol: symbol = %p\n", symbol);
// 	gpgpu_ptx_sim_memcpy_symbol(symbol,dst,count,offset,0);
// 	return g_last_cudaError = cudaSuccess;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

//	__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
void
cudaMemcpyAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);


// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
// 	uint64_t arg2 = process->getSyscallArg(tc, index);
// 	uint64_t arg3 = process->getSyscallArg(tc, index);
// 	uint64_t arg4 = process->getSyscallArg(tc, index);
//
// 	cuda_not_implemented(__my_func__,__LINE__);
//
// 	void *dst = (void *)arg0;
// 	const void *src = (const void *)arg1;
// 	size_t count = (size_t)arg2;
// 	enum cudaMemcpyKind kind = (enum cudaMemcpyKind)arg3;
// 	cudaStream_t stream = (cudaStream_t)arg4;
//
// 	printf("GPGPU-Sim PTX: warning cudaMemcpyAsync is implemented as blocking in this version of GPGPU-Sim...\n");
// 	return cudaMemcpy(dst,src,count,kind);
}


//	__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
void
cudaMemcpyToArrayAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}


//	__host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
void
cudaMemcpyFromArrayAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}


//	__host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
void
cudaMemcpy2DAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

void
cudaMemcpy2DToArrayAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

void
cudaMemcpy2DFromArrayAsync(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void
cudaMemset(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall Stub: cudaMemcpy()\n");

    g_last_cudaError = cudaSuccess;
    // Tell CUDA runtime to use CPU memset by default
    cudaError_t to_return = cudaErrorApiFailureBase;
    helper.setReturn((uint8_t*)&to_return, sizeof(cudaError_t));
    return;
}

void
cudaMemset2D(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void
cudaGetSymbolAddress(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

void
cudaGetSymbolSize(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void
cudaGetDeviceCount(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    Addr sim_count = *((Addr*)helper.getParam(0));

    _cuda_device_id *dev = GPGPUSim_Init(tc);
    int count = dev->num_devices();
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaGetDeviceCount(count* = %x) = %d\n", sim_count, count);

    helper.writeBlob(sim_count, (uint8_t*)(&count), sizeof(int));
    g_last_cudaError = cudaSuccess;
}

void
cudaGetDeviceProperties(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_prop = *((Addr*)helper.getParam(0));
    int sim_device = *((int*)helper.getParam(1));
    _cuda_device_id *dev = GPGPUSim_Init(tc);
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaGetDeviceProperties(prop* = %x, device = %d)\n", sim_prop, sim_device);
    if (sim_device <= dev->num_devices())  {
        const struct cudaDeviceProp prop = *dev->get_prop();
        helper.writeBlob(sim_prop, (uint8_t*)(&prop), sizeof(struct cudaDeviceProp));
        g_last_cudaError = cudaSuccess;
    } else {
        g_last_cudaError = cudaErrorInvalidDevice;
    }
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

void
cudaChooseDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	int *device = (int *)arg0;
// 	const struct cudaDeviceProp *prop = (const struct cudaDeviceProp *)arg1;
//
// 	//goal: Choose the best matching device (just returns *device == 0 for now)
// 	int i;
// 	*device = -1; // intended to show a non-existing device
// 	GPGPUSIM_INIT
// 	for (i=0; i < MY_DEVICE_COUNT ; i++)  {
// 		if( *device == -1 ) {
// 			*device= i;  // default, pick the first device
// 		}
// 		if( prop->totalGlobalMem <=  gpgpu_cuda_devices[i]->totalGlobalMem &&
// 				prop->sharedMemPerBlock    <=  gpgpu_cuda_devices[i]->sharedMemPerBlock &&
// 				prop->regsPerBlock    <=  gpgpu_cuda_devices[i]->regsPerBlock &&
// 				prop->regsPerBlock    <=  gpgpu_cuda_devices[i]->regsPerBlock &&
// 				prop->maxThreadsPerBlock   <=  gpgpu_cuda_devices[i]->maxThreadsPerBlock  &&
// 				prop->totalConstMem   <=  gpgpu_cuda_devices[i]->totalConstMem )
// 		{
// 			// if/when we study heterogenous multicpu configurations
// 			*device= i;
// 			break;
// 		}
// 	}
// 	if ( *device !=-1 )
// 		return g_last_cudaError = cudaSuccess;
// 	else {
// 		printf("GPGPU-Sim PTX: Exeuction error: no suitable GPU devices found??? in a simulator??? (%s:%u in %s)\n",
// 						__FILE__,__LINE__,__my_func__);
// 		abort();
// 		return g_last_cudaError = cudaErrorInvalidConfiguration;
// 	}
}

void
cudaSetDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    int sim_device = *((int*)helper.getParam(0));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaSetDevice(device = %d)\n", sim_device);
    if (sim_device <= GPGPUSim_Init(tc)->num_devices()) {
        g_active_device = sim_device;
        g_last_cudaError = cudaSuccess;
    } else {
        g_last_cudaError = cudaErrorInvalidDevice;
    }
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

void
cudaGetDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_device = *((Addr*)helper.getParam(0));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaGetDevice(device = 0x%x)\n", sim_device);
    if (g_active_device <= GPGPUSim_Init(tc)->num_devices()) {
        helper.writeBlob(sim_device, (uint8_t*)&g_active_device, sizeof(int));
        g_last_cudaError = cudaSuccess;
    } else {
        g_last_cudaError = cudaErrorInvalidDevice;
    }
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX))
void
cudaBindTexture(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    cuda_not_implemented(__my_func__,__LINE__);
//    int index = 1;
//    uint64_t arg0 = process->getSyscallArg(tc, index);
//    uint64_t arg1 = process->getSyscallArg(tc, index);
//    uint64_t arg2 = process->getSyscallArg(tc, index);
//    uint64_t arg3 = process->getSyscallArg(tc, index);
//    uint64_t arg4 = process->getSyscallArg(tc, index);
//
//
//
//    size_t *offset = (size_t *)arg0;
//
//    uint8_t *buf = new uint8_t[sizeof(const struct textureReference)];
//    tc->getMemProxy().readBlob(arg1, buf, sizeof(const struct textureReference));
//    const struct textureReference *texref = (const struct textureReference *)buf;
//
//    const void *devPtr = (const void *)arg2;
//
//    uint8_t *buf2 = new uint8_t[sizeof(const struct cudaChannelFormatDesc)];
//    tc->getMemProxy().readBlob(arg3, buf2, sizeof(const struct cudaChannelFormatDesc));
//    const struct cudaChannelFormatDesc *desc = (const struct cudaChannelFormatDesc *)buf2;
//
//    size_t size = (size_t)arg4; //__dv(UINT_MAX)
//
//
//    CUctx_st *context = GPGPUSim_Context(process, tc);
//    gpgpu_t *gpu = context->get_device()->get_gpgpu();
//    printf("GPGPU-Sim PTX: in cudaBindTexture: sizeof(struct textureReference) = %zu\n", sizeof(struct textureReference));
//    struct cudaArray *array;
//    array = (struct cudaArray*) malloc(sizeof(struct cudaArray));
//    array->desc = *desc;
//    array->size = size;
//    array->width = size;
//    array->height = 1;
//    array->dimensions = 1;
//    array->devPtr = (void*)devPtr;
//    array->devPtr32 = (int)(long long)devPtr;
//    offset = 0;
//    printf("GPGPU-Sim PTX:   size = %zu\n", size);
//    printf("GPGPU-Sim PTX:   texref = %p, array = %p\n", texref, array);
//    printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
//    printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n", gpu->gpgpu_ptx_sim_findNamefromTexture((const struct textureReference *)arg1));
//    printf("GPGPU-Sim PTX:   ChannelFormatDesc: x=%d, y=%d, z=%d, w=%d\n", desc->x, desc->y, desc->z, desc->w);
//    printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
//    //gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
//    gpu->gpgpu_ptx_sim_bindTextureToArray((const struct textureReference *)arg1, array);
//    //devPtr = (void*)(long long)array->devPtr32;
//    printf("GPGPU-Sim PTX: devPtr = %p\n", devPtr);
//    return g_last_cudaError = cudaSuccess;

}

void
cudaBindTextureToArray(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

}

void
cudaUnbindTexture(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaGetTextureAlignmentOffset(ThreadContext *tc, gpusyscall_t *call_params)
{
     cuda_not_implemented(__my_func__,__LINE__);

}

void
cudaGetTextureReference(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

void
cudaGetChannelDesc(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

}

void
cudaCreateChannelDesc(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);


// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
// 	uint64_t arg2 = process->getSyscallArg(tc, index);
// 	uint64_t arg3 = process->getSyscallArg(tc, index);
// 	uint64_t arg4 = process->getSyscallArg(tc, index);
//
// 	int x = (int)arg0;
// 	int y = (int)arg1;
// 	int z = (int)arg2;
// 	int w = (int)arg3;
// 	enum cudaChannelFormatKind f = (enum cudaChannelFormatKind)arg4;
//
//
// 	cuda_not_implemented(__my_func__,__LINE__);
//

// 	struct cudaChannelFormatDesc dummy;
// 	dummy.x = x;
// 	dummy.y = y;
// 	dummy.z = z;
// 	dummy.w = w;
// 	dummy.f = f;
// 	return dummy;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

void
cudaGetLastError(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaGetLastError()\n");
    helper.setReturn((uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

void
cudaGetErrorString(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);


// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	cudaError_t error = (cudaError_t)arg0;
//
// 	if( g_last_cudaError == cudaSuccess )
// 		return (uint64_t)("no error");
// 	char buf[1024];
// 	snprintf(buf,1024,"<<GPGPU-Sim PTX: there was an error (code = %d)>>", (int)g_last_cudaError);
// 	return (uint64_t)strdup(buf); // NOTE Not sure if this will work, change syscall func
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

void
cudaConfigureCall(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    dim3 sim_gridDim = *((dim3*)helper.getParam(0));
    dim3 sim_blockDim = *((dim3*)helper.getParam(1));
    size_t sim_sharedMem = *((size_t*)helper.getParam(2));
//    cudaStream_t sim_stream = *((cudaStream_t*)helper.getParam(3));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaConfigureCall(gridDim = (%u,%u,%u), blockDim = (%u,%u,%u), sharedMem = %x, stream)\n", sim_gridDim.x, sim_gridDim.y, sim_gridDim.z, sim_blockDim.x, sim_blockDim.y, sim_blockDim.z, sim_sharedMem);

    struct CUstream_st *stream = NULL;
    assert(!(*((cudaStream_t*)helper.getParam(3))));

    g_cuda_launch_stack.push_back( kernel_config(sim_gridDim, sim_blockDim, sim_sharedMem, stream) );
    g_last_cudaError = cudaSuccess;
}

void
cudaSetupArgument(ThreadContext *tc, gpusyscall_t *call_params){
    GPUSyscallHelper helper(tc, call_params);

    Addr sim_arg = *((Addr*)helper.getParam(0));
    size_t sim_size = *((size_t*)helper.getParam(1));
    size_t sim_offset = *((size_t*)helper.getParam(2));
    const void* arg = new uint8_t[sim_size];
    helper.readBlob(sim_arg, (uint8_t*)arg, sim_size);
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaSetupArgument(arg = %x, size = %d, offset = %d)\n", sim_arg, sim_size, sim_offset);

    //actual function contents
    assert(!g_cuda_launch_stack.empty());
    kernel_config &config = g_cuda_launch_stack.back();
    config.set_arg(arg, sim_size, sim_offset);

// This code, copied from GPGPU-Sim, isn't even used there...?
//    struct gpgpu_ptx_sim_arg *param = (gpgpu_ptx_sim_arg*) calloc(1, sizeof(struct gpgpu_ptx_sim_arg));
//    param->m_start = arg;
//    param->m_nbytes = sim_size;
//    param->m_offset = sim_offset;
    g_last_cudaError = cudaSuccess;
}


void
cudaLaunch(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    const char* sim_hostFun = *((char**)helper.getParam(0));

    GPGPUSim_Init(tc);
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    char *mode = getenv("PTX_SIM_MODE_FUNC");
    if (mode)
        sscanf(mode,"%u", &g_ptx_sim_mode);
    assert(!g_cuda_launch_stack.empty());
    kernel_config config = g_cuda_launch_stack.back();
    struct CUstream_st *stream = config.get_stream();
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaLaunch(hostFun* = %x)\n", (void*)sim_hostFun);
    kernel_info_t *grid = gpgpu_cuda_ptx_sim_init_grid(config.get_args(), config.grid_dim(), config.block_dim(), spa->get_kernel(sim_hostFun));
    grid->set_inst_base_vaddr(spa->getInstBaseVaddr());
    std::string kname = grid->name();
    //dim3 gridDim = config.grid_dim();
    //dim3 blockDim = config.block_dim();
    stream_operation op(grid, g_ptx_sim_mode, stream);
    g_stream_manager->push(op);
    g_cuda_launch_stack.pop_back();
    g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream)
void
cudaStreamCreate(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

// __host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
void
cudaStreamDestroy(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

// __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
void
cudaStreamSynchronize(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

// __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream)
void
cudaStreamQuery(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event)
void
cudaEventCreate(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
//    int args_off = 0;
//    int arg_lengths_offset = 0;
//    cudaEvent_t *event = (cudaEvent_t *)unpack(call_params->args, args_off, call_params->arg_lengths, arg_lengths_offset);
//    CUevent_st *e = new CUevent_st(false);
//    DPRINTF(GPUSyscalls, "Event pointer %p\n", e);
//    g_timer_events[e->get_uid()] = e;
//#if CUDART_VERSION >= 3000
//    // NOTE: when this write happens the event given to the user is a pointer in
//    // simulator space, not user space. If the application were to try to dereference it
//    // would get a seg fault. All other uses of event when the user passes it in do not
//    // need to call read blob on it.
//    tc->getMemProxy().writeBlob((uint64_t)event, (uint8_t*)(&e), sizeof(cudaEvent_t));
//    //*event = e;
//#else
//    tc->getMemProxy().writeBlob((uint64_t)event, (uint8_t*)(e->get_uid()), sizeof(cudaEvent_t));
//    //*event = e->get_uid();
//#endif
}

CUevent_st *get_event(cudaEvent_t event)
{
    unsigned event_uid;
#if CUDART_VERSION >= 3000
   event_uid = event->get_uid();
#else
   event_uid = event;
#endif
   event_tracker_t::iterator e = g_timer_events.find(event_uid);
   if( e == g_timer_events.end() ) {
       return NULL;
   }
   return e->second;
}

void
cudaEventRecord(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
//    call_params = decode_package(tc, call_params);
//    int args_off = 0;
//    int arg_lengths_offset = 0;
//
//    cudaStream_t stream;
//
//    cudaEvent_t event = (cudaEvent_t)unpack(call_params->args, args_off, call_params->arg_lengths, arg_lengths_offset);
//    assert(event);
//    uint64_t arg1 = (uint64_t)unpack(call_params->args, args_off, call_params->arg_lengths, arg_lengths_offset);
//
//    CUevent_st *e = get_event(event);
//    if (!e) {
//        g_last_cudaError = cudaErrorUnknown;
//        tc->getMemProxy().writeBlob((Addr)call_params->ret, (uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
//        return;
//    }
//
//    struct CUstream_st *s;
//    if (arg1 != 0) {
//        tc->getMemProxy().readBlob((uint64_t)arg1, (uint8_t*)&stream, sizeof(cudaStream_t));
//        s = (struct CUstream_st *)stream;
//    } else {
//        s = NULL;
//    }
//    stream_operation op(e,s);
//    g_stream_manager->push(op);
}

void
cudaEventQuery(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
//    call_params = decode_package(tc, call_params);
//
//    int args_off = 0;
//    int arg_lengths_offset = 0;
//    cudaEvent_t event = (cudaEvent_t)unpack(call_params->args, args_off, call_params->arg_lengths, arg_lengths_offset);
//    CUevent_st *e = get_event(event);
//    if (e == NULL) {
//        g_last_cudaError = cudaErrorInvalidValue;
//    } else if( e->done() ) {
//        g_last_cudaError = cudaSuccess;
//    } else {
//        g_last_cudaError = cudaErrorNotReady;
//    }
//    tc->getMemProxy().writeBlob((Addr)call_params->ret, (uint8_t*)&g_last_cudaError, sizeof(cudaError_t));
}

void
cudaEventSynchronize(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

//    int index = 1;
//    uint64_t arg0 = process->getSyscallArg(tc, index);
//
//
//
//    cudaEvent_t event = (cudaEvent_t)arg0;
//
//    printf("GPGPU-Sim API: cudaEventSynchronize ** waiting for event\n");
//    fflush(stdout);
//    CUevent_st *e = get_event(event);
//    DPRINTF(GPUSyscalls, "Event pointer %p\n", e);
//    if( !e->done() ) {
//        DPRINTF(GPUSyscalls, "Blocking on event %d\n", e->get_uid());
//        e->set_needs_unblock(true);
//        tc->suspend();
//    }
//    printf("GPGPU-Sim API: cudaEventSynchronize ** event detected\n");
//    fflush(stdout);
//    return g_last_cudaError = cudaSuccess;
}

void
cudaEventDestroy(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

//    int index = 1;
//    uint64_t arg0 = process->getSyscallArg(tc, index);
//
//
//
//    cudaEvent_t event = (cudaEvent_t)arg0;
//
//    CUevent_st *e = get_event(event);
//    unsigned event_uid = e->get_uid();
//    event_tracker_t::iterator pe = g_timer_events.find(event_uid);
//    if( pe == g_timer_events.end() )
//        return g_last_cudaError = cudaErrorInvalidValue;
//    g_timer_events.erase(pe);
//    return g_last_cudaError = cudaSuccess;
}

void
cudaEventElapsedTime(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

//    int index = 1;
//    uint64_t arg0 = process->getSyscallArg(tc, index);
//    uint64_t arg1 = process->getSyscallArg(tc, index);
//    uint64_t arg2 = process->getSyscallArg(tc, index);
//
//
//
//    float ms;
//
//    cudaEvent_t start = (cudaEvent_t)arg1;
//    cudaEvent_t end = (cudaEvent_t)arg2;
//
//    CUevent_st *s = get_event(start);
//    CUevent_st *e = get_event(end);
//    if( s==NULL || e==NULL )
//        return g_last_cudaError = cudaErrorUnknown;
//    //elapsed_time = e->clock() - s->clock();  // NOTE: I don't think this is right
//
//    //*ms = 1000*elapsed_time;
//    unsigned long long elapsed_ticks = e->ticks() - s->ticks();
//    ms = (double)elapsed_ticks/1e9; // 1e9 ticks per ms
//    tc->getMemProxy().writeBlob(arg0, (uint8_t*)(&ms), sizeof(float));
//
//    return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

void
cudaThreadExit(ThreadContext *tc, gpusyscall_t *call_params)
{
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    bool suspend = spa->setUnblock();
    if (suspend) {
        tc->suspend();
        DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaThreadExit(), tc = %x\n", tc);
    }
    g_last_cudaError = cudaSuccess;
}

void
cudaThreadSynchronize(ThreadContext *tc, gpusyscall_t *call_params)
{
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    bool suspend = spa->setUnblock();
    if (suspend) {
        tc->suspend();
        DPRINTF(GPUSyscalls, "gem5 GPU Syscall: cudaThreadSynchronize(), tc = %x\n", tc);
    }
    g_last_cudaError = cudaSuccess;
}

void
__cudaSynchronizeThreads(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
deleteFatCudaBinary(__cudaFatCudaBinary* fat_cubin) {
    if (fat_cubin->ident) delete[] fat_cubin->ident;
    if (fat_cubin->ptx) {
        // @TODO: This might need to loop... consider splitting out the
        // CUDA binary read into a separate helper class that tracks the
        // number of ptx_entries that are read in
        if (fat_cubin->ptx->gpuProfileName) delete[] fat_cubin->ptx->gpuProfileName;
        if (fat_cubin->ptx->ptx) delete[] fat_cubin->ptx->ptx;
        delete[] fat_cubin->ptx;
    }
    delete fat_cubin;
}

symbol_table* registering_symtab = NULL;
unsigned registering_fat_cubin_handle = 0;
int registering_allocation_size = -1;

unsigned
get_global_and_constant_alloc_size(symbol_table* symtab)
{
    unsigned total_bytes = 0;
    symbol_table::iterator iter;
    for (iter = symtab->global_iterator_begin(); iter != symtab->global_iterator_end(); iter++) {
        symbol* global = *iter;
        total_bytes += global->get_size_in_bytes();
    }

    for (iter = symtab->const_iterator_begin(); iter != symtab->const_iterator_end(); iter++) {
        symbol* constant = *iter;
        total_bytes += constant->get_size_in_bytes();
    }

    return total_bytes;
}

void
finalize_global_and_constant_setup(Addr base_addr, symbol_table* symtab)
{
    Addr curr_addr = base_addr;
    Addr next_addr = 0;
    symbol_table::iterator iter;
    for (iter = symtab->global_iterator_begin(); iter != symtab->global_iterator_end(); iter++) {
        symbol* global = *iter;
        global->set_address(curr_addr);
        next_addr = curr_addr + global->get_size_in_bytes();
        if (next_addr - base_addr > registering_allocation_size) {
            panic("Didn't allocate enough global+const memory. Bailing!");
        } else {
            DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: Updated symbol \"%s\" to address range 0x%x to 0x%x\n", global->name(), curr_addr, next_addr-1);
        }
        curr_addr = next_addr;
    }

    for (iter = symtab->const_iterator_begin(); iter != symtab->const_iterator_end(); iter++) {
        symbol* constant = *iter;
        constant->set_address(curr_addr);
        next_addr = curr_addr + constant->get_size_in_bytes();
        if (next_addr - base_addr > registering_allocation_size) {
            panic("Didn't allocate enough global+const memory. Bailing!");
        } else {
            DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: Updated symbol \"%s\" to address range 0x%x to 0x%x\n", constant->name(), curr_addr, next_addr-1);
        }
        curr_addr = next_addr;
    }
}

void registerFatBinaryTop(Addr sim_fatCubin, size_t sim_binSize, ThreadContext *tc)
{
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    gpgpu_t *gpu = spa->getTheGPU();

    // Get primary arguments
    __cudaFatCudaBinary* fat_cubin = new __cudaFatCudaBinary;
    GPUSyscallHelper::readBlob(sim_fatCubin, (uint8_t*)fat_cubin, sizeof(struct __cudaFatCudaBinaryRec), tc);

    if (sim_binSize < 0) {
        panic("Used wrong __cudaRegisterFatBinary call!!! Did you run the sizeHack.py?");
    }

    // Read in the fat PTX entries
    uint8_t* ptx_entries = NULL;
    __cudaFatPtxEntry* ptx_entry_ptr;
    int ptx_count = 0;
    do {
        uint8_t* temp_ptx_entry_buf = new uint8_t[sizeof(__cudaFatPtxEntry) * (ptx_count + 1)];
        if (ptx_entries) {
            memcpy(temp_ptx_entry_buf, ptx_entries, sizeof(__cudaFatPtxEntry) * ptx_count);
        }
        GPUSyscallHelper::readBlob((Addr)(fat_cubin->ptx + ptx_count), temp_ptx_entry_buf + sizeof(__cudaFatPtxEntry) * ptx_count, sizeof(__cudaFatPtxEntry), tc);

        ptx_entry_ptr = (__cudaFatPtxEntry *)temp_ptx_entry_buf + ptx_count;
        if(ptx_entry_ptr->ptx != 0) {
            DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: Found instruction text segment: %p", ptx_entry_ptr->ptx);
            spa->setInstBaseVaddr((address_type)(Addr)ptx_entry_ptr->ptx);
            uint8_t* ptx_code = new uint8_t[sim_binSize];
            GPUSyscallHelper::readBlob((Addr)ptx_entry_ptr->ptx, ptx_code, sim_binSize, tc);
            uint8_t* gpu_profile = new uint8_t[MAX_STRING_LEN];
            GPUSyscallHelper::readString((Addr)ptx_entry_ptr->gpuProfileName, gpu_profile, MAX_STRING_LEN, gpu, tc);

            ptx_entry_ptr->ptx = (char*)ptx_code;
            ptx_entry_ptr->gpuProfileName = (char*)gpu_profile;
        }
        ptx_count++;
        if (ptx_entries) delete[] ptx_entries;
        ptx_entries = temp_ptx_entry_buf;
    } while(ptx_entry_ptr->gpuProfileName != 0);
    fat_cubin->ptx = (__cudaFatPtxEntry *)ptx_entries;

    // Read ident member
    Addr ident_addr = (Addr)fat_cubin->ident;
    fat_cubin->ident = new char[MAX_STRING_LEN];
    GPUSyscallHelper::readString(ident_addr, (uint8_t*)fat_cubin->ident, MAX_STRING_LEN, gpu, tc);


    static unsigned next_fat_bin_handle = 1;
    static unsigned source_num = 1;
    assert(registering_fat_cubin_handle == 0);
    registering_fat_cubin_handle = next_fat_bin_handle++;
    assert(fat_cubin->version >= 3);
    unsigned num_ptx_versions = 0;
    unsigned max_capability = 0;
    unsigned selected_capability = 0;
    bool found = false;
    unsigned forced_max_capability = gpu->get_config().get_forced_max_capability();
    while (fat_cubin->ptx[num_ptx_versions].gpuProfileName != NULL) {
        unsigned capability = 0;
        sscanf(fat_cubin->ptx[num_ptx_versions].gpuProfileName, "compute_%u", &capability);
        DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: __cudaRegisterFatBinary found PTX versions for '%s', capability = %s\n", fat_cubin->ident, fat_cubin->ptx[num_ptx_versions].gpuProfileName);
        if( forced_max_capability ) {
            if( capability > max_capability && capability <= forced_max_capability ) {
                found = true;
                max_capability = capability;
                selected_capability = num_ptx_versions;
            }
        } else {
            if( capability > max_capability ) {
                found = true;
                max_capability = capability;
                selected_capability = num_ptx_versions;
            }
        }
        num_ptx_versions++;
    }
    if (found) {
        DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: Loading PTX for %s, capability = %s\n",
                fat_cubin->ident, fat_cubin->ptx[selected_capability].gpuProfileName );
        const char *ptx = fat_cubin->ptx[selected_capability].ptx;
        if (gpu->get_config().convert_to_ptxplus()) {
            panic("GPGPU-Sim PTXPLUS: gem5 + GPGPU-Sim does not support PTXPLUS!");
        } else {
            assert(registering_symtab == NULL);
            registering_symtab = gpgpu_ptx_sim_load_ptx_from_string(ptx, source_num);
            spa->add_binary(registering_symtab, registering_fat_cubin_handle);
            gpgpu_ptxinfo_load_from_string(ptx, source_num);
        }
        source_num++;
        assert(registering_allocation_size == -1);
        registering_allocation_size = get_global_and_constant_alloc_size(registering_symtab);
    } else {
        panic("GPGPU-Sim PTX: warning -- did not find an appropriate PTX in cubin");
    }
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterFatBinary needs %d bytes allocated\n", registering_allocation_size);

    deleteFatCudaBinary(fat_cubin);
}

void
__cudaRegisterFatBinary(ThreadContext *tc, gpusyscall_t *call_params)
{
#if (CUDART_VERSION < 2010)
    printf("GPGPU-Sim PTX: ERROR ** this version of GPGPU-Sim requires CUDA 2.1 or higher\n");
    exit(1);
#endif

    GPUSyscallHelper helper(tc, call_params);

    // Get CUDA call simulated parameters
    Addr sim_fatCubin = *((Addr*)helper.getParam(0));
    int sim_binSize = *((int*)helper.getParam(1));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterFatBinary(fatCubin* = %x, binSize = %d)\n", sim_fatCubin, sim_binSize);
    GPGPUSim_Init(tc);
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    
    registerFatBinaryTop(sim_fatCubin, sim_binSize, tc);

    spa->saveFatBinaryInfoTop(registering_fat_cubin_handle, sim_fatCubin, sim_binSize);

    helper.setReturn((uint8_t*)&registering_allocation_size, sizeof(int));
}


unsigned int registerFatBinaryBottom(Addr sim_alloc_ptr)
{
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterFatBinaryFinalize(alloc_ptr* = 0x%x)\n", sim_alloc_ptr);

    assert(registering_symtab);
    assert(registering_fat_cubin_handle > 0);
    assert(registering_allocation_size >= 0);
    assert(sim_alloc_ptr || registering_allocation_size == 0);

    if (registering_allocation_size > 0) {
        finalize_global_and_constant_setup(sim_alloc_ptr, registering_symtab);
    }

    load_static_globals(registering_symtab, spa->getTheGPU());
    load_constants(registering_symtab, spa->getTheGPU());

    unsigned int handle = registering_fat_cubin_handle;

    registering_symtab = NULL;
    registering_fat_cubin_handle = 0;
    registering_allocation_size = -1;

    return handle;
}

void
__cudaRegisterFatBinaryFinalize(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);
    GPGPUSim_Init(tc);

    Addr sim_alloc_ptr = *((Addr*)helper.getParam(0));

    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();

    spa->saveFatBinaryInfoBottom(sim_alloc_ptr);

    unsigned int handle = registerFatBinaryBottom(sim_alloc_ptr);

    helper.setReturn((uint8_t*)&handle, sizeof(void**));
}

void
__cudaUnregisterFatBinary(ThreadContext *tc, gpusyscall_t *call_params)
{
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaUnregisterFatBinary() Faked\n");
}

void
__cudaRegisterFunction(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

    void **sim_fatCubinHandle = *((void***)helper.getParam(0));
    const char *sim_hostFun = *((const char**)helper.getParam(1));
    Addr sim_deviceFun = *((Addr*)helper.getParam(2));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterFunction(fatCubinHandle** = %x, hostFun* = %x, deviceFun* = %x)\n",
            sim_fatCubinHandle, (void*)sim_hostFun, (void*)sim_deviceFun);

    // Read device function name from simulated system memory
    char* device_fun = new char[MAX_STRING_LEN];
    GPGPUSim_Init(tc);
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    helper.readString(sim_deviceFun, (uint8_t*)device_fun, MAX_STRING_LEN, spa->getTheGPU());

    // Register function
    unsigned fat_cubin_handle = (unsigned)(unsigned long long)sim_fatCubinHandle;
    spa->register_function(fat_cubin_handle, sim_hostFun, device_fun);
    spa->saveFunctionNames(fat_cubin_handle, sim_hostFun, device_fun);
    delete[] device_fun;
}

void register_var(Addr sim_deviceAddress, const char* deviceName, int sim_size, int sim_constant, int sim_global, int sim_ext, Addr sim_hostVar) 
{

    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterVar(fatCubinHandle** = %x, hostVar* = 0x%x, deviceAddress* = 0x%x, deviceName* = %s, ext = %d, size = %d, constant = %d, global = %d)\n",
            /*sim_fatCubinHandle*/ 0, sim_hostVar, sim_deviceAddress,
            deviceName, sim_ext, sim_size, sim_constant, sim_global);

    if (sim_constant && !sim_global && !sim_ext) {
        gpgpu_ptx_sim_register_const_variable((void*)sim_hostVar, deviceName, sim_size);
    } else if (!sim_constant && !sim_global && !sim_ext) {
        gpgpu_ptx_sim_register_global_variable((void*)sim_hostVar, deviceName, sim_size);
    } else {
        panic("__cudaRegisterVar: Don't know how to register variable!");
    }
}

void __cudaRegisterVar(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

//    void** sim_fatCubinHandle = *((void***)helper.getParam(0));
    Addr sim_hostVar = *((Addr*)helper.getParam(1));
    Addr sim_deviceAddress = *((Addr*)helper.getParam(2));
    Addr sim_deviceName = *((Addr*)helper.getParam(3));
    int sim_ext = *((int*)helper.getParam(4));
    int sim_size = *((int*)helper.getParam(5));
    int sim_constant = *((int*)helper.getParam(6));
    int sim_global = *((int*)helper.getParam(7));

    const char* deviceName = new char[MAX_STRING_LEN];
    GPGPUSim_Init(tc);
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    helper.readString(sim_deviceName, (uint8_t*)deviceName, MAX_STRING_LEN, spa->getTheGPU());

    spa->saveVar(sim_deviceAddress, deviceName, sim_size, sim_constant, sim_global, sim_ext, sim_hostVar);
    
    register_var(sim_deviceAddress, deviceName, sim_size, sim_constant, sim_global, sim_ext, sim_hostVar);
}


//  void __cudaRegisterShared(
// 		 void **fatCubinHandle,
//  void **devicePtr
// 						  )
void
__cudaRegisterShared(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

}

void
__cudaRegisterSharedVar(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

}

void
__cudaRegisterTexture(ThreadContext *tc, gpusyscall_t *call_params)
{
    GPUSyscallHelper helper(tc, call_params);

//    void** sim_fatCubinHandle = *((void***)helper.getParam(0));
    const struct textureReference* sim_hostVar = *((const struct textureReference**)helper.getParam(1));
//    Addr sim_deviceAddress = *((Addr*)helper.getParam(2));
    Addr sim_deviceName = *((Addr*)helper.getParam(3));
    int sim_dim = *((int*)helper.getParam(4));
    int sim_norm = *((int*)helper.getParam(5));
    int sim_ext = *((int*)helper.getParam(6));
    DPRINTF(GPUSyscalls, "gem5 GPU Syscall: __cudaRegisterTexture(fatCubinHandle** = %x, hostVar* = %x, deviceAddress* = %x, deviceName* = %x, dim = %d, norm = %d, ext = %d)\n",
            /*sim_fatCubinHandle*/ 0, (void*)sim_hostVar, /*sim_deviceAddress*/ 0,
            sim_deviceName, sim_dim, sim_norm, sim_ext);

    const char* deviceName = new char[MAX_STRING_LEN];
    GPGPUSim_Init(tc);
    StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
    gpgpu_t *gpu = spa->getTheGPU();
    helper.readString(sim_deviceName, (uint8_t*)deviceName, MAX_STRING_LEN, gpu);

    gpu->gpgpu_ptx_sim_bindNameToTexture(deviceName, sim_hostVar);
    warn("__cudaRegisterTexture implementation is not complete!");
}

#ifndef OPENGL_SUPPORT
typedef unsigned long GLuint;
#endif

void
cudaGLRegisterBufferObject(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

}

struct glbmap_entry {
        GLuint m_bufferObj;
        void *m_devPtr;
        size_t m_size;
        struct glbmap_entry *m_next;
};
typedef struct glbmap_entry glbmap_entry_t;

glbmap_entry_t* g_glbmap = NULL;

void
cudaGLMapBufferObject(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
//    int args_off = 0;
//    int arg_lengths_offset = 0;
// 	uint64_t arg0 = (uint64_t)unpack(call_params->args, args_off, call_params->arg_lengths, arg_lengths_offset);
// 	uint64_t arg1 = (uint64_t)unpack(call_params->args, args_off, call_params->arg_lengths, arg_lengths_offset);
//
// 	void** devPtr = (void**)arg0;
// 	GLuint bufferObj = (GLuint)arg1;
//
// #ifdef OPENGL_SUPPORT
//    GLint buffer_size=0;
//    GPGPUSIM_INIT
//
// 		   glbmap_entry_t *p = g_glbmap;
//    while ( p && p->m_bufferObj != bufferObj )
// 	   p = p->m_next;
//    if ( p == NULL ) {
// 	   glBindBuffer(GL_ARRAY_BUFFER,bufferObj);
// 	   glGetBufferParameteriv(GL_ARRAY_BUFFER,GL_BUFFER_SIZE,&buffer_size);
// 	   assert( buffer_size != 0 );
// 	   *devPtr = gpgpu_ptx_sim_malloc(buffer_size);
//
//       // create entry and insert to front of list
// 	   glbmap_entry_t *n = (glbmap_entry_t *) calloc(1,sizeof(glbmap_entry_t));
// 	   n->m_next = g_glbmap;
// 	   g_glbmap = n;
//
//       // initialize entry
// 	   n->m_bufferObj = bufferObj;
// 	   n->m_devPtr = *devPtr;
// 	   n->m_size = buffer_size;
//
// 	   p = n;
//    } else {
// 	   buffer_size = p->m_size;
// 	   *devPtr = p->m_devPtr;
//    }
//
//    if ( *devPtr  ) {
// 	   char *data = (char *) calloc(p->m_size,1);
// 	   glGetBufferSubData(GL_ARRAY_BUFFER,0,buffer_size,data);
// 	   gpgpu_ptx_sim_memcpy_to_gpu( (size_t) *devPtr, data, buffer_size );
// 	   free(data);
// 	   printf("GPGPU-Sim PTX: cudaGLMapBufferObject %zu bytes starting at 0x%llx..\n", (size_t)buffer_size,
// 			  (unsigned long long) *devPtr);
// 	   return g_last_cudaError = cudaSuccess;
//    } else {
// 	   return g_last_cudaError = cudaErrorMemoryAllocation;
//    }
//
//    return g_last_cudaError = cudaSuccess;
// #else
//    fflush(stdout);
//    fflush(stderr);
//    printf("GPGPU-Sim PTX: GPGPU-Sim support for OpenGL integration disabled -- exiting\n");
//    fflush(stdout);
//    exit(50);
// #endif
}

//cudaError_t cudaGLUnmapBufferObject(GLuint bufferObj)
void
cudaGLUnmapBufferObject(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

}

//cudaError_t cudaGLUnregisterBufferObject(GLuint bufferObj)
void
cudaGLUnregisterBufferObject(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);

}

#if (CUDART_VERSION >= 2010)

void
cudaHostAlloc(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

void
cudaHostGetDevicePointer(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

void
cudaSetValidDevices(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);

}

void
cudaSetDeviceFlags(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaFuncGetAttributes(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaEventCreateWithFlags(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaDriverGetVersion(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaRuntimeGetVersion(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

#endif

void
cudaGLSetGLDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
cudaWGLGetDevice(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
__cudaMutexOperation(ThreadContext *tc, gpusyscall_t *call_params)
{
    cuda_not_implemented(__my_func__,__LINE__);
}

void
__cudaTextureFetch(ThreadContext *tc, gpusyscall_t *call_params)
{
        cuda_not_implemented(__my_func__,__LINE__);
}

namespace cuda_math {
    uint64_t __cudaMutexOperation(ThreadContext *tc, gpusyscall_t *call_params)
    {
        cuda_not_implemented(__my_func__,__LINE__);
        return 0;
    }

    uint64_t __cudaTextureFetch(ThreadContext *tc, gpusyscall_t *call_params)
    {
        cuda_not_implemented(__my_func__,__LINE__);
        return 0;
    }

    uint64_t __cudaSynchronizeThreads(ThreadContext *tc, gpusyscall_t *call_params)
    {
        //TODO This function should syncronize if we support Asyn kernel calls
        return g_last_cudaError = cudaSuccess;
    }

    //so that m5.debug will compile
    void  __cudaTextureFetch(const void *tex, void *index, int integer, void *val){ assert(0); }
    void __cudaMutexOperation(int lock){ assert(0); }
}

//gpusyscall_t *decode_package(ThreadContext *tc, gpusyscall_t *call_params)
//{
//    gpusyscall_t params;
//
//    tc->getMemProxy().readBlob((Addr)call_params, (unsigned char*)&params, sizeof(gpusyscall_t));
//
//    uint8_t *buf = new uint8_t[params.num_args * sizeof(int)];
//    tc->getMemProxy().readBlob((Addr)params.arg_lengths, buf, params.num_args * sizeof(int));
//    params.arg_lengths = (int*)buf;
//
//    buf = new uint8_t[params.total_bytes];
//    tc->getMemProxy().readBlob((Addr)params.args, buf, params.total_bytes);
//    params.args = (char*)buf;
//
//    // @TODO: This will leak memory...
//    call_params = new gpusyscall_t;
//    call_params->arg_lengths = params.arg_lengths;
//    call_params->args = params.args;
//    call_params->num_args = params.num_args;
//    call_params->total_bytes = params.total_bytes;
//    call_params->ret = params.ret;
//    return call_params;
//}

char *unpack(char *bytes, int &bytes_off, int *lengths, int &lengths_off)
{
    int arg_size = *(lengths + lengths_off);
    char *arg = new char[arg_size];
    for (int i = 0; i < arg_size; i++) {
            arg[i] = bytes[i + bytes_off];
    }

    bytes_off += arg_size;
    lengths_off += 1;

    return arg;
}

////////

extern "C" int ptx_parse();
extern "C" int ptx__scan_string(const char*);
extern "C" FILE *ptx_in;

extern "C" const char *g_ptxinfo_filename;
extern "C" int ptxinfo_parse();
extern "C" int ptxinfo_debug;
extern "C" FILE *ptxinfo_in;

/// static functions

static int load_static_globals(symbol_table *symtab, gpgpu_t *gpu)
{
    DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: loading globals with explicit initializers\n");
    int ng_bytes = 0;
    symbol_table::iterator g = symtab->global_iterator_begin();

    for (; g != symtab->global_iterator_end(); g++) {
        symbol *global = *g;
        if (global->has_initializer()) {
            DPRINTF(GPUSyscalls, "GPGPU-Sim PTX:     initializing '%s'\n", global->name().c_str());
            unsigned addr = global->get_address();
            const type_info *type = global->type();
            type_info_key ti = type->get_key();
            size_t size;
            int t;
            ti.type_decode(size, t);
            int nbytes = size/8;
            int offset = 0;
            std::list<operand_info> init_list = global->get_initializer();
            for (std::list<operand_info>::iterator i = init_list.begin(); i != init_list.end(); i++) {
                operand_info op = *i;
                ptx_reg_t value = op.get_literal_value();
                panic("Global statics load untested!!");
                gpu->gem5_spa->writeFunctional(addr+offset, nbytes, (uint8_t*)&value);
                offset += nbytes;
                ng_bytes += nbytes;
            }
            DPRINTF(GPUSyscalls, " wrote %u bytes to \'%s\'\n", offset, global->name());
        }
    }
    DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: finished loading globals (%u bytes total).\n", ng_bytes);
    return ng_bytes;
}

static int load_constants(symbol_table *symtab, gpgpu_t *gpu)
{
   DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: loading constants with explicit initializers\n");
   int nc_bytes = 0;
   symbol_table::iterator g = symtab->const_iterator_begin();

   for (; g != symtab->const_iterator_end(); g++) {
      symbol *constant = *g;
      if (constant->is_const() && constant->has_initializer()) {

         // get the constant element data size
         int basic_type;
         size_t num_bits;
         constant->type()->get_key().type_decode(num_bits, basic_type);

         std::list<operand_info> init_list = constant->get_initializer();
         int nbytes_written = 0;
         for ( std::list<operand_info>::iterator i=init_list.begin(); i!=init_list.end(); i++ ) {
            operand_info op = *i;
            ptx_reg_t value = op.get_literal_value();
            int nbytes = num_bits/8;
            switch (op.get_type()) {
            case int_t: assert(nbytes >= 1); break;
            case float_op_t: assert(nbytes == 4); break;
            case double_op_t: assert(nbytes >= 4); break; // account for double DEMOTING
            default:
               panic("Op type not recognized in load_constants"); break;
            }
            unsigned addr = constant->get_address() + nbytes_written;

            gpu->gem5_spa->writeFunctional(addr, nbytes, (uint8_t*)&value);
            DPRINTF(GPUSyscalls, " wrote %u bytes to \'%s\'\n", nbytes, constant->name());
            nc_bytes += nbytes;
            nbytes_written += nbytes;
         }
      }
   }
   DPRINTF(GPUSyscalls, "GPGPU-Sim PTX: finished loading constants (%u bytes total).\n", nc_bytes);
   return nc_bytes;
}

kernel_info_t *gpgpu_cuda_ptx_sim_init_grid(gpgpu_ptx_sim_arg_list_t args,
                                            struct dim3 gridDim,
                                            struct dim3 blockDim,
                                            function_info* entry )
{
   kernel_info_t *result = new kernel_info_t(gridDim,blockDim,entry);
   if (entry == NULL) {
       panic("GPGPU-Sim PTX: ERROR launching kernel -- no PTX implementation found");
   }
   unsigned argcount=args.size();
   unsigned argn=1;
   for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end(); a++) {
      entry->add_param_data(argcount-argn, &(*a));
      argn++;
   }

   entry->finalize(result->get_param_memory());
   g_ptx_kernel_count++;

   return result;
}
