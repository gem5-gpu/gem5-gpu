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
#include "cpu/thread_context.hh"

//<marc>
//#include "gpgpu-sim/src/cuda-sim/ptx_ir.h"

#include "../spa_obj/sp_array.hh"
#include "debug/GPGPUSyscalls.hh"
#include "g3_syscalls.hh"

//<marc> maximum length of a string passed to the api
#define MAX_STRING_LEN 1000

//typedef int cudaStream_t;
typedef struct CUstream_st *cudaStream_t;

/*******************************
       CUDA API STUFF
********************************/
enum cudaError
{
    cudaSuccess                           =      0,   ///< No errors
    cudaErrorMissingConfiguration         =      1,   ///< Missing configuration error
    cudaErrorMemoryAllocation             =      2,   ///< Memory allocation error
    cudaErrorInitializationError          =      3,   ///< Initialization error
    cudaErrorLaunchFailure                =      4,   ///< Launch failure
    cudaErrorPriorLaunchFailure           =      5,   ///< Prior launch failure
    cudaErrorLaunchTimeout                =      6,   ///< Launch timeout error
    cudaErrorLaunchOutOfResources         =      7,   ///< Launch out of resources error
    cudaErrorInvalidDeviceFunction        =      8,   ///< Invalid device function
    cudaErrorInvalidConfiguration         =      9,   ///< Invalid configuration
    cudaErrorInvalidDevice                =     10,   ///< Invalid device
    cudaErrorInvalidValue                 =     11,   ///< Invalid value
    cudaErrorInvalidPitchValue            =     12,   ///< Invalid pitch value
    cudaErrorInvalidSymbol                =     13,   ///< Invalid symbol
    cudaErrorMapBufferObjectFailed        =     14,   ///< Map buffer object failed
    cudaErrorUnmapBufferObjectFailed      =     15,   ///< Unmap buffer object failed
    cudaErrorInvalidHostPointer           =     16,   ///< Invalid host pointer
    cudaErrorInvalidDevicePointer         =     17,   ///< Invalid device pointer
    cudaErrorInvalidTexture               =     18,   ///< Invalid texture
    cudaErrorInvalidTextureBinding        =     19,   ///< Invalid texture binding
    cudaErrorInvalidChannelDescriptor     =     20,   ///< Invalid channel descriptor
    cudaErrorInvalidMemcpyDirection       =     21,   ///< Invalid memcpy direction
    cudaErrorAddressOfConstant            =     22,   ///< Address of constant error
                                                    ///< \deprecated
                                                    ///< This error return is deprecated as of
                                                    ///< Cuda 3.1. Variables in constant memory
                                                    ///< may now have their address taken by the
                                                    ///< runtime via ::cudaGetSymbolAddress().
    cudaErrorTextureFetchFailed           =     23,   ///< Texture fetch failed
    cudaErrorTextureNotBound              =     24,   ///< Texture not bound error
    cudaErrorSynchronizationError         =     25,   ///< Synchronization error
    cudaErrorInvalidFilterSetting         =     26,   ///< Invalid filter setting
    cudaErrorInvalidNormSetting           =     27,   ///< Invalid norm setting
    cudaErrorMixedDeviceExecution         =     28,   ///< Mixed device execution
    cudaErrorCudartUnloading              =     29,   ///< CUDA runtime unloading
    cudaErrorUnknown                      =     30,   ///< Unknown error condition
    cudaErrorNotYetImplemented            =     31,   ///< Function not yet implemented
    cudaErrorMemoryValueTooLarge          =     32,   ///< Memory value too large
    cudaErrorInvalidResourceHandle        =     33,   ///< Invalid resource handle
    cudaErrorNotReady                     =     34,   ///< Not ready error
    cudaErrorInsufficientDriver           =     35,   ///< CUDA runtime is newer than driver
    cudaErrorSetOnActiveProcess           =     36,   ///< Set on active process error
    cudaErrorInvalidSurface               =     37,   ///< Invalid surface
    cudaErrorNoDevice                     =     38,   ///< No Cuda-capable devices detected
    cudaErrorECCUncorrectable             =     39,   ///< Uncorrectable ECC error detected
    cudaErrorSharedObjectSymbolNotFound   =     40,   ///< Link to a shared object failed to resolve
    cudaErrorSharedObjectInitFailed       =     41,   ///< Shared object initialization failed
    cudaErrorUnsupportedLimit             =     42,   ///< ::cudaLimit not supported by device
    cudaErrorDuplicateVariableName        =     43,   ///< Duplicate global variable lookup by string name
    cudaErrorDuplicateTextureName         =     44,   ///< Duplicate texture lookup by string name
    cudaErrorDuplicateSurfaceName         =     45,   ///< Duplicate surface lookup by string name
    cudaErrorDevicesUnavailable           =     46,   ///< All Cuda-capable devices are busy (see ::cudaComputeMode) or unavailable
    cudaErrorStartupFailure               =   0x7f,   ///< Startup failure
    cudaErrorApiFailureBase               =  10000    ///< API failure base
};
typedef enum cudaError cudaError_t;
struct cudaDeviceProp
{
    char   name[256];                 ///< ASCII string identifying device
    size_t totalGlobalMem;            ///< Global memory available on device in bytes
    size_t sharedMemPerBlock;         ///< Shared memory available per block in bytes
    int    regsPerBlock;              ///< 32-bit registers available per block
    int    warpSize;                  ///< Warp size in threads
    size_t memPitch;                  ///< Maximum pitch in bytes allowed by memory copies
    int    maxThreadsPerBlock;        ///< Maximum number of threads per block
    int    maxThreadsDim[3];          ///< Maximum size of each dimension of a block
    int    maxGridSize[3];            ///< Maximum size of each dimension of a grid
    int    clockRate;                 ///< Clock frequency in kilohertz
    size_t totalConstMem;             ///< Constant memory available on device in bytes
    int    major;                     ///< Major compute capability
    int    minor;                     ///< Minor compute capability
    size_t textureAlignment;          ///< Alignment requirement for textures
    int    deviceOverlap;             ///< Device can concurrently copy memory and execute a kernel
    int    multiProcessorCount;       ///< Number of multiprocessors on device
    int    kernelExecTimeoutEnabled;  ///< Specified whether there is a run time limit on kernels
    int    integrated;                ///< Device is integrated as opposed to discrete
    int    canMapHostMemory;          ///< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
    int    computeMode;               ///< Compute mode (See ::cudaComputeMode)
    int    maxTexture1D;              ///< Maximum 1D texture size
    int    maxTexture2D[2];           ///< Maximum 2D texture dimensions
    int    maxTexture3D[3];           ///< Maximum 3D texture dimensions
    int    maxTexture2DArray[3];      ///< Maximum 2D texture array dimensions
    size_t surfaceAlignment;          ///< Alignment requirements for surfaces
    int    concurrentKernels;         ///< Device can possibly execute multiple kernels concurrently
    int    ECCEnabled;                ///< Device has ECC support enabled
    int    pciBusID;                  ///< PCI bus ID of the device
    int    pciDeviceID;               ///< PCI device ID of the device
    int    __cudaReserved[22];
};
enum cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      ///< Host   -> Host
    cudaMemcpyHostToDevice        =   1,      ///< Host   -> Device
    cudaMemcpyDeviceToHost        =   2,      ///< Device -> Host
    cudaMemcpyDeviceToDevice      =   3       ///< Device -> Device
};

typedef struct {
    char* gpuProfileName;
    char* cubin;
} __cudaFatCubinEntry;
typedef struct {
    char* gpuProfileName;
    char* ptx;
} __cudaFatPtxEntry;
typedef struct __cudaFatDebugEntryRec {
    char* gpuProfileName;
    char* debug;
    struct __cudaFatDebugEntryRec *next;
    unsigned int size;
} __cudaFatDebugEntry;

typedef struct __cudaFatElfEntryRec {
    char* gpuProfileName;
    char* elf;
    struct __cudaFatElfEntryRec *next;
    unsigned int size;
} __cudaFatElfEntry;

// typedef enum {
//       __cudaFatDontSearchFlag = (1 << 0),
//       __cudaFatDontCacheFlag = (1 << 1),
//       __cudaFatSassDebugFlag = (1 << 2)
// } __cudaFatCudaBinaryFlag;
typedef struct {
    char* name;
} __cudaFatSymbol;
typedef struct __cudaFatCudaBinaryRec {
    unsigned long magic;
    unsigned long version;
    unsigned long gpuInfoVersion;
    char* key;
    char* ident;
    char* usageMode;
    __cudaFatPtxEntry *ptx;
    __cudaFatCubinEntry *cubin;
    __cudaFatDebugEntry *debug;
    void* debugInfo;
    unsigned int flags;
    __cudaFatSymbol *exported;
    __cudaFatSymbol *imported;
    struct __cudaFatCudaBinaryRec *dependends;
    unsigned int characteristic;
    __cudaFatElfEntry *elf;
} __cudaFatCudaBinary;

/*DEVICE_BUILTIN*/
struct uint3
{
  unsigned int x, y, z;
};

typedef struct CUevent_st *cudaEvent_t;
/*******************************
       CUDA API STUFF END
********************************/

//using namespace std;

extern void synchronize();
extern void exit_simulation();

void decode_package(LiveProcess *process, ThreadContext *tc, int **arg_sizes, char **args);
char *unpack(char *bytes, int &bytes_off, int *lengths, int &lengths_off);

static int load_static_globals( symbol_table *symtab, unsigned min_gaddr, unsigned max_gaddr, gpgpu_t *gpu );
static int load_constants( symbol_table *symtab, addr_t min_gaddr, gpgpu_t *gpu );

static kernel_info_t *gpgpu_cuda_ptx_sim_init_grid( const char *kernel_key,
        gpgpu_ptx_sim_arg_list_t args,
        struct dim3 gridDim,
        struct dim3 blockDim,
        struct CUctx_st* context );

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

struct CUctx_st {
    CUctx_st( _cuda_device_id *gpu ) { m_gpu = gpu; }

    _cuda_device_id *get_device() { return m_gpu; }

    void add_binary( symbol_table *symtab, unsigned fat_cubin_handle )
    {
        m_code[fat_cubin_handle] = symtab;
        m_last_fat_cubin_handle = fat_cubin_handle;
    }

    void add_ptxinfo( const char *deviceFun, const struct gpgpu_ptx_sim_kernel_info &info )
    {
        symbol *s = m_code[m_last_fat_cubin_handle]->lookup(deviceFun);
        assert( s != NULL );
        function_info *f = s->get_pc();
        assert( f != NULL );
        f->set_kernel_info(info);
    }

    void register_function( unsigned fat_cubin_handle, const char *hostFun, const char *deviceFun )
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

    function_info *get_kernel(const char *hostFun)
    {
        std::map<const void*,function_info*>::iterator i=m_kernel_lookup.find(hostFun);
        assert( i != m_kernel_lookup.end() );
        return i->second;
    }

    private:
        _cuda_device_id *m_gpu; // selected gpu
        std::map<unsigned,symbol_table*> m_code; // fat binary handle => global symbol table
        unsigned m_last_fat_cubin_handle;
        std::map<const void*,function_info*> m_kernel_lookup; // unique id (CUDA app function address) => kernel entry point
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


class _cuda_device_id *GPGPUSim_Init(LiveProcess *process, ThreadContext *tc)
{
    static _cuda_device_id *the_device = NULL;
    if( !the_device ) {
        System *sys = process->system;

        stream_manager *p_stream_manager;
        StreamProcessorArray *spa = StreamProcessorArray::getStreamProcessorArray();
        gpgpu_sim *the_gpu = gem5_ptx_sim_init_perf(&p_stream_manager, spa->getUseGem5Mem(), spa->getSharedMemDelay());

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
        spa->start(process, tc, the_gpu, p_stream_manager);
    }
    //start_sim_thread(1);
    return the_device;
}

static CUctx_st* GPGPUSim_Context(LiveProcess *process, ThreadContext *tc)
{
    static CUctx_st *the_context = NULL;
    if( the_context == NULL ) {
      assert(process!=NULL && tc!=NULL);
      _cuda_device_id *the_gpu = GPGPUSim_Init(process, tc);
      the_context = new CUctx_st(the_gpu);
    }
    return the_context;
}

extern "C" void ptxinfo_addinfo()
{
    if( !strcmp("__cuda_dummy_entry__",get_ptxinfo_kname()) ) {
      // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
        clear_ptxinfo();
        return;
    }
    CUctx_st *context = GPGPUSim_Context(NULL, NULL);
    print_ptxinfo();
    context->add_ptxinfo( get_ptxinfo_kname(), get_ptxinfo_kinfo() );
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


#define gpgpusim_ptx_error(msg, ...) gpgpusim_ptx_error_impl(__func__, __FILE__,__LINE__, msg, ##__VA_ARGS__)
#define gpgpusim_ptx_assert(cond,msg, ...) gpgpusim_ptx_assert_impl((cond),__func__, __FILE__,__LINE__, msg, ##__VA_ARGS__)

void gpgpusim_ptx_error_impl( const char *func, const char *file, unsigned line, const char *msg, ... )
{
    va_list ap;
    char buf[1024];
    va_start(ap,msg);
    vsnprintf(buf,1024,msg,ap);
    va_end(ap);

    printf("GPGPU-Sim CUDA API: %s\n", buf);
    printf("                    [%s:%u : %s]\n", file, line, func );
    abort();
}

void gpgpusim_ptx_assert_impl( int test_value, const char *func, const char *file, unsigned line, const char *msg, ... )
{
    va_list ap;
    char buf[1024];
    va_start(ap,msg);
    vsnprintf(buf,1024,msg,ap);
    va_end(ap);

    if ( test_value == 0 )
        gpgpusim_ptx_error_impl(func, file, line, msg);
}


typedef std::map<unsigned,CUevent_st*> event_tracker_t;

int CUevent_st::m_next_event_uid;
event_tracker_t g_timer_events;
int g_active_device = 0; //active gpu that runs the code
std::list<kernel_config> g_cuda_launch_stack;

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

//__host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
uint64_t cudaMalloc(LiveProcess *process, ThreadContext *tc)
{
    CUctx_st* context = GPGPUSim_Context(process, tc);

    if(context->get_device()->get_gpgpu()->useGem5Mem) {
        g_last_cudaError = cudaSuccess;
        return cudaErrorApiFailureBase; //use this error flag to communicate to api that it should allocate memory on ruby system
    }

    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);
    uint64_t arg1 = process->getSyscallArg(tc, index);



    uint64_t i = 0;
    uint64_t *ip = &i;
    //void **devPtr = (void**)&i;
    void **devPtr = (void**)&ip;

    //void **devPtr = (void**)(arg0);
    size_t size = (size_t)(arg1);

    *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
    tc->getMemProxy().writeBlob(arg0, (uint8_t*)(devPtr), sizeof(void *));

    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",size, (unsigned long long) *devPtr);
    if ( *devPtr  ) {
        return g_last_cudaError = cudaSuccess;
    } else {
        return g_last_cudaError = cudaErrorMemoryAllocation;
    }
}

//__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size){
uint64_t cudaMallocHost(LiveProcess *process, ThreadContext *tc) {
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
//
// 	cuda_not_implemented(__my_func__,__LINE__);
//
// 	void **ptr = (void**)(arg0);
// 	size_t size = (size_t)(arg1);
//
// 	GPGPUSIM_INIT
// 			*ptr = malloc(size);
// 	if ( *ptr  ) {
// 		return  cudaSuccess;
// 	} else {
// 		return g_last_cudaError = cudaErrorMemoryAllocation;
// 	}
}
//	__host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
uint64_t cudaMallocPitch(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: cudaMallocPitch (width = %d)\n", malloc_width_inbytes);
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
uint64_t cudaMallocArray(LiveProcess *process, ThreadContext *tc) {
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: cudaMallocArray: devPtr32 = %d\n", ((*array)->devPtr32));
// 	((*array)->devPtr32) = (int) (long long) ((*array)->devPtr);
// 	if ( ((*array)->devPtr) ) {
// 		return g_last_cudaError = cudaSuccess;
// 	} else {
// 		return g_last_cudaError = cudaErrorMemoryAllocation;
// 	}
}

//__host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
uint64_t cudaFree(LiveProcess *process, ThreadContext *tc) {
    // TODO...  manage g_global_mem space?
    return g_last_cudaError = cudaSuccess;
}
//__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
uint64_t cudaFreeHost(LiveProcess *process, ThreadContext *tc) {
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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
uint64_t cudaFreeArray(LiveProcess *process, ThreadContext *tc) {
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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

// __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
uint64_t cudaMemcpy(LiveProcess *process, ThreadContext *tc) {
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);
    uint64_t arg1 = process->getSyscallArg(tc, index);
    uint64_t arg2 = process->getSyscallArg(tc, index);
    uint64_t arg3 = process->getSyscallArg(tc, index);

    void *dst;
    const void *src;
    uint8_t *buf;
    uint8_t *buf2;
    size_t count = (size_t)(arg2);
    enum cudaMemcpyKind kind = (enum cudaMemcpyKind)(arg3);



    CUctx_st *context = GPGPUSim_Context(process, tc);
    gpgpu_t *gpu = context->get_device()->get_gpgpu();


    if(gpu->useGem5Mem) {
        dst = (void*)(arg0);
        src = (const void *)(arg1);
    } else {
        if(kind == cudaMemcpyHostToDevice)
        {
            dst = (void*)(arg0); //should already be in gpu's address space

            buf = new uint8_t[count];
            tc->getMemProxy().readBlob(arg1, buf, (int)count);

            src = (const void *)buf;
        }
        else if(kind == cudaMemcpyDeviceToHost)
        {
            dst = (void *)arg0;
            src = (const void *)(arg1);
        }
        else if(kind == cudaMemcpyDeviceToDevice)
        {
            dst = (void*)(arg0); //should already be in gpu's address space
            src = (const void *)(arg1);
        }
    }

   //CUctx_st *context = GPGPUSim_Context();
   //gpgpu_t *gpu = context->get_device()->get_gpgpu();
   printf("GPGPU-Sim PTX: cudaMemcpy(): devPtr = %p\n", dst);
   if( kind == cudaMemcpyHostToDevice )
       g_stream_manager->push( stream_operation(src,(size_t)dst,count,0) );
   else if( kind == cudaMemcpyDeviceToHost )
       g_stream_manager->push( stream_operation((size_t)src,dst,count,0) );
   else if( kind == cudaMemcpyDeviceToDevice )
       g_stream_manager->push( stream_operation((size_t)src,(size_t)dst,count,0) );
   else {
      printf("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
      abort();
   }

   bool suspend = gpu->gem5_spa->setUnblock();
   assert(suspend);
   if (suspend) {
       tc->suspend();
   }

   return g_last_cudaError = cudaSuccess;
}


uint64_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	gpgpu_ptx_sim_init_memory();
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: cudaMemcpy(): devPtr = %p\n", dst);
// 	if( kind == cudaMemcpyHostToDevice )
// 		gpgpu_ptx_sim_memcpy_to_gpu( (size_t)dst, src, count );
// 	else if( kind == cudaMemcpyDeviceToHost )
// 		gpgpu_ptx_sim_memcpy_from_gpu( dst, (size_t)src, count );
// 	else if( kind == cudaMemcpyDeviceToDevice )
// 		gpgpu_ptx_sim_memcpy_gpu_to_gpu( (size_t)dst, (size_t)src, count );
// 	else {
// 		printf("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
// 		abort();
// 	}
// 	return g_last_cudaError = cudaSuccess;
}

//__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
uint64_t cudaMemcpyToArray(LiveProcess *process, ThreadContext *tc) {
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: cudaMemcpyToArray\n");
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
uint64_t cudaMemcpyFromArray(LiveProcess *process, ThreadContext *tc) {
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
        return g_last_cudaError = cudaErrorUnknown;
}


//__host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
uint64_t cudaMemcpyArrayToArray(LiveProcess *process, ThreadContext *tc) {
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
        return g_last_cudaError = cudaErrorUnknown;
}


//__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
uint64_t cudaMemcpy2D(LiveProcess *process, ThreadContext *tc) {
        int index = 1;
        uint64_t arg0 = process->getSyscallArg(tc, index);
        uint64_t arg1 = process->getSyscallArg(tc, index);
        uint64_t arg2 = process->getSyscallArg(tc, index);
        uint64_t arg3 = process->getSyscallArg(tc, index);
        uint64_t arg4 = process->getSyscallArg(tc, index);
        uint64_t arg5 = process->getSyscallArg(tc, index);
        uint64_t arg6 = process->getSyscallArg(tc, index);


        cuda_not_implemented(__my_func__,__LINE__);

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
        return g_last_cudaError = cudaSuccess;
}


//__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
uint64_t cudaMemcpy2DToArray(LiveProcess *process, ThreadContext *tc) {
        int index = 1;
        uint64_t arg0 = process->getSyscallArg(tc, index);
        uint64_t arg1 = process->getSyscallArg(tc, index);
        uint64_t arg2 = process->getSyscallArg(tc, index);
        uint64_t arg3 = process->getSyscallArg(tc, index);
        uint64_t arg4 = process->getSyscallArg(tc, index);
        uint64_t arg5 = process->getSyscallArg(tc, index);
        uint64_t arg6 = process->getSyscallArg(tc, index);
        uint64_t arg7 = process->getSyscallArg(tc, index);

        cuda_not_implemented(__my_func__,__LINE__);

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
        return g_last_cudaError = cudaSuccess;
}


//__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
uint64_t cudaMemcpy2DFromArray(LiveProcess *process, ThreadContext *tc) {
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}


//__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
uint64_t cudaMemcpy2DArrayToArray(LiveProcess *process, ThreadContext *tc) {
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}


//__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice)) {
uint64_t cudaMemcpyToSymbol(LiveProcess *process, ThreadContext *tc) {
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
// 	uint64_t arg2 = process->getSyscallArg(tc, index);
// 	uint64_t arg3 = process->getSyscallArg(tc, index);
// 	uint64_t arg4 = process->getSyscallArg(tc, index);
//
// 	cuda_not_implemented(__my_func__,__LINE__);
//
// 	const char *symbol = (const char *)(arg0);
// 	const void *src = (const void *)(arg1);
// 	size_t count = (size_t)(arg2);
// 	size_t offset = (size_t)(arg3);
// 	enum cudaMemcpyKind kind = (enum cudaMemcpyKind)(arg4); //__dv(cudaMemcpyDeviceToDevice)
//
// 	assert(kind == cudaMemcpyHostToDevice);
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: cudaMemcpyToSymbol: symbol = %p\n", symbol);
// 	gpgpu_ptx_sim_memcpy_symbol(symbol,src,count,offset,1);
// 	return g_last_cudaError = cudaSuccess;
}


//__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) {
uint64_t cudaMemcpyFromSymbol(LiveProcess *process, ThreadContext *tc) {
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: cudaMemcpyFromSymbol: symbol = %p\n", symbol);
// 	gpgpu_ptx_sim_memcpy_symbol(symbol,dst,count,offset,0);
// 	return g_last_cudaError = cudaSuccess;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

//	__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
uint64_t cudaMemcpyAsync(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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
uint64_t cudaMemcpyToArrayAsync(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}


//	__host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
uint64_t cudaMemcpyFromArrayAsync(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}


//	__host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
uint64_t cudaMemcpy2DAsync(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}


//	__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
uint64_t cudaMemcpy2DToArrayAsync(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}


//	__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
uint64_t cudaMemcpy2DFromArrayAsync(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

//	__host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count)
uint64_t cudaMemset(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);
    uint64_t arg1 = process->getSyscallArg(tc, index);
    uint64_t arg2 = process->getSyscallArg(tc, index);

    void *mem = (void *)arg0;
    int c = (int)arg1;
    size_t count = (size_t)arg2;

    CUctx_st *context = GPGPUSim_Context(process, tc);
    gpgpu_t *gpu = context->get_device()->get_gpgpu();
    if(gpu->useGem5Mem) {
        DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: setting %zu bytes of memory to 0x%x starting at 0x%Lx... ",
                count, (unsigned char) c, (unsigned long long) mem );
        unsigned char c_value = (unsigned char)c;
        for (unsigned n=0; n < count; n ++ )
            gpu->gem5_spa->writeFunctional((Addr)mem+n, 1, const_cast<const uint8_t*>(&c_value));
            //g_global_mem->write(dst_start_addr+n,1,&c_value);
        DPRINTF(GPGPUSyscalls,  " done.\n");
        //uint8 *buf = new uint8_t[count];
        //m5_spa->readFunctional((Addr)src, count, buf);
        //m5_spa->writeFunctional((Addr)dst, count, const_cast<const uint8_t*>(buf));
    } else {
        gpu->gpu_memset((size_t)mem, c, count);
    }
    return g_last_cudaError = cudaSuccess;
}

//__host__ cudaError_t CUDARTAPI cudaMemset2D(void *mem, size_t pitch, int c, size_t width, size_t height)
uint64_t cudaMemset2D(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

//	__host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const char *symbol)
uint64_t cudaGetSymbolAddress(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}


//__host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const char *symbol)
uint64_t cudaGetSymbolSize(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
        *                                                                              *
        *                                                                              *
        *                                                                              *
*******************************************************************************/
//	__host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
uint64_t cudaGetDeviceCount(LiveProcess *process, ThreadContext *tc)
{
   int index = 1;
   uint64_t arg0 = process->getSyscallArg(tc, index);

   _cuda_device_id *dev = GPGPUSim_Init(process, tc);
   int count = dev->num_devices();

   tc->getMemProxy().writeBlob(arg0, (uint8_t*)(&count), sizeof(int));
   return g_last_cudaError = cudaSuccess;
}

extern unsigned int warp_size;

// __host__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
uint64_t cudaGetDeviceProperties(LiveProcess *process, ThreadContext *tc)
{
   int index = 1;
   uint64_t arg0 = process->getSyscallArg(tc, index);
   uint64_t arg1 = process->getSyscallArg(tc, index);
   int device = (int)arg1;

   _cuda_device_id *dev = GPGPUSim_Init(process, tc);
   if (device <= dev->num_devices() )  {
      const struct cudaDeviceProp prop = *dev->get_prop();
      tc->getMemProxy().writeBlob(arg0, (uint8_t*)(&prop), sizeof(struct cudaDeviceProp));

      return g_last_cudaError = cudaSuccess;
   } else {
      return g_last_cudaError = cudaErrorInvalidDevice;
   }
}

// __host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
uint64_t cudaChooseDevice(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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

// __host__ cudaError_t CUDARTAPI cudaSetDevice(int device)
uint64_t cudaSetDevice(LiveProcess *process, ThreadContext *tc)
{
   int index = 1;
   uint64_t arg0 = process->getSyscallArg(tc, index);

   int device = (int)arg0;

   //set the active device to run cuda
   if ( device <= GPGPUSim_Init(process, tc)->num_devices() ) {
       g_active_device = device;
       return g_last_cudaError = cudaSuccess;
   } else {
      return g_last_cudaError = cudaErrorInvalidDevice;
   }
}

// __host__ cudaError_t CUDARTAPI cudaGetDevice(int *device)
uint64_t cudaGetDevice(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	int *device = (int *)arg0;
//
// 	*device = g_active_device;
// 	return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX))
uint64_t cudaBindTexture(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);
    uint64_t arg1 = process->getSyscallArg(tc, index);
    uint64_t arg2 = process->getSyscallArg(tc, index);
    uint64_t arg3 = process->getSyscallArg(tc, index);
    uint64_t arg4 = process->getSyscallArg(tc, index);



    size_t *offset = (size_t *)arg0;

    uint8_t *buf = new uint8_t[sizeof(const struct textureReference)];
    tc->getMemProxy().readBlob(arg1, buf, sizeof(const struct textureReference));
    const struct textureReference *texref = (const struct textureReference *)buf;

    const void *devPtr = (const void *)arg2;

    uint8_t *buf2 = new uint8_t[sizeof(const struct cudaChannelFormatDesc)];
    tc->getMemProxy().readBlob(arg3, buf2, sizeof(const struct cudaChannelFormatDesc));
    const struct cudaChannelFormatDesc *desc = (const struct cudaChannelFormatDesc *)buf2;

    size_t size = (size_t)arg4; //__dv(UINT_MAX)


    CUctx_st *context = GPGPUSim_Context(process, tc);
    gpgpu_t *gpu = context->get_device()->get_gpgpu();
    printf("GPGPU-Sim PTX: in cudaBindTexture: sizeof(struct textureReference) = %zu\n", sizeof(struct textureReference));
    struct cudaArray *array;
    array = (struct cudaArray*) malloc(sizeof(struct cudaArray));
    array->desc = *desc;
    array->size = size;
    array->width = size;
    array->height = 1;
    array->dimensions = 1;
    array->devPtr = (void*)devPtr;
    array->devPtr32 = (int)(long long)devPtr;
    offset = 0;
    printf("GPGPU-Sim PTX:   size = %zu\n", size);
    printf("GPGPU-Sim PTX:   texref = %p, array = %p\n", texref, array);
    printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
    printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n", gpu->gpgpu_ptx_sim_findNamefromTexture((const struct textureReference *)arg1));
    printf("GPGPU-Sim PTX:   ChannelFormatDesc: x=%d, y=%d, z=%d, w=%d\n", desc->x, desc->y, desc->z, desc->w);
    printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
    //gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
    gpu->gpgpu_ptx_sim_bindTextureToArray((const struct textureReference *)arg1, array);
    //devPtr = (void*)(long long)array->devPtr32;
    printf("GPGPU-Sim PTX: devPtr = %p\n", devPtr);
    return g_last_cudaError = cudaSuccess;

// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: in cudaBindTexture: sizeof(struct textureReference) = %zu\n", sizeof(struct textureReference));
// 	struct cudaArray *array;
// 	array = (struct cudaArray*) malloc(sizeof(struct cudaArray));
// 	array->desc = *desc;
// 	array->size = size;
// 	array->width = size;
// 	array->height = 1;
// 	array->dimensions = 1;
// 	array->devPtr = (void*)devPtr;
// 	array->devPtr32  = (int)(long long)devPtr;
// 	offset = 0;
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX:   size = %zu\n", size);
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX:   texref = %p, array = %p\n", texref, array);
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
// 	//printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n", (const struct textureReference *)arg1);
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX:   ChannelFormatDesc: x=%d, y=%d, z=%d, w=%d\n", desc->x, desc->y, desc->z, desc->w);
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
// 	//gpgpu_ptx_sim_bindTextureToArray(texref, array);
// 	gpgpu_ptx_sim_bindTextureToArray((const struct textureReference *)arg1, array);
// 	//devPtr = (void*)(long long)array->devPtr32;
// 	DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: devPtr = %p\n", devPtr);
// 	return g_last_cudaError = cudaSuccess;
}


// __host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
uint64_t cudaBindTextureToArray(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
// 	uint64_t arg2 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	const struct textureReference *texref = (const struct textureReference *)arg0;
// 	const struct cudaArray *array = (const struct cudaArray *)arg1;
// 	const struct cudaChannelFormatDesc *desc = (const struct cudaChannelFormatDesc *)arg2;
//
// 	printf("GPGPU-Sim PTX: in cudaBindTextureToArray: %p %p\n", texref, array);
// 	printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
// 	//printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n", gpgpu_ptx_sim_findNamefromTexture(texref));
// 	printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
// 	gpgpu_ptx_sim_bindTextureToArray(texref, array);
// 	return g_last_cudaError = cudaSuccess;
}


// __host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref)
uint64_t cudaUnbindTexture(LiveProcess *process, ThreadContext *tc)
{
        //int index = 1;
        //uint64_t arg0 = process->getSyscallArg(tc, index);

        //const struct textureReference *texref = (const struct textureReference *)arg0;
   cuda_not_implemented(__my_func__,__LINE__);

        return g_last_cudaError = cudaSuccess;
}


// __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref)
uint64_t cudaGetTextureAlignmentOffset(LiveProcess *process, ThreadContext *tc)
{
 cuda_not_implemented(__my_func__,__LINE__);
 return g_last_cudaError = cudaErrorUnknown;
}


// __host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const char *symbol)
uint64_t cudaGetTextureReference(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array)
uint64_t cudaGetChannelDesc(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	struct cudaChannelFormatDesc *desc = (struct cudaChannelFormatDesc *)arg0;
// 	const struct cudaArray *array = (const struct cudaArray *)arg1;
//
// 	*desc = array->desc;
// 	return g_last_cudaError = cudaSuccess;
}


// __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
uint64_t cudaCreateChannelDesc(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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
// 	return g_last_cudaError = cudaErrorUnknown;

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

// __host__ cudaError_t CUDARTAPI cudaGetLastError(void)
uint64_t cudaGetLastError(LiveProcess *process, ThreadContext *tc)
{
    return g_last_cudaError;
}

// __host__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error)
uint64_t cudaGetErrorString(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

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
// NOTE IMPORTANT: I changed this method header to pass pointers. Needs to be addressed
//                 in the function which calls the syscall.
// __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0))
uint64_t cudaConfigureCall(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);
    uint64_t arg1 = process->getSyscallArg(tc, index);
    uint64_t arg2 = process->getSyscallArg(tc, index);
    uint64_t arg3 = process->getSyscallArg(tc, index);



    uint8_t *buf = new uint8_t[sizeof(dim3)];
    tc->getMemProxy().readBlob(arg0, buf, sizeof(dim3));
    dim3 gridDim = *((dim3*)buf);

    uint8_t *buf2 = new uint8_t[sizeof(dim3)];
    tc->getMemProxy().readBlob(arg1, buf2, sizeof(dim3));
    dim3 blockDim = *((dim3*)buf2);

    size_t sharedMem = (size_t)arg2; //__dv(0)

    struct CUstream_st *s;
    if(arg3 == NULL) {
        s = 0;
    } else {
        assert(0); //this else statement is not tested, and needs to be confirmed we get code that can exercise it
        uint8_t *buf3 = new uint8_t[sizeof(struct CUstream_st)];
        tc->getMemProxy().readBlob(arg3, buf3, sizeof(struct CUstream_st));
        s = (struct CUstream_st *)buf3;

        //printf("s->get_uid() = %d\n", s->get_uid());
    }

   //actual function contents
   //struct CUstream_st *s = (struct CUstream_st *)stream;
   g_cuda_launch_stack.push_back( kernel_config(gridDim,blockDim,sharedMem,s) );
   delete buf;
   delete buf2;
   return g_last_cudaError = cudaSuccess;
}

// __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset){
uint64_t cudaSetupArgument(LiveProcess *process, ThreadContext *tc){
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);
    uint64_t arg1 = process->getSyscallArg(tc, index);
    uint64_t arg2 = process->getSyscallArg(tc, index);

    //const void *arg = (const void *)arg0;
    size_t size = (size_t)arg1;
    size_t offset = (size_t)arg2;



    uint8_t *buf = new uint8_t[size];
    tc->getMemProxy().readBlob(arg0, buf, size);
    const void *arg = (const void *)buf;

    //actual function contents
    gpgpusim_ptx_assert( !g_cuda_launch_stack.empty(), "empty launch stack" );
    kernel_config &config = g_cuda_launch_stack.back();
    config.set_arg(arg,size,offset);

    struct gpgpu_ptx_sim_arg *param = (gpgpu_ptx_sim_arg*) calloc(1,sizeof(struct gpgpu_ptx_sim_arg));
    param->m_start = arg;
    param->m_nbytes = size;
    param->m_offset = offset;

    return g_last_cudaError = cudaSuccess;
}


// __host__ cudaError_t CUDARTAPI cudaLaunch( const char *hostFun )
extern double core_time;
uint64_t cudaLaunch(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);

    const char *hostFun = (const char *)arg0;

    //actual function...
    CUctx_st* context = GPGPUSim_Context(process, tc);
    char *mode = getenv("PTX_SIM_MODE_FUNC");
    if( mode )
        sscanf(mode,"%u", &g_ptx_sim_mode);
    gpgpusim_ptx_assert( !g_cuda_launch_stack.empty(), "empty launch stack" );
    kernel_config config = g_cuda_launch_stack.back();
    struct CUstream_st *stream = config.get_stream();
    printf("\nGPGPU-Sim PTX: cudaLaunch for 0x%p (mode=%s) on stream %u\n", hostFun,
            g_ptx_sim_mode?"functional simulation":"performance simulation", stream?stream->get_uid():0 );
    kernel_info_t *grid = gpgpu_cuda_ptx_sim_init_grid(hostFun,config.get_args(),config.grid_dim(),config.block_dim(),context);
    std::string kname = grid->name();
    dim3 gridDim = config.grid_dim();
    dim3 blockDim = config.block_dim();
    printf("GPGPU-Sim PTX: pushing kernel \'%s\' to stream %u, gridDim= (%u,%u,%u) blockDim = (%u,%u,%u) \n",
            kname.c_str(), stream?stream->get_uid():0, gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z );
    stream_operation op(grid,g_ptx_sim_mode,stream);
    g_stream_manager->push(op);
    g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaSuccess;



//    int index = 1;
//    uint64_t arg0 = process->getSyscallArg(tc, index);
//
//    const char *symbol = (const char *)arg0;
//
//    DPRINTF(GPGPUSyscalls, "\n\n\n");
//    char *mode = getenv("PTX_SIM_MODE_FUNC");
//    if( mode )
//       sscanf(mode,"%u", &g_ptx_sim_mode);
//    DPRINTF(GPGPUSyscalls, "GPGPU-Sim PTX: cudaLaunch for %p (mode=%s)\n", symbol,
//                   g_ptx_sim_mode?"functional simulation":"performance simulation");
//
//    if( !m5_spa->isRunning() ) {
//       m5_spa->beginRunning();
//       if( g_ptx_sim_mode ) {
//          //gpgpu_ptx_sim_main_func( symbol, g_cudaGridDim, g_cudaBlockDim, g_ptx_sim_params );
//          printf("GPGPU-Sim's functional mode does not work, so it could not be ported to g3 :-(.\n");
//          assert(0);
//       }
//       else {
//          gpgpu_ptx_sim_main_perf_gem5( symbol, g_cudaGridDim, g_cudaBlockDim, g_ptx_sim_params, m5_spa->getLaunchDelay());
//       }
//    } else {
//       launchedKernel_t *kern = (launchedKernel_t *)malloc(sizeof(launchedKernel_t));
//       kern->symbol = symbol;
//       kern->cudaGridDim.x = g_cudaGridDim.x;
//       kern->cudaGridDim.y = g_cudaGridDim.y;
//       kern->cudaGridDim.z = g_cudaGridDim.z;
//       kern->cudaBlockDim.x = g_cudaBlockDim.x;
//       kern->cudaBlockDim.y = g_cudaBlockDim.y;
//       kern->cudaBlockDim.z = g_cudaBlockDim.z;
//       kern->ptx_sim_params = g_ptx_sim_params;
//       kern->launchTime = core_time;
//
//       m5_spa->queueKernel(kern);
//    }
//    g_ptx_sim_params=NULL;
//
//    if (!m5_spa->getNonBlocking()) {
//       tc->suspend();
//    }
//
// 	return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream)
uint64_t cudaStreamCreate(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

// __host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
uint64_t cudaStreamDestroy(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

// __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
uint64_t cudaStreamSynchronize(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

// __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream)
uint64_t cudaStreamQuery(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event)
uint64_t cudaEventCreate(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);


    cudaEvent_t *event = (cudaEvent_t *)arg0;

    CUevent_st *e = new CUevent_st(false);
    DPRINTF(GPGPUSyscalls, "Event pointer %p\n", e);
    g_timer_events[e->get_uid()] = e;
#if CUDART_VERSION >= 3000
    // NOTE: when this write happens the event given to the user is a pointer in
    // simulator space, not user space. If the application were to try to dereference it
    // would get a seg fault. All other uses of event when the user passes it in do not
    // need to call read blob on it.
    tc->getMemProxy().writeBlob((uint64_t)event, (uint8_t*)(&e), sizeof(cudaEvent_t));
    //*event = e;
#else
    tc->getMemProxy().writeBlob((uint64_t)event, (uint8_t*)(e->get_uid()), sizeof(cudaEvent_t));
    //*event = e->get_uid();
#endif
    return g_last_cudaError = cudaSuccess;
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

// __host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
uint64_t cudaEventRecord(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);
    uint64_t arg1 = process->getSyscallArg(tc, index);



    cudaStream_t stream;
    assert(arg0 != 0);

    cudaEvent_t event = (cudaEvent_t)arg0;

    CUevent_st *e = get_event(event);
    if( !e ) return g_last_cudaError = cudaErrorUnknown;

    struct CUstream_st *s;
    if (arg1 != 0) {
        tc->getMemProxy().readBlob((uint64_t)arg1, (uint8_t*)&stream, sizeof(cudaStream_t));
        s = (struct CUstream_st *)stream;
    } else {
        s = NULL;
    }
    stream_operation op(e,s);
    g_stream_manager->push(op);
    return g_last_cudaError = cudaSuccess;
}

// __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event)
uint64_t cudaEventQuery(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);



    cudaEvent_t event = (cudaEvent_t)arg0;

    CUevent_st *e = get_event(event);
    if( e == NULL ) {
        return g_last_cudaError = cudaErrorInvalidValue;
    } else if( e->done() ) {
        return g_last_cudaError = cudaSuccess;
    } else {
        return g_last_cudaError = cudaErrorNotReady;
    }
}

// __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event)
uint64_t cudaEventSynchronize(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);



    cudaEvent_t event = (cudaEvent_t)arg0;

    printf("GPGPU-Sim API: cudaEventSynchronize ** waiting for event\n");
    fflush(stdout);
    CUevent_st *e = get_event(event);
    DPRINTF(GPGPUSyscalls, "Event pointer %p\n", e);
    if( !e->done() ) {
        DPRINTF(GPGPUSyscalls, "Blocking on event %d\n", e->get_uid());
        e->set_needs_unblock(true);
        tc->suspend();
    }
    printf("GPGPU-Sim API: cudaEventSynchronize ** event detected\n");
    fflush(stdout);
    return g_last_cudaError = cudaSuccess;
}

// __host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event)
uint64_t cudaEventDestroy(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);



    cudaEvent_t event = (cudaEvent_t)arg0;

    CUevent_st *e = get_event(event);
    unsigned event_uid = e->get_uid();
    event_tracker_t::iterator pe = g_timer_events.find(event_uid);
    if( pe == g_timer_events.end() )
        return g_last_cudaError = cudaErrorInvalidValue;
    g_timer_events.erase(pe);
    return g_last_cudaError = cudaSuccess;
}


// __host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
uint64_t cudaEventElapsedTime(LiveProcess *process, ThreadContext *tc)
{
    int index = 1;
    uint64_t arg0 = process->getSyscallArg(tc, index);
    uint64_t arg1 = process->getSyscallArg(tc, index);
    uint64_t arg2 = process->getSyscallArg(tc, index);



    float ms;

    cudaEvent_t start = (cudaEvent_t)arg1;
    cudaEvent_t end = (cudaEvent_t)arg2;

    CUevent_st *s = get_event(start);
    CUevent_st *e = get_event(end);
    if( s==NULL || e==NULL )
        return g_last_cudaError = cudaErrorUnknown;
    //elapsed_time = e->clock() - s->clock();  // NOTE: I don't think this is right

    //*ms = 1000*elapsed_time;
    unsigned long long elapsed_ticks = e->ticks() - s->ticks();
    ms = (double)elapsed_ticks/1e9; // 1e9 ticks per ms
    tc->getMemProxy().writeBlob(arg0, (uint8_t*)(&ms), sizeof(float));

    return g_last_cudaError = cudaSuccess;
}



/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// __host__ cudaError_t CUDARTAPI cudaThreadExit(void)
uint64_t cudaThreadExit(LiveProcess *process, ThreadContext *tc)
{
    //Called on host side
    _cuda_device_id *dev = GPGPUSim_Init(process, tc);
    bool suspend = dev->get_gpgpu()->gem5_spa->setUnblock();
    if (suspend) {
        tc->suspend();
    }
    return g_last_cudaError = cudaSuccess;
}


// __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void)
uint64_t cudaThreadSynchronize(LiveProcess *process, ThreadContext *tc)
{
    //Called on host side
    _cuda_device_id *dev = GPGPUSim_Init(process, tc);
    bool suspend = dev->get_gpgpu()->gem5_spa->setUnblock();
    if (suspend) {
        tc->suspend();
    }
    return g_last_cudaError = cudaSuccess;
}

// int CUDARTAPI __cudaSynchronizeThreads(void**, void*)
uint64_t __cudaSynchronizeThreads(LiveProcess *process, ThreadContext *tc)
{
        //Called on host side
   cuda_not_implemented(__my_func__,__LINE__);
// 	bool suspend = m5_spa->setUnblock();
// 	if (suspend) {
// 		tc->suspend();
// 	}
        return g_last_cudaError = cudaSuccess;
}


/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
*******************************************************************************/

// void** CUDARTAPI __cudaRegisterFatBinary( void *fatCubin )
uint64_t __cudaRegisterFatBinary(LiveProcess *process, ThreadContext *tc)
{
   int i = 0;

#if (CUDART_VERSION < 2010)
   printf("GPGPU-Sim PTX: ERROR ** this version of GPGPU-Sim requires CUDA 2.1 or higher\n");
   exit(1);
#endif

   //get parameters
   int index = 1;
   uint64_t arg0 = process->getSyscallArg(tc, index);
   uint64_t arg1 = process->getSyscallArg(tc, index);

   //Get primary arguments
   void *fatCubin = (void *)(new uint8_t[sizeof(struct __cudaFatCudaBinaryRec)]);
   tc->getMemProxy().readBlob(arg0, (uint8_t *)fatCubin, sizeof(struct __cudaFatCudaBinaryRec));
   __cudaFatCudaBinary *info =   (__cudaFatCudaBinary *)fatCubin;

   int buf_size = (int)(arg1);
   if (buf_size < 0) { gpgpusim_ptx_error("Used wrong __cudaRegisterFatBinary call!!! Did you run the sizeHack.py?\n"); }

   //Translate members of fatCubin
   //1. ptx member
   uint8_t *buf2 = (uint8_t *)malloc(sizeof(__cudaFatPtxEntry)*2);
   __cudaFatPtxEntry *ptx;
   int ptxCount = 0;
   do {
      realloc (buf2, sizeof(__cudaFatPtxEntry)*(ptxCount+1) );
      tc->getMemProxy().readBlob((uint64_t)(info->ptx+ptxCount), buf2+sizeof(__cudaFatPtxEntry)*ptxCount, sizeof(__cudaFatPtxEntry));

      ptx = (__cudaFatPtxEntry *)buf2+ptxCount;
      if(ptx->ptx != 0) {
         uint8_t *buf3 = (uint8_t *)malloc(buf_size);
         tc->getMemProxy().readBlob((uint64_t)ptx->ptx, buf3, buf_size);
         uint8_t *buf4 = new uint8_t[MAX_STRING_LEN];
         tc->getMemProxy().readBlob((uint64_t)ptx->gpuProfileName, buf4, MAX_STRING_LEN);
         for(i=0; i<MAX_STRING_LEN; i++)
         {
            if(buf4[i] == '\0')
               break;
         }
         if(i == MAX_STRING_LEN){
            gpgpusim_ptx_error("WAYYY TO LONG OF A FUNCTION NAME???:?\n");
            delete buf4;
            return -1;
         }

         ptx->ptx = (char *)buf3;
         ptx->gpuProfileName = (char *)buf4;
      }
      ptxCount++;
   } while(ptx->gpuProfileName != 0);
   info->ptx = (__cudaFatPtxEntry *)buf2;

   //2. ident member
   uint8_t *buf5 = new uint8_t[MAX_STRING_LEN];
   tc->getMemProxy().readBlob((uint64_t)info->ident, buf5, MAX_STRING_LEN);
   for(i=0; i<MAX_STRING_LEN; i++)
   {
      if(buf5[i] == '\0')
         break;
   }
   if(i == MAX_STRING_LEN){
      gpgpusim_ptx_error("WAY TO LONG OF A FILE NAME???:?\n");
      delete buf5;
      return -1;
   }
   info->ident = (char *)buf5;

   //Actual function...
   CUctx_st *context = GPGPUSim_Context(process, tc);
   static unsigned next_fat_bin_handle = 1;
   static unsigned source_num=1;
   unsigned fat_cubin_handle = next_fat_bin_handle++;
   //__cudaFatCudaBinary *info =   (__cudaFatCudaBinary *)fatCubin;
   assert( info->version >= 3 );
   unsigned num_ptx_versions=0;
   unsigned max_capability=0;
   unsigned selected_capability=0;
   bool found=false;
   unsigned forced_max_capability = context->get_device()->get_gpgpu()->get_config().get_forced_max_capability();
   while( info->ptx[num_ptx_versions].gpuProfileName != NULL ) {
      unsigned capability=0;
      sscanf(info->ptx[num_ptx_versions].gpuProfileName,"compute_%u",&capability);
      printf("GPGPU-Sim PTX: __cudaRegisterFatBinary found PTX versions for '%s', ", info->ident);
      printf("capability = %s\n", info->ptx[num_ptx_versions].gpuProfileName );
      if( forced_max_capability ) {
          if( capability > max_capability && capability <= forced_max_capability ) {
             found = true;
             max_capability=capability;
             selected_capability = num_ptx_versions;
          }
      } else {
          if( capability > max_capability ) {
             found = true;
             max_capability=capability;
             selected_capability = num_ptx_versions;
          }
      }
      num_ptx_versions++;
   }
   if( found  ) {
      printf("GPGPU-Sim PTX: Loading PTX for %s, capability = %s\n",
             info->ident, info->ptx[selected_capability].gpuProfileName );
      symbol_table *symtab;
      const char *ptx = info->ptx[selected_capability].ptx;
      if(context->get_device()->get_gpgpu()->get_config().convert_to_ptxplus() ) {
         assert(0);
         char *ptxplus_str = gpgpu_ptx_sim_convert_ptx_to_ptxplus(ptx, info->cubin[selected_capability].cubin, source_num++,
                                                context->get_device()->get_gpgpu()->get_config().saved_converted_ptxplus());
         symtab=gpgpu_ptx_sim_load_ptx_from_string(ptxplus_str,source_num);
         context->add_binary(symtab,fat_cubin_handle);
         gpgpu_ptxinfo_load_from_string(ptx,source_num);
         delete[] ptxplus_str;
      } else {
         symtab=gpgpu_ptx_sim_load_ptx_from_string(ptx,source_num);
         context->add_binary(symtab,fat_cubin_handle);
         gpgpu_ptxinfo_load_from_string( ptx, source_num );
      }
      source_num++;
      load_static_globals(symtab,STATIC_ALLOC_LIMIT,0xFFFFFFFF,context->get_device()->get_gpgpu());
      load_constants(symtab,STATIC_ALLOC_LIMIT,context->get_device()->get_gpgpu());
   } else {
      printf("GPGPU-Sim PTX: warning -- did not find an appropriate PTX in cubin\n");
   }
   //return (void**)fat_cubin_handle;
   return fat_cubin_handle;
}
// void __cudaUnregisterFatBinary(void **fatCubinHandle)
uint64_t __cudaUnregisterFatBinary(LiveProcess *process, ThreadContext *tc)
{
    ;
}


//  void CUDARTAPI __cudaRegisterFunction(
// 		 void   **fatCubinHandle,
//  const char    *hostFun,
//  char    *deviceFun,
//  const char    *deviceName,
//  int      thread_limit,
//  uint3   *tid,
//  uint3   *bid,
//  dim3    *bDim,
//  dim3    *gDim
// 									  )
uint64_t __cudaRegisterFunction(LiveProcess *process, ThreadContext *tc)
{
   // >6 params, so they have been packed
   // 1st decode the package
   int *arg_sizes;
   char *args;
   decode_package(process, tc, &arg_sizes, &args);

   // 2nd, extract parameters from package
   int args_off = 0;
   int arg_sizes_off = 0;

   void **fatCubinHandle = *((void ***)unpack(args, args_off, arg_sizes, arg_sizes_off));
   const char *hostFun = *((const char **)unpack(args, args_off, arg_sizes, arg_sizes_off));
   char *deviceFun = *((char **)unpack(args, args_off, arg_sizes, arg_sizes_off));
   const char *deviceName = (const char *)unpack(args, args_off, arg_sizes, arg_sizes_off);
   int thread_limit = *((int *)unpack(args, args_off, arg_sizes, arg_sizes_off));
   uint3 *tid = *((uint3 **)unpack(args, args_off, arg_sizes, arg_sizes_off));
   uint3 *bid = *((uint3 **)unpack(args, args_off, arg_sizes, arg_sizes_off));
   dim3 *bDim = *((dim3 **)unpack(args, args_off, arg_sizes, arg_sizes_off));
   dim3 *gDim = *((dim3 **)unpack(args, args_off, arg_sizes, arg_sizes_off));

   uint8_t *buf3 = new uint8_t[MAX_STRING_LEN];
   tc->getMemProxy().readBlob((uint64_t)deviceFun, buf3, MAX_STRING_LEN);

   //check that string is a valid length
   int i;
   for(i=0; i<MAX_STRING_LEN; i++)
   {
      if(buf3[i] == '\0')
         break;
   }
   if(i == MAX_STRING_LEN){
      gpgpusim_ptx_error("WAYYY TO LONG OF A FUNCTION NAME???:?\n");
      delete buf3;
      return -1;
   }
   deviceFun = (char *)buf3;

//    uint8_t *buf4 = new uint8_t[MAX_STRING_LEN];
//    tp->readBlob((uint64_t)hostFun, buf4, MAX_STRING_LEN);

//    //check that string is a valid length
//    for(i=0; i<MAX_STRING_LEN; i++)
//    {
//       if(buf4[i] == '\0')
//          break;
//    }
//    if(i == MAX_STRING_LEN){
//       gpgpusim_ptx_error("WAYYY TO LONG OF A FUNCTION NAME???:?\n");
//       delete buf4;
//       return -1;
//    }
//    hostFun = (const char *)buf4;


   //actual function
   CUctx_st *context = GPGPUSim_Context(process, tc);
   unsigned fat_cubin_handle = (unsigned)(unsigned long long)fatCubinHandle;
   printf("GPGPU-Sim PTX: __cudaRegisterFunction %s : hostFun 0x%p, fat_cubin_handle = %u\n",
          deviceFun, hostFun, fat_cubin_handle);
   context->register_function( fat_cubin_handle, hostFun, deviceFun );

   return 0;
}

//  extern void __cudaRegisterVar(
// 		 void **fatCubinHandle,
//  char *hostVar, //pointer to...something
//  char *deviceAddress, //name of variable
//  const char *deviceName, //name of variable (same as above)
//  int ext,
//  int size,
//  int constant,
//  int global )
extern uint64_t __cudaRegisterVar(LiveProcess *process, ThreadContext *tc)
{
    // >6 params, so they have been packed
    // 1st decode the package
    int *arg_sizes;
    char *args;
    decode_package(process, tc, &arg_sizes, &args);

    // 2nd, extract parameters from package

    int args_off = 0;
    int arg_sizes_off = 0;

    void **fatCubinHandle = *((void ***)unpack(args, args_off, arg_sizes, arg_sizes_off));
    char *hostVar = *((char **)unpack(args, args_off, arg_sizes, arg_sizes_off));

    uint64_t deviceAddress_ptr = *((uint64_t *)unpack(args, args_off, arg_sizes, arg_sizes_off));
    uint8_t *buf = new uint8_t[MAX_STRING_LEN];
    tc->getMemProxy().readBlob((uint64_t)deviceAddress_ptr, buf, MAX_STRING_LEN);

    //check that string is a valid length
    int i;
    for(i=0; i<MAX_STRING_LEN; i++)
    {
        if(buf[i] == '\0')
            break;
    }
    if(i == MAX_STRING_LEN){
        gpgpusim_ptx_error("WAYYY TO LONG OF A Variable NAME???:?\n");
        delete buf;
        return -1;
    }
    char *deviceAddress = (char *)buf;


    uint64_t deviceName_ptr = *((uint64_t *)unpack(args, args_off, arg_sizes, arg_sizes_off));
    uint8_t *buf2 = new uint8_t[MAX_STRING_LEN];
    tc->getMemProxy().readBlob((uint64_t)deviceName_ptr, buf2, MAX_STRING_LEN);

    //check that string is a valid length
    for(i=0; i<MAX_STRING_LEN; i++)
    {
        if(buf2[i] == '\0')
            break;
    }
    if(i == MAX_STRING_LEN){
        gpgpusim_ptx_error("WAYYY TO LONG OF A Variable NAME???:?\n");
        delete buf;
        delete buf2;
        return -1;
    }
    const char *deviceName = (const char *)buf2;


    int ext = *((int *)unpack(args, args_off, arg_sizes, arg_sizes_off));
    int size = *((int *)unpack(args, args_off, arg_sizes, arg_sizes_off));
    int constant = *((int *)unpack(args, args_off, arg_sizes, arg_sizes_off));
    int global = *((int *)unpack(args, args_off, arg_sizes, arg_sizes_off));

    printf("GPGPU-Sim PTX: __cudaRegisterVar: hostVar = %p; deviceAddress = %s; deviceName = %s\n", hostVar, deviceAddress, deviceName);
    printf("GPGPU-Sim PTX: __cudaRegisterVar: Registering const memory space of %d bytes\n", size);
    fflush(stdout);
    if ( constant && !global && !ext ) {
        gpgpu_ptx_sim_register_const_variable(hostVar,deviceName,size);
    } else if ( !constant && !global && !ext ) {
        gpgpu_ptx_sim_register_global_variable(hostVar,deviceName,size);
    } else cuda_not_implemented(__my_func__,__LINE__);
}


//  void __cudaRegisterShared(
// 		 void **fatCubinHandle,
//  void **devicePtr
// 						  )
uint64_t __cudaRegisterShared(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
//
// 	void **fatCubinHandle = (void **)arg0;
// 	void **devicePtr = (void **)arg1;
//
// 	// we don't do anything here
// 	printf("GPGPU-Sim PTX: __cudaRegisterShared\n" );
//
// 	return 0;
}

//  void CUDARTAPI __cudaRegisterSharedVar(
// 		 void   **fatCubinHandle,
//  void   **devicePtr,
//  size_t   size,
//  size_t   alignment,
//  int      storage
// 									   )
uint64_t __cudaRegisterSharedVar(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
// 	uint64_t arg2 = process->getSyscallArg(tc, index);
// 	uint64_t arg3 = process->getSyscallArg(tc, index);
// 	uint64_t arg4 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	void **fatCubinHandle = (void **)arg0;
// 	void **devicePtr = (void **)arg1;
// 	size_t size = (size_t)arg2;
// 	size_t alignment = (size_t)arg3;
// 	int storage = (int)arg4;
//
// 	// we don't do anything here
// 	printf("GPGPU-Sim PTX: __cudaRegisterSharedVar\n" );
//
// 	return 0;
}

//  void __cudaRegisterTexture(
// 		 void **fatCubinHandle,
//  const struct textureReference *hostVar,
//  const void **deviceAddress,
//  const char *deviceName,
//  int dim,
//  int norm,
//  int ext
// 						   ) //passes in a newly created textureReference
uint64_t __cudaRegisterTexture(LiveProcess *process, ThreadContext *tc)
{
    // >6 params, so they have been packed
    // 1st decode the package
    int *arg_sizes;
    char *args;
    decode_package(process, tc, &arg_sizes, &args);

    // 2nd, extract parameters from package

    int args_off = 0;
    int arg_sizes_off = 0;

    void **fatCubinHandle = *((void ***)unpack(args, args_off, arg_sizes, arg_sizes_off));
    const struct textureReference *hostVar = *((const struct textureReference **)unpack(args, args_off, arg_sizes, arg_sizes_off));
    const void **deviceAddress = *((const void ***)unpack(args, args_off, arg_sizes, arg_sizes_off));


    uint64_t deviceName_ptr = *((uint64_t *)unpack(args, args_off, arg_sizes, arg_sizes_off));
    uint8_t *buf = new uint8_t[MAX_STRING_LEN];
    tc->getMemProxy().readBlob((uint64_t)deviceName_ptr, buf, MAX_STRING_LEN);

    //check that string is a valid length
    int i;
    for(i=0; i<MAX_STRING_LEN; i++)
    {
        if(buf[i] == '\0')
            break;
    }
    if(i == MAX_STRING_LEN){
        gpgpusim_ptx_error("WAYYY TO LONG OF A Texture Memory NAME???:?\n");
        delete buf;
        return -1;
    }
    const char *deviceName = (const char *)buf;

    int dim = *((int *)unpack(args, args_off, arg_sizes, arg_sizes_off));
    int norm = *((int *)unpack(args, args_off, arg_sizes, arg_sizes_off));
    int ext = *((int *)unpack(args, args_off, arg_sizes, arg_sizes_off));


    CUctx_st *context = GPGPUSim_Context(process, tc);
    gpgpu_t *gpu = context->get_device()->get_gpgpu();
    printf("GPGPU-Sim PTX: in __cudaRegisterTexture:\n");
    gpu->gpgpu_ptx_sim_bindNameToTexture(deviceName, hostVar);
    printf("GPGPU-Sim PTX:   int dim = %d\n", dim);
    printf("GPGPU-Sim PTX:   int norm = %d\n", norm);
    printf("GPGPU-Sim PTX:   int ext = %d\n", ext);
    printf("GPGPU-Sim PTX:   Execution warning: Not finished implementing \"%s\"\n", __my_func__ );
}

#ifndef OPENGL_SUPPORT
typedef unsigned long GLuint;
#endif

//cudaError_t cudaGLRegisterBufferObject(GLuint bufferObj)
uint64_t cudaGLRegisterBufferObject(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

        int index = 1;
        uint64_t arg0 = process->getSyscallArg(tc, index);

   cuda_not_implemented(__my_func__,__LINE__);

        GLuint bufferObj = (GLuint)arg0;

        printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
        return g_last_cudaError = cudaSuccess;
}

struct glbmap_entry {
        GLuint m_bufferObj;
        void *m_devPtr;
        size_t m_size;
        struct glbmap_entry *m_next;
};
typedef struct glbmap_entry glbmap_entry_t;

glbmap_entry_t* g_glbmap = NULL;

//cudaError_t cudaGLMapBufferObject(void** devPtr, GLuint bufferObj)
uint64_t cudaGLMapBufferObject(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
// 	uint64_t arg1 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
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
uint64_t cudaGLUnmapBufferObject(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	GLuint bufferObj = (GLuint)arg0;
//
// #ifdef OPENGL_SUPPORT
//    glbmap_entry_t *p = g_glbmap;
//    while ( p && p->m_bufferObj != bufferObj )
// 	   p = p->m_next;
//    if ( p == NULL )
// 	   return g_last_cudaError = cudaErrorUnknown;
//
//    char *data = (char *) calloc(p->m_size,1);
//    gpgpu_ptx_sim_memcpy_from_gpu( data,(size_t)p->m_devPtr,p->m_size );
//    glBufferSubData(GL_ARRAY_BUFFER,0,p->m_size,data);
//    free(data);
//
//    return g_last_cudaError = cudaSuccess;
// #else
//    fflush(stdout);
//    fflush(stderr);
//    printf("GPGPU-Sim PTX: support for OpenGL integration disabled -- exiting\n");
//    fflush(stdout);
//    exit(50);
// #endif
}

//cudaError_t cudaGLUnregisterBufferObject(GLuint bufferObj)
uint64_t cudaGLUnregisterBufferObject(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	GLuint bufferObj = (GLuint)arg0;
//
// 	printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
// 	return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION >= 2010)

//cudaError_t CUDARTAPI cudaHostAlloc(void **pHost,  size_t bytes, unsigned int flags)
uint64_t cudaHostAlloc(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

//cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
uint64_t cudaHostGetDevicePointer(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

//cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len)
uint64_t cudaSetValidDevices(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

//cudaError_t CUDARTAPI cudaSetDeviceFlags( int flags )
uint64_t cudaSetDeviceFlags(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

//cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func)
uint64_t cudaFuncGetAttributes(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

//cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, int flags)
uint64_t cudaEventCreateWithFlags(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

//cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion)
uint64_t cudaDriverGetVersion(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
//
// 	int *driverVersion = (int *)arg0;
//
// 	*driverVersion = CUDART_VERSION;
// 	return g_last_cudaError = cudaErrorUnknown;
}

//cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion)
uint64_t cudaRuntimeGetVersion(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
//
// 	int *runtimeVersion = (int *)arg0;
//
// 	*runtimeVersion = CUDART_VERSION;
// 	return g_last_cudaError = cudaErrorUnknown;
}

#endif

//cudaError_t CUDARTAPI cudaGLSetGLDevice(int device)
uint64_t cudaGLSetGLDevice(LiveProcess *process, ThreadContext *tc)
{
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;

// 	int index = 1;
// 	uint64_t arg0 = process->getSyscallArg(tc, index);
//
//    cuda_not_implemented(__my_func__,__LINE__);
//
// 	int device = (int)arg0;
//
// 	printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
// 	return g_last_cudaError = cudaErrorUnknown;
}

typedef void* HGPUNV;

//cudaError_t CUDARTAPI cudaWGLGetDevice(int *device, HGPUNV hGpu)
uint64_t cudaWGLGetDevice(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return g_last_cudaError = cudaErrorUnknown;
}

//void CUDARTAPI __cudaMutexOperation(int lock)
uint64_t __cudaMutexOperation(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return 0;
}

//void  CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val)
uint64_t __cudaTextureFetch(LiveProcess *process, ThreadContext *tc)
{
        cuda_not_implemented(__my_func__,__LINE__);
        return 0;
}

namespace cuda_math {

        //void CUDARTAPI __cudaMutexOperation(int lock)
        uint64_t __cudaMutexOperation(LiveProcess *process, ThreadContext *tc)
        {
                cuda_not_implemented(__my_func__,__LINE__);
                return 0;
        }

        //void  CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val)
        uint64_t __cudaTextureFetch(LiveProcess *process, ThreadContext *tc)
        {
                cuda_not_implemented(__my_func__,__LINE__);
                return 0;
        }

        //int CUDARTAPI __cudaSynchronizeThreads(void**, void*)
        uint64_t __cudaSynchronizeThreads(LiveProcess *process, ThreadContext *tc)
        {
   //TODO This function should syncronize if we support Asyn kernel calls
                return g_last_cudaError = cudaSuccess;
        }

        //so that m5.debug will compile
        void  __cudaTextureFetch(const void *tex, void *index, int integer, void *val){ assert(0); }
        void __cudaMutexOperation(int lock){ assert(0); }
}


typedef uint64_t (*cudaFunc_t)(LiveProcess *, ThreadContext *);

cudaFunc_t gpgpu_funcs[] = {
        cudaMalloc,             /* 0 */
        cudaMallocHost,         /* 1 */
        cudaMallocPitch,        /* 2 */
        cudaMallocArray,        /* 3 */
        cudaFree,               /* 4 */
        cudaFreeHost,           /* 5 */
        cudaFreeArray,          /* 6 */
        cudaMemcpy,             /* 7 */
        cudaMemcpyToArray,      /* 8 */
        cudaMemcpyFromArray,   /* 9 */
        cudaMemcpyArrayToArray,/* 10 */
        cudaMemcpy2D,          /* 11 */
        cudaMemcpy2DToArray,   /* 12 */
        cudaMemcpy2DFromArray, /* 13 */
        cudaMemcpy2DArrayToArray,/* 14 */
        cudaMemcpyToSymbol,    /* 15 */
        cudaMemcpyFromSymbol,  /* 16 */
        cudaMemcpyAsync,       /* 17 */
        cudaMemcpyToArrayAsync,/* 18 */
        cudaMemcpyFromArrayAsync,/* 19 */
        cudaMemcpy2DAsync,     /* 20 */
        cudaMemcpy2DToArrayAsync,/* 21 */
        cudaMemcpy2DFromArrayAsync,/* 22 */
        cudaMemset,            /* 23 */
        cudaMemset2D,          /* 24 */
        cudaGetSymbolAddress,  /* 25 */
        cudaGetSymbolSize,     /* 26 */
        cudaGetDeviceCount,    /* 27 */
        cudaGetDeviceProperties,/* 28 */
        cudaChooseDevice,      /* 29 */
        cudaSetDevice,         /* 30 */
        cudaGetDevice,         /* 31 */
        cudaBindTexture,       /* 32 */
        cudaBindTextureToArray,/* 33 */
        cudaUnbindTexture,     /* 34 */
        cudaGetTextureAlignmentOffset,/* 35 */
        cudaGetTextureReference,/* 36 */
        cudaGetChannelDesc,        /* 37 */
        cudaCreateChannelDesc, /* 38 */
        cudaGetLastError,        /* 39 */
        cudaGetErrorString,        /* 40 */
        cudaConfigureCall,        /* 41 */
        cudaSetupArgument,        /* 42 */
        cudaLaunch,        /* 43 */
        cudaStreamCreate,        /* 44 */
        cudaStreamDestroy,        /* 45 */
        cudaStreamSynchronize,        /* 46 */
        cudaStreamQuery,        /* 47 */
        cudaEventCreate,        /* 48 */
        cudaEventRecord,        /* 49 */
        cudaEventQuery,        /* 50 */
        cudaEventSynchronize,        /* 51 */
        cudaEventDestroy,        /* 52 */
        cudaEventElapsedTime,        /* 53 */
        cudaThreadExit,        /* 54 */
        cudaThreadSynchronize,        /* 55 */
        __cudaSynchronizeThreads,    /* 56 */
        __cudaRegisterFatBinary,    /* 57 */
        __cudaUnregisterFatBinary,   /* 58 */
        __cudaRegisterFunction,        /* 59 */
        __cudaRegisterVar,        /* 60 */
        __cudaRegisterShared,        /* 61 */
        __cudaRegisterSharedVar,        /* 62 */
        __cudaRegisterTexture,        /* 63 */
        cudaGLRegisterBufferObject,  /* 64 */
        cudaGLMapBufferObject,        /* 65 */
        cudaGLUnmapBufferObject,        /* 66 */
        cudaGLUnregisterBufferObject,/* 67 */
        cudaHostAlloc,        /* 68 */
        cudaHostGetDevicePointer,        /* 69 */
        cudaSetValidDevices,        /* 70 */
        cudaSetDeviceFlags,        /* 71 */
        cudaFuncGetAttributes,        /* 72 */
        cudaEventCreateWithFlags,        /* 73 */
        cudaDriverGetVersion,        /* 74 */
        cudaRuntimeGetVersion,        /* 75 */
        cudaGLSetGLDevice,        /* 76 */
        cudaWGLGetDevice,        /* 77 */
        __cudaMutexOperation,        /* 78 */
        __cudaTextureFetch,        /* 79 */
        __cudaSynchronizeThreads        /* 80 */
};


/*
 *
 */
void decode_package(LiveProcess *process, ThreadContext *tc, int **arg_sizes, char **args)
{
        //1. get buffer that holds parameters
        int index = 1;
        uint64_t arg0 = process->getSyscallArg(tc, index);
        uint64_t arg1 = process->getSyscallArg(tc, index);
        uint64_t arg2 = process->getSyscallArg(tc, index);

        int num_args = arg0;

        //now get array of argument sizes

        uint8_t *buf = new uint8_t[num_args*sizeof(int)];
        tc->getMemProxy().readBlob(arg1, buf, num_args*sizeof(int));
        *arg_sizes = (int *)buf;

        //finally, get array of arguments
        int num_bytes = 0;
        for(int i=0; i<num_args; i++)
        {
                num_bytes += (*arg_sizes)[i];
        }
        uint8_t *buf2 = new uint8_t[num_bytes];
        tc->getMemProxy().readBlob(arg2, buf2, num_bytes);
        *args = (char *)buf2;
}


/*
 * It is necessary use the pack function in cuda_runtime_api.cc and unpack function
 * in gpgpusyscalls.cc when more than 6 arguments are passed to one of the gpgpu
 * syscalls. This is because the hardware primitives in m5 only support syscalls
 * containing 6 parameters.
 */
char *unpack(char *bytes, int &bytes_off, int *lengths, int &lengths_off)
{
        int arg_size = *(lengths+lengths_off);
        char *arg = new char[arg_size];
        for(int i=0; i<arg_size; i++)
        {
                arg[i] = bytes[i+bytes_off];
        }

        bytes_off += arg_size;
        lengths_off += 1;

        return arg;
}


SyscallReturn
gpgpucallFunc(SyscallDesc *desc, int num, LiveProcess *process,
                                ThreadContext *tc)
{

        int index = 0;
        uint64_t functionNum = process->getSyscallArg(tc, index);
    if (functionNum > 80) {
        warn("Ignoring syscall set_robust_list(%d,...)\n", functionNum);
        return SyscallReturn(0, true);
    }

        DPRINTF(GPGPUSyscalls, "*******************************\n");
        DPRINTF(GPGPUSyscalls, "Calling gpgpu func num %d\n", functionNum);
        DPRINTF(GPGPUSyscalls, "*******************************\n");

        uint64_t ret = gpgpu_funcs[functionNum](process, tc);

        return SyscallReturn(ret, true);
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

static int load_static_globals( symbol_table *symtab, unsigned min_gaddr, unsigned max_gaddr, gpgpu_t *gpu )
{
   printf( "GPGPU-Sim PTX: loading globals with explicit initializers... \n" );
   fflush(stdout);
   int ng_bytes=0;
   symbol_table::iterator g=symtab->global_iterator_begin();

   for ( ; g!=symtab->global_iterator_end(); g++) {
      symbol *global = *g;
      if ( global->has_initializer() ) {
         printf( "GPGPU-Sim PTX:     initializing '%s' ... ", global->name().c_str() );
         unsigned addr=global->get_address();
         const type_info *type = global->type();
         type_info_key ti=type->get_key();
         size_t size;
         int t;
         ti.type_decode(size,t);
         int nbytes = size/8;
         int offset=0;
         std::list<operand_info> init_list = global->get_initializer();
         for ( std::list<operand_info>::iterator i=init_list.begin(); i!=init_list.end(); i++ ) {
            operand_info op = *i;
            ptx_reg_t value = op.get_literal_value();
            assert( (addr+offset+nbytes) < min_gaddr ); // min_gaddr is start of "heap" for cudaMalloc
            gpu->get_global_memory()->write(addr+offset,nbytes,&value,NULL,NULL); // assuming little endian here
            offset+=nbytes;
            ng_bytes+=nbytes;
         }
         printf(" wrote %u bytes\n", offset );
      }
   }
   printf( "GPGPU-Sim PTX: finished loading globals (%u bytes total).\n", ng_bytes );
   fflush(stdout);
   return ng_bytes;
}

static int load_constants( symbol_table *symtab, addr_t min_gaddr, gpgpu_t *gpu )
{
   printf( "GPGPU-Sim PTX: loading constants with explicit initializers... " );
   fflush(stdout);
   int nc_bytes = 0;
   symbol_table::iterator g=symtab->const_iterator_begin();

   for ( ; g!=symtab->const_iterator_end(); g++) {
      symbol *constant = *g;
      if ( constant->is_const() && constant->has_initializer() ) {

         // get the constant element data size
         int basic_type;
         size_t num_bits;
         constant->type()->get_key().type_decode(num_bits,basic_type);

         std::list<operand_info> init_list = constant->get_initializer();
         int nbytes_written = 0;
         for ( std::list<operand_info>::iterator i=init_list.begin(); i!=init_list.end(); i++ ) {
            operand_info op = *i;
            ptx_reg_t value = op.get_literal_value();
            int nbytes = num_bits/8;
            switch ( op.get_type() ) {
            case int_t: assert(nbytes >= 1); break;
            case float_op_t: assert(nbytes == 4); break;
            case double_op_t: assert(nbytes >= 4); break; // account for double DEMOTING
            default:
               abort();
            }
            unsigned addr=constant->get_address() + nbytes_written;
            if(!gpu->useGem5Mem) { assert( addr+nbytes < min_gaddr ); }

            gpu->get_global_memory()->write(addr,nbytes,&value,NULL,NULL); // assume little endian (so u8 is the first byte in u32)
            nc_bytes+=nbytes;
            nbytes_written += nbytes;
         }
      }
   }
   printf( " done.\n");
   fflush(stdout);
   return nc_bytes;
}

kernel_info_t *gpgpu_cuda_ptx_sim_init_grid( const char *hostFun,
                                            gpgpu_ptx_sim_arg_list_t args,
                                            struct dim3 gridDim,
                                            struct dim3 blockDim,
                                            CUctx_st* context )
{
   function_info *entry = context->get_kernel(hostFun);
   kernel_info_t *result = new kernel_info_t(gridDim,blockDim,entry);
   if( entry == NULL ) {
       printf("GPGPU-Sim PTX: ERROR launching kernel -- no PTX implementation found\n");
       abort();
   }
   unsigned argcount=args.size();
   unsigned argn=1;
   for( gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end(); a++ ) {
      entry->add_param_data(argcount-argn,&(*a));
      argn++;
   }

   entry->finalize(result->get_param_memory());
   g_ptx_kernel_count++;
   fflush(stdout);

   return result;
}
