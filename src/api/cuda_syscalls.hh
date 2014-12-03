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

#ifndef __SIM_GPGPU_SYSCALLS_HH__
#define __SIM_GPGPU_SYSCALLS_HH__

#include "api/gpu_syscall_helper.hh"
#include "sim/syscall_emul.hh"

/*******************************
       CUDA API MEMBERS
********************************/

enum cudaError
{
    cudaSuccess                           =      0,   // No errors
    cudaErrorMissingConfiguration         =      1,   // Missing configuration error
    cudaErrorMemoryAllocation             =      2,   // Memory allocation error
    cudaErrorInitializationError          =      3,   // Initialization error
    cudaErrorLaunchFailure                =      4,   // Launch failure
    cudaErrorPriorLaunchFailure           =      5,   // Prior launch failure
    cudaErrorLaunchTimeout                =      6,   // Launch timeout error
    cudaErrorLaunchOutOfResources         =      7,   // Launch out of resources error
    cudaErrorInvalidDeviceFunction        =      8,   // Invalid device function
    cudaErrorInvalidConfiguration         =      9,   // Invalid configuration
    cudaErrorInvalidDevice                =     10,   // Invalid device
    cudaErrorInvalidValue                 =     11,   // Invalid value
    cudaErrorInvalidPitchValue            =     12,   // Invalid pitch value
    cudaErrorInvalidSymbol                =     13,   // Invalid symbol
    cudaErrorMapBufferObjectFailed        =     14,   // Map buffer object failed
    cudaErrorUnmapBufferObjectFailed      =     15,   // Unmap buffer object failed
    cudaErrorInvalidHostPointer           =     16,   // Invalid host pointer
    cudaErrorInvalidDevicePointer         =     17,   // Invalid device pointer
    cudaErrorInvalidTexture               =     18,   // Invalid texture
    cudaErrorInvalidTextureBinding        =     19,   // Invalid texture binding
    cudaErrorInvalidChannelDescriptor     =     20,   // Invalid channel descriptor
    cudaErrorInvalidMemcpyDirection       =     21,   // Invalid memcpy direction
    cudaErrorAddressOfConstant            =     22,   // Address of constant error
                                                      // \deprecated
                                                      // This error return is deprecated as of
                                                      // Cuda 3.1. Variables in constant memory
                                                      // may now have their address taken by the
                                                      // runtime via ::cudaGetSymbolAddress().
    cudaErrorTextureFetchFailed           =     23,   // Texture fetch failed
    cudaErrorTextureNotBound              =     24,   // Texture not bound error
    cudaErrorSynchronizationError         =     25,   // Synchronization error
    cudaErrorInvalidFilterSetting         =     26,   // Invalid filter setting
    cudaErrorInvalidNormSetting           =     27,   // Invalid norm setting
    cudaErrorMixedDeviceExecution         =     28,   // Mixed device execution
    cudaErrorCudartUnloading              =     29,   // CUDA runtime unloading
    cudaErrorUnknown                      =     30,   // Unknown error condition
    cudaErrorNotYetImplemented            =     31,   // Function not yet implemented
    cudaErrorMemoryValueTooLarge          =     32,   // Memory value too large
    cudaErrorInvalidResourceHandle        =     33,   // Invalid resource handle
    cudaErrorNotReady                     =     34,   // Not ready error
    cudaErrorInsufficientDriver           =     35,   // CUDA runtime is newer than driver
    cudaErrorSetOnActiveProcess           =     36,   // Set on active process error
    cudaErrorInvalidSurface               =     37,   // Invalid surface
    cudaErrorNoDevice                     =     38,   // No Cuda-capable devices detected
    cudaErrorECCUncorrectable             =     39,   // Uncorrectable ECC error detected
    cudaErrorSharedObjectSymbolNotFound   =     40,   // Link to a shared object failed to resolve
    cudaErrorSharedObjectInitFailed       =     41,   // Shared object initialization failed
    cudaErrorUnsupportedLimit             =     42,   // ::cudaLimit not supported by device
    cudaErrorDuplicateVariableName        =     43,   // Duplicate global variable lookup by string name
    cudaErrorDuplicateTextureName         =     44,   // Duplicate texture lookup by string name
    cudaErrorDuplicateSurfaceName         =     45,   // Duplicate surface lookup by string name
    cudaErrorDevicesUnavailable           =     46,   // All Cuda-capable devices are busy (see ::cudaComputeMode) or unavailable
    cudaErrorStartupFailure               =   0x7f,   // Startup failure
    cudaErrorApiFailureBase               =  10000    // API failure base
};

typedef enum cudaError cudaError_t;

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      // Host   -> Host
    cudaMemcpyHostToDevice        =   1,      // Host   -> Device
    cudaMemcpyDeviceToHost        =   2,      // Device -> Host
    cudaMemcpyDeviceToDevice      =   3       // Device -> Device
};

const char* cudaMemcpyKindStrings[] =
{
    "cudaMemcpyHostToHost",
    "cudaMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice"
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

typedef struct cudaFuncAttributes {
   size_t sharedSizeBytes;
   size_t constSizeBytes;
   size_t localSizeBytes;
   int maxThreadsPerBlock;
   int numRegs;
   int ptxVersion;
   int binaryVersion;
   int __cudaReserved[6];
} cudaFuncAttributes;

/*******************************
     CUDA API GEM5 HANDLERS
********************************/

void cudaMalloc(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMallocHost(ThreadContext *tc, gpusyscall_t *call_params);
void cudaRegisterDeviceMemory(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMallocPitch(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMallocArray(ThreadContext *tc, gpusyscall_t *call_params);
void cudaFree(ThreadContext *tc, gpusyscall_t *call_params);
void cudaFreeHost(ThreadContext *tc, gpusyscall_t *call_params);
void cudaFreeArray(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpy(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpyToArray(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpyFromArray(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpyArrayToArray(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpy2D(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpy2DToArray(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpy2DFromArray(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpy2DArrayToArray(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpyToSymbol(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpyFromSymbol(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpyAsync(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpyToArrayAsync(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpyFromArrayAsync(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpy2DAsync(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpy2DToArrayAsync(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemcpy2DFromArrayAsync(ThreadContext *tc, gpusyscall_t *call_params);
void cudaBlockThread(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemset(ThreadContext *tc, gpusyscall_t *call_params);
void cudaMemset2D(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetSymbolAddress(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetSymbolSize(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetDeviceCount(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetDeviceProperties(ThreadContext *tc, gpusyscall_t *call_params);
void cudaChooseDevice(ThreadContext *tc, gpusyscall_t *call_params);
void cudaSetDevice(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetDevice(ThreadContext *tc, gpusyscall_t *call_params);
void cudaBindTexture(ThreadContext *tc, gpusyscall_t *call_params);
void cudaBindTextureToArray(ThreadContext *tc, gpusyscall_t *call_params);
void cudaUnbindTexture(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetTextureAlignmentOffset(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetTextureReference(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetChannelDesc(ThreadContext *tc, gpusyscall_t *call_params);
void cudaCreateChannelDesc(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetLastError(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGetErrorString(ThreadContext *tc, gpusyscall_t *call_params);
void cudaConfigureCall(ThreadContext *tc, gpusyscall_t *call_params);
void cudaSetupArgument(ThreadContext *tc, gpusyscall_t *call_params);
void cudaLaunch(ThreadContext *tc, gpusyscall_t *call_params);
void cudaStreamCreate(ThreadContext *tc, gpusyscall_t *call_params);
void cudaStreamDestroy(ThreadContext *tc, gpusyscall_t *call_params);
void cudaStreamSynchronize(ThreadContext *tc, gpusyscall_t *call_params);
void cudaStreamQuery(ThreadContext *tc, gpusyscall_t *call_params);
void cudaEventCreate(ThreadContext *tc, gpusyscall_t *call_params);
void cudaEventRecord(ThreadContext *tc, gpusyscall_t *call_params);
void cudaEventQuery(ThreadContext *tc, gpusyscall_t *call_params);
void cudaEventSynchronize(ThreadContext *tc, gpusyscall_t *call_params);
void cudaEventDestroy(ThreadContext *tc, gpusyscall_t *call_params);
void cudaEventElapsedTime(ThreadContext *tc, gpusyscall_t *call_params);
void cudaThreadExit(ThreadContext *tc, gpusyscall_t *call_params);
void cudaThreadSynchronize(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaSynchronizeThreads(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaRegisterFatBinary(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaRegisterFatBinaryFinalize(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaCheckAllocateLocal(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaSetLocalAllocation(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaUnregisterFatBinary(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaRegisterFunction(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaRegisterVar(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaRegisterShared(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaRegisterSharedVar(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaRegisterTexture(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGLRegisterBufferObject(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGLMapBufferObject(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGLUnmapBufferObject(ThreadContext *tc, gpusyscall_t *call_params);
void cudaGLUnregisterBufferObject(ThreadContext *tc, gpusyscall_t *call_params);

#if (CUDART_VERSION >= 2010)

void cudaHostAlloc(ThreadContext *tc, gpusyscall_t *call_params);
void cudaHostGetDevicePointer(ThreadContext *tc, gpusyscall_t *call_params);
void cudaSetValidDevices(ThreadContext *tc, gpusyscall_t *call_params);
void cudaSetDeviceFlags(ThreadContext *tc, gpusyscall_t *call_params);
void cudaFuncGetAttributes(ThreadContext *tc, gpusyscall_t *call_params);
void cudaEventCreateWithFlags(ThreadContext *tc, gpusyscall_t *call_params);
void cudaDriverGetVersion(ThreadContext *tc, gpusyscall_t *call_params);
void cudaRuntimeGetVersion(ThreadContext *tc, gpusyscall_t *call_params);

#endif

void cudaGLSetGLDevice(ThreadContext *tc, gpusyscall_t *call_params);
void cudaWGLGetDevice(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaMutexOperation(ThreadContext *tc, gpusyscall_t *call_params);
void __cudaTextureFetch(ThreadContext *tc, gpusyscall_t *call_params);

namespace cuda_math {
    uint64_t __cudaMutexOperation(ThreadContext *tc, gpusyscall_t *call_params);
    uint64_t __cudaTextureFetch(ThreadContext *tc, gpusyscall_t *call_params);
    uint64_t __cudaSynchronizeThreads(ThreadContext *tc, gpusyscall_t *call_params);
    void  __cudaTextureFetch(const void *tex, void *index, int integer, void *val);
    void __cudaMutexOperation(int lock);
}

typedef void (*cudaFunc_t)(ThreadContext *, gpusyscall_t *);

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
        __cudaSynchronizeThreads,        /* 80 */
        __cudaRegisterFatBinaryFinalize,    /* 81 */
        cudaRegisterDeviceMemory,    /* 82 */
        cudaBlockThread,    /* 83 */
        __cudaCheckAllocateLocal,    /* 84 */
        __cudaSetLocalAllocation    /* 85 */
};

#endif


