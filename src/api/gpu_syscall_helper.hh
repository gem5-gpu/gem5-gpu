/*
 * Copyright (c) 2012 Mark D. Hill and David A. Wood
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

#ifndef __GPU_SYSCALL_HELPER_HH__
#define __GPU_SYSCALL_HELPER_HH__

#include "arch/isa.hh"
#include "base/types.hh"
#include "cpu/thread_context.hh"

#if THE_ISA == ARM_ISA
    // Currently supports 32-bit ARM platform
    #define __POINTER_SIZE__ 4
    #define __POINTER_MASK__ 0xFFFFFFFF

    // Helper functions for unpacking data on 32-bit targets. NOTE: This code
    // assumes that the data is laid out little-endian, as on x86 hosts.
    template <typename T>
    T unpackData(uint8_t *package, unsigned index) {
        return (T) *((T*)&package[index]);
    }

    template <typename T>
    T unpackPointer(uint8_t *package, unsigned index) {
        return (T) ((Addr)unpackData<T>(package, index) & __POINTER_MASK__);
    }

#elif THE_ISA == X86_ISA
    // Currently supports 64-bit x86 platform
    #define __POINTER_SIZE__ 8
    #define __POINTER_MASK__ 0xFFFFFFFFFFFFFFFF
#else
    #error Currently gem5-gpu is only known to support x86 and ARM
#endif

typedef struct gpucall {
    int total_bytes;
    int num_args;
    Addr arg_lengths;
    Addr args;
    Addr ret;
} gpusyscall_t;

class GPUSyscallHelper {
    ThreadContext* tc;
    Addr sim_params_ptr;
    gpusyscall_t sim_params;
    int* arg_lengths;
    unsigned char* args;
    int total_bytes;

    // Without being too invasive in the CUDA syscalls code, temporarily hold
    // the current unpackaged parameter for the calling function to grab in
    // this variable. TODO: Eventually convert the getParam() function to a
    // template function that handles casting appropriately for the CUDA
    // syscalls functions to avoid messy casting and dereferencing.
    unsigned char* live_param;

    void decode_package();
    void readBlob(Addr addr, uint8_t* p, int size, ThreadContext *tc);
    void readString(Addr addr, uint8_t* p, int size, ThreadContext *tc);
    void writeBlob(Addr addr, uint8_t* p, int size,
                   ThreadContext *tc, bool is_ptr);
  public:
    GPUSyscallHelper(ThreadContext* _tc, gpusyscall_t* _call_params = NULL);
    ~GPUSyscallHelper();
    void* getParam(int index, bool is_ptr = false);
    void setReturn(unsigned char* retValue, size_t size, bool is_ptr = false);
    ThreadContext* getThreadContext() { return tc; }
    void readBlob(Addr addr, uint8_t* p, int size) {
        readBlob(addr, p, size, tc);
    }
    void readString(Addr addr, uint8_t* p, int size) {
        readString(addr, p, size, tc);
    }
    void writeBlob(Addr addr, uint8_t* p, int size, bool is_ptr = false) {
        writeBlob(addr, p, size, tc, is_ptr);
    }
};

#endif
