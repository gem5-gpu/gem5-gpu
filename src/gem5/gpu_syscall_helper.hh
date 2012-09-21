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

#include "base/types.hh"
#include "cpu/thread_context.hh"

typedef struct gpucall {
    int total_bytes;
    int num_args;
    int* arg_lengths;
    char* args;
    char* ret;
} gpusyscall_t;

class GPUSyscallHelper {
    ThreadContext* tc;
    Addr sim_params_ptr;
    gpusyscall_t sim_params;
    int* arg_lengths;
    unsigned char* args;
    int total_bytes;

    void decode_package();
    void readBlob(Addr addr, uint8_t* p, int size, ThreadContext *tc);
    void readString(Addr addr, uint8_t* p, int size, ThreadContext *tc);
    void writeBlob(Addr addr, uint8_t* p, int size, ThreadContext *tc);
  public:
    GPUSyscallHelper(ThreadContext* _tc, gpusyscall_t* _call_params);
    GPUSyscallHelper(ThreadContext* _tc);
    ~GPUSyscallHelper();
    void* getParam(int index);
    void setReturn(unsigned char* retValue, size_t size);
    ThreadContext* getThreadContext() { return tc; }
    void readBlob(Addr addr, uint8_t* p, int size) { readBlob(addr, p, size, tc); }
    void readString(Addr addr, uint8_t* p, int size) { readString(addr, p, size, tc); }
    void writeBlob(Addr addr, uint8_t* p, int size) { writeBlob(addr, p, size, tc); }
};

#endif
