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

#include "gpu_syscall_helper.hh"
#include "mem/ruby/system/System.hh"
#include "mem/fs_translating_port_proxy.hh"
#include "mem/se_translating_port_proxy.hh"
#include "sim/full_system.hh"

GPUSyscallHelper::GPUSyscallHelper(ThreadContext *_tc, gpusyscall_t* _call_params)
    : tc(_tc), sim_params_ptr((Addr)_call_params), arg_lengths(NULL),
      args(NULL), total_bytes(0)
{
    decode_package();
}

GPUSyscallHelper::GPUSyscallHelper(ThreadContext *_tc)
    : tc(_tc), sim_params_ptr(0), arg_lengths(NULL),
      args(NULL), total_bytes(0)
{

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
GPUSyscallHelper::readString(Addr addr, uint8_t* p, int size, ThreadContext *tc)
{
    // Ensure that the memory buffer is cleared
    memset(p, 0, size);

    // For each line in the read, grab the system's memory and check for
    // null-terminating character
    bool null_not_found = true;
    Addr curr_addr;
    int read_size;
    unsigned block_size = RubySystem::getBlockSizeBytes();
    int bytes_read = 0;
    for (; bytes_read < size && null_not_found; bytes_read += read_size) {
        curr_addr = addr + bytes_read;
        read_size = block_size;
        if (bytes_read == 0) read_size -= curr_addr % block_size;
        if (bytes_read + read_size >= size) read_size = size - bytes_read;
        readBlob(curr_addr, &p[bytes_read], read_size, tc);
        for (int index = 0; index < read_size; ++index) {
            if (p[bytes_read + index] == 0) null_not_found = false;
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

void*
GPUSyscallHelper::getParam(int index)
{
    int arg_index = 0;
    for (int i = 0; i < index; i++) {
        arg_index += arg_lengths[i];
    }
    return (void*)&args[arg_index];
}

void
GPUSyscallHelper::setReturn(unsigned char* retValue, size_t size)
{
    writeBlob((uint64_t)sim_params.ret, retValue, size);
}
