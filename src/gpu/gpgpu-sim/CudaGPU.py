# Copyright (c) 2011 Mark D. Hill and David A. Wood
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from ClockedObject import ClockedObject
from ShaderMMU import ShaderMMU
from m5.defines import buildEnv
from m5.params import *
from m5.proxy import *

class GPGPUSimComponentWrapper(ClockedObject):
    type = 'GPGPUSimComponentWrapper'
    cxx_class = 'GPGPUSimComponentWrapper'
    cxx_header = "gpu/gpgpu-sim/cuda_gpu.hh"

class CudaGPU(ClockedObject):
    type = 'CudaGPU'
    cxx_class = 'CudaGPU'
    cxx_header = "gpu/gpgpu-sim/cuda_gpu.hh"

    sys = Param.System(Parent.any, "system sp will run on")
    shared_mem_delay = Param.Int(1, "Delay to access shared memory in gpgpu-sim ticks")
    kernel_launch_delay = Param.Float(0.00000025, "Kernel launch delay in seconds")
    kernel_return_delay = Param.Float(0.0000001, "Kernel return delay in seconds")

    warp_size = Param.Int(32, "Number of threads in each warp. Same as cores/SM")

    ruby = Param.RubySystem(Parent.any, "ruby system")

    stats_filename = Param.String("gpu_stats.txt",
          "file to which gpgpu-sim dumps its stats")
    config_path = Param.String('gpgpusim.config', "File from which to configure GPGPU-Sim")
    dump_kernel_stats = Param.Bool(False, "Dump and reset simulator statistics at the beginning and end of kernels")

    # When using a segmented physical address space, the SPA can manage memory
    manage_gpu_memory = Param.Bool(False, "Handle all GPU memory allocations in this SPA")
    access_host_pagetable = Param.Bool(False, \
                "Whether to allow accesses to host page table")
    gpu_memory_range = Param.AddrRange(AddrRange('1kB'), "The address range for the GPU memory space")

    shader_mmu = Param.ShaderMMU(ShaderMMU(), "Memory managment unit for this GPU")

    # Wrapper class to clock the GPGPU-Sim side shader cores and interconnect
    # Must be specified or gem5-gpu will error during initialization
    cores_wrapper = Param.GPGPUSimComponentWrapper("Must define a wrapper to clock the GPGPU-Sim cores")
    icnt_wrapper = Param.GPGPUSimComponentWrapper("Must define a wrapper to clock the GPGPU-Sim interconnect")

    # TODO: Eventually, we want to remove the need for the GPGPU-Sim L2 cache
    # and DRAM. Currently, these are necessary to handle parameter memory
    # accesses.
    # Wrapper class to clock the GPGPU-Sim side L2 cache and DRAM
    # Must be specified or gem5-gpu will error during initialization
    l2_wrapper = Param.GPGPUSimComponentWrapper("Must define a wrapper to clock the GPGPU-Sim L2 cache")
    dram_wrapper = Param.GPGPUSimComponentWrapper("Must define a wrapper to clock the GPGPU-Sim DRAM")
