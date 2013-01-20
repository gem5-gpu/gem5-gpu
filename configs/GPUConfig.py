# Copyright (c) 2012 Mark D. Hill and David A. Wood
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
# Authors: Jason Power

from m5.objects import *

import GPUOptions

def createGPU(options):
    gpu_segment_base_addr = Addr(0)
    gpu_mem_size_bytes = 0
    total_mem_range = AddrRange(options.total_mem_size)

    gpgpusimOptions = GPUOptions.parseGpgpusimConfig(options)

    gpu_addr_range = None

    if options.split:
        buildEnv['PROTOCOL'] +=  '_split'
        total_mem_size_bytes = long(total_mem_range.second) - long(total_mem_range.first) + 1
        gpu_addr_range = AddrRange(options.gpu_mem_size)
        gpu_mem_size_bytes = long(gpu_addr_range.second) - long(gpu_addr_range.first) + 1
        if gpu_mem_size_bytes >= total_mem_size_bytes:
            print "GPU memory size (%s) won't fit within total memory size (%s)!" % (options.gpu_mem_size, options.total_mem_size)
            sys.exit(1)
        gpu_segment_base_addr = Addr(total_mem_size_bytes - gpu_mem_size_bytes)
        gpu_addr_range = AddrRange(gpu_segment_base_addr, size = options.gpu_mem_size)
        options.total_mem_size = long(gpu_segment_base_addr)
        cpu_mem_range = AddrRange(long(gpu_segment_base_addr))
    else:
        buildEnv['PROTOCOL'] +=  '_fusion'
        cpu_mem_range = total_mem_range

    gpu = StreamProcessorArray(manage_gpu_memory = options.split,
            gpu_segment_base = gpu_segment_base_addr, gpu_memory_size = gpu_mem_size_bytes)

    gpu.shader_cores = [ShaderCore(id=i) for i in xrange(options.num_sc)]
    gpu.ce = SPACopyEngine(driver_delay=5000000)

    gpu.frequency = options.gpu_core_clock
    gpu.warp_size = options.gpu_warp_size

    for sc in gpu.shader_cores:
        sc.lsq = ShaderLSQ()
        sc.lsq.warp_size = options.gpu_warp_size

    # This is a stop-gap solution until we implement a better way to register device memory
    if options.access_host_pagetable:
        for sc in gpu.shader_cores:
            sc.itb.access_host_pagetable = True
            sc.dtb.access_host_pagetable = True
            sc.lsq.data_tlb.access_host_pagetable = True
        gpu.ce.device_dtb.access_host_pagetable = True
        gpu.ce.host_dtb.access_host_pagetable = True

    gpu.shared_mem_delay = options.shMemDelay
    gpu.config_path = gpgpusimOptions
    gpu.dump_kernel_stats = options.kernel_stats

    return gpu, cpu_mem_range, gpu_addr_range

def connectGPUPorts(gpu, ruby, options):
    for i,sc in enumerate(gpu.shader_cores):
        sc.data_port = ruby._cpu_ruby_ports[options.num_cpus+i].slave
        sc.inst_port = ruby._cpu_ruby_ports[options.num_cpus+i].slave
        for j in xrange(options.gpu_warp_size):
            sc.lsq_port[j] = sc.lsq.lane_port[j]
        sc.lsq.cache_port = ruby._cpu_ruby_ports[options.num_cpus+i].slave

    gpu.ce.host_port = ruby._cpu_ruby_ports[options.num_cpus+options.num_sc].slave
    if options.split:
        gpu.ce.device_port = ruby._cpu_ruby_ports[options.num_cpus+options.num_sc+1].slave
    else:
        # With a unified address space, tie both copy engine ports to the same
        # copy engine controller
        gpu.ce.device_port = ruby._cpu_ruby_ports[options.num_cpus+options.num_sc].slave
