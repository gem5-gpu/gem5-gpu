# Copyright (c) 2006-2008 The Regents of The University of Michigan
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
# Authors: Steve Reinhardt

# Simple test script
#
# "m5 test.py"

import os
import optparse
import sys
from os.path import join as joinpath

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import addToPath, panic

import string

if buildEnv['FULL_SYSTEM']:
    panic("This script requires syscall emulation mode (*_SE).")

addToPath('../../gem5/configs/common')
addToPath('../../gem5/configs/ruby')
addToPath('../../gem5/configs/topologies')
addToPath('gpu_protocol')

import GPUOptions
import Options
import Ruby

import Simulation
from cpu2000 import *

# Get paths we might need.  It's expected this file is in m5/configs/example.
config_path = os.path.dirname(os.path.abspath(__file__))
config_root = os.path.join(config_path,"../../configs")

parser = optparse.OptionParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)
GPUOptions.addMemCtrlOptions(parser)
GPUOptions.addGPUOptions(parser)

#
# Add the ruby specific and protocol specific options
#
Ruby.define_options(parser)

(options, args) = parser.parse_args()

if args:
    print "Error: script doesn't take any positional arguments"
    sys.exit(1)

process = LiveProcess()
process.executable = options.cmd
process.cmd = [options.cmd] + options.options.split()

if options.input != "":
    process.input = options.input
if options.output != "":
    process.output = options.output
if options.errout != "":
    process.errout = options.errout

cpu_type = options.cpu_type
if cpu_type != 'timing' and cpu_type != 'detailed':
    cpu_type = 'timing'

restore_cpu_type = options.restore_with_cpu
if restore_cpu_type != 'timing' and restore_cpu_type != 'detailed':
    restore_cpu_type = 'timing'

FutureClass = None
if options.checkpoint_restore != None:
    if restore_cpu_type != cpu_type:
        if cpu_type == 'timing':
            class FutureClass(TimingSimpleCPU): pass
            cpu_type = restore_cpu_type
            FutureClass.clock = options.clock
        elif cpu_type == 'detailed':
            class FutureClass(DerivO3CPU): pass
            cpu_type = restore_cpu_type
            FutureClass.clock = options.clock

if cpu_type == 'timing':
    class CPUClass(TimingSimpleCPU): pass
elif cpu_type == 'detailed':
    class CPUClass(DerivO3CPU): pass

test_mem_mode = 'timing'

CPUClass.clock = options.clock

#
# GPGPU-Sim configuration
#
gpgpusimconfig = GPUOptions.parseGpgpusimConfig(options)

if buildEnv['TARGET_ISA'] != "x86":
    fatal("gem5-fusion doesn't currently work with non-x86 system!")

# Variables for creating the stream processor array
gpu_segment_base_addr = Addr(0)
gpu_mem_size_bytes = 0
total_mem_range = AddrRange(options.total_mem_size)

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

system = System(cpu = [CPUClass(cpu_id=i) for i in xrange(options.num_cpus)],
                physmem = SimpleMemory(range=cpu_mem_range))
system.mem_mode = test_mem_mode
Simulation.setWorkCountOptions(system, options)

system.stream_proc_array = StreamProcessorArray(manage_gpu_memory = options.split,
            gpu_segment_base = gpu_segment_base_addr, gpu_memory_size = gpu_mem_size_bytes)
if options.split:
    system.gpu_physmem = SimpleMemory(range=AddrRange(gpu_addr_range))
system.stream_proc_array.shader_cores = [ShaderCore(id=i) for i in xrange(options.num_sc)]
system.stream_proc_array.frequency = options.gpu_core_clock
system.stream_proc_array.ce = SPACopyEngine(driver_delay=5000000)
system.stream_proc_array.warp_size = options.gpu_warp_size

for sc in system.stream_proc_array.shader_cores:
    sc.lsq = ShaderLSQ()
    sc.lsq.warp_size = options.gpu_warp_size

# This is a stop-gap solution until we implement a better way to register device memory
if options.access_host_pagetable:
    for sc in system.stream_proc_array.shader_cores:
        sc.itb.access_host_pagetable = True
        sc.dtb.access_host_pagetable = True
        sc.lsq.data_tlb.access_host_pagetable = True
    system.stream_proc_array.ce.device_dtb.access_host_pagetable = True
    system.stream_proc_array.ce.host_dtb.access_host_pagetable = True

system.stream_proc_array.shared_mem_delay = options.shMemDelay
system.stream_proc_array.config_path = gpgpusimconfig
system.stream_proc_array.dump_kernel_stats = options.kernel_stats
# Hard code the cache block width to at least 128B for now
# TODO: Remove this if/when block size can be less than 128B
if options.cacheline_size < 128:
    options.cacheline_size = 128
Ruby.create_system(options, system)
system.stream_proc_array.ruby = system.ruby

for i,sc in enumerate(system.stream_proc_array.shader_cores):
    sc.data_port = system.ruby._cpu_ruby_ports[options.num_cpus+i].slave
    sc.inst_port = system.ruby._cpu_ruby_ports[options.num_cpus+i].slave
    for j in xrange(options.gpu_warp_size):
        sc.lsq_port[j] = sc.lsq.lane_port[j]
    sc.lsq.cache_port = system.ruby._cpu_ruby_ports[options.num_cpus+i].slave

for (i, cpu) in enumerate(system.cpu):
    ruby_port = system.ruby._cpu_ruby_ports[i]

    cpu.createInterruptController()
    cpu.interrupts.pio = ruby_port.master
    cpu.interrupts.int_master = ruby_port.slave
    cpu.interrupts.int_slave = ruby_port.master
    #
    # Tie the cpu ports to the ruby cpu ports
    #
    cpu.icache_port = system.ruby._cpu_ruby_ports[i].slave
    cpu.dcache_port = system.ruby._cpu_ruby_ports[i].slave
    if buildEnv['TARGET_ISA'] == "x86":
        cpu.itb.walker.port = system.ruby._cpu_ruby_ports[i].slave
        cpu.dtb.walker.port = system.ruby._cpu_ruby_ports[i].slave

    cpu.workload = process

# Tie the copy engine ports to controllers
system.stream_proc_array.ce.host_port = system.ruby._cpu_ruby_ports[options.num_cpus+options.num_sc].slave
if options.split:
    system.stream_proc_array.ce.device_port = system.ruby._cpu_ruby_ports[options.num_cpus+options.num_sc+1].slave
else:
    # With a unified address space, tie both copy engine ports to the same
    # copy engine controller
    system.stream_proc_array.ce.device_port = system.ruby._cpu_ruby_ports[options.num_cpus+options.num_sc].slave

GPUOptions.setMemoryControlOptions(system, options)

system.fusion_profiler = FusionProfiler(ruby_system = system.ruby, num_sc = options.num_sc)

root = Root(full_system = False, system = system)

m5.disableAllListeners()

Simulation.run(options, root, system, FutureClass)
