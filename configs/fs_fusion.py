# Copyright (c) 2009-2011 Advanced Micro Devices, Inc.
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
# Authors: Brad Beckmann

#
# Full system configuraiton for ruby
#

import optparse
import os
import sys
from os.path import join as joinpath

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import addToPath, fatal

addToPath('../../gem5/configs/common')
addToPath('../../gem5/configs/ruby')
addToPath('../../gem5/configs/topologies')
addToPath('gpu_protocol')

import GPUOptions
import Options
import Ruby

import GPUConfig

from FSConfig import *
from SysPaths import *
from Benchmarks import *
import Simulation
from Caches import *

# Get paths we might need.  It's expected this file is in m5/configs/example.
config_path = os.path.dirname(os.path.abspath(__file__))
config_root = os.path.join(config_path,"../../configs")

parser = optparse.OptionParser()
Options.addCommonOptions(parser)
Options.addFSOptions(parser)
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

bm = [SysConfig()]

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

# Create the GPU.
# NOTE: If using split memory, create GPU sets up cpu_mem_range
gpu, cpu_mem_range, gpu_addr_range = GPUConfig.createGPU(options)

if buildEnv['TARGET_ISA'] == "x86":
    bm[0].memsize = cpu_mem_range.size()
    system = makeLinuxX86System(test_mem_mode, options.num_cpus, bm[0], True)
    Simulation.setWorkCountOptions(system, options)
else:
    fatal("Incapable of building non-x86 full system!")

if options.kernel is not None:
    system.kernel = binary(options.kernel)

if options.script is not None:
    system.readfile = options.script

system.cpu = [CPUClass(cpu_id=i) for i in xrange(options.num_cpus)]

system.gpu = gpu

if options.split:
    system.gpu_physmem = SimpleMemory(range=AddrRange(gpu_addr_range))

# Hard code the cache block width to at least 128B for now
# TODO: Remove this if/when block size can be less than 128B
if options.cacheline_size < 128:
    options.cacheline_size = 128
Ruby.create_system(options, system, system.piobus, system._dma_ports)

system.gpu.ruby = system.ruby

GPUConfig.connectGPUPorts(system.gpu, system.ruby, options)

for (i, cpu) in enumerate(system.cpu):
    cpu.createInterruptController()
    cpu.interrupts.pio = system.piobus.master
    cpu.interrupts.int_master = system.piobus.slave
    cpu.interrupts.int_slave = system.piobus.master
    #
    # Tie the cpu ports to the correct ruby system ports
    #
    cpu.icache_port = system.ruby._cpu_ruby_ports[i].slave
    cpu.dcache_port = system.ruby._cpu_ruby_ports[i].slave
    cpu.itb.walker.port = system.ruby._cpu_ruby_ports[i].slave
    cpu.dtb.walker.port = system.ruby._cpu_ruby_ports[i].slave

system.fusion_profiler = FusionProfiler(ruby_system = system.ruby, num_sc = options.num_sc)

GPUOptions.setMemoryControlOptions(system, options)

root = Root(full_system = True, system = system)

m5.disableAllListeners()

Simulation.run(options, root, system, FutureClass)
