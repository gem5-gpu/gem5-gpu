# Copyright (c) 2006-2008 The Regents of The University of Michigan
# Copyright (c) 2012-2013 Mark D. Hill and David A. Wood
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
# Authors: Jason Power, Joel Hestness

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

import GPUConfig
import MemConfig
import Options
import Ruby
import Simulation

parser = optparse.OptionParser()
GPUConfig.addGPUOptions(parser)
MemConfig.addMemCtrlOptions(parser)
Options.addCommonOptions(parser)
Options.addSEOptions(parser)

#
# Add the ruby specific and protocol specific options
#
Ruby.define_options(parser)

(options, args) = parser.parse_args()

options.ruby = True

if args:
    print "Error: script doesn't take any positional arguments"
    sys.exit(1)

if buildEnv['TARGET_ISA'] != "x86":
    fatal("gem5-gpu doesn't currently work with non-x86 system!")

#
# CPU type configuration
#
if options.cpu_type != "timing" and options.cpu_type != "detailed":
    print "Warning: gem5-gpu only works with timing and detailed CPUs. Defaulting to timing"
    options.cpu_type = "timing"
(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(options)
CPUClass.clock = options.clock

#
# Memory space configuration
#
(cpu_mem_range, gpu_mem_range) = GPUConfig.configureMemorySpaces(options)

#
# Setup benchmark to be run
#
process = LiveProcess()
process.executable = options.cmd
process.cmd = [options.cmd] + options.options.split()

if options.input != "":
    process.input = options.input
if options.output != "":
    process.output = options.output
if options.errout != "":
    process.errout = options.errout

#
# Instantiate system
#
system = System(cpu = [CPUClass(cpu_id = i, workload = process) for i in xrange(options.num_cpus)],
                physmem = SimpleMemory(range = cpu_mem_range))
system.mem_mode = test_mem_mode
Simulation.setWorkCountOptions(system, options)

#
# Create the GPU
#
system.gpu = GPUConfig.createGPU(options, gpu_mem_range)

if options.split:
    system.gpu_physmem = SimpleMemory(range = gpu_mem_range)

# Hard code the cache block width to at least 128B for now
# TODO: Remove this if/when block size can be less than 128B
if options.cacheline_size < 128:
    print "Warning: Minimum cache block size is currently 128B. Defaulting to 128."
    options.cacheline_size = 128
Ruby.create_system(options, system)

system.gpu.ruby = system.ruby

#
# Connect CPU ports
#
for (i, cpu) in enumerate(system.cpu):
    ruby_port = system.ruby._cpu_ruby_ports[i]

    cpu.createInterruptController()
    cpu.interrupts.pio = ruby_port.master
    cpu.interrupts.int_master = ruby_port.slave
    cpu.interrupts.int_slave = ruby_port.master
    #
    # Tie the cpu ports to the correct ruby system ports
    #
    cpu.icache_port = system.ruby._cpu_ruby_ports[i].slave
    cpu.dcache_port = system.ruby._cpu_ruby_ports[i].slave
    if buildEnv['TARGET_ISA'] == "x86":
        cpu.itb.walker.port = system.ruby._cpu_ruby_ports[i].slave
        cpu.dtb.walker.port = system.ruby._cpu_ruby_ports[i].slave
    else:
        fatal("Not sure how to connect TLB walker ports in non-x86 system!")

#
# Connect GPU ports
#
GPUConfig.connectGPUPorts(system.gpu, system.ruby, options)

MemConfig.setMemoryControlOptions(system, options)

#
# Finalize setup and run
#
system.fusion_profiler = FusionProfiler(ruby_system = system.ruby, num_sc = options.num_sc)

root = Root(full_system = False, system = system)

m5.disableAllListeners()

Simulation.run(options, root, system, FutureClass)
