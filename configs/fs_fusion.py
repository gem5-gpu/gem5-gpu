# Copyright (c) 2009-2012 Advanced Micro Devices, Inc.
# Copyright (c) 2012-2015 Mark D. Hill and David A. Wood
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
import GPUMemConfig
import Options
import Ruby
import Simulation

from FSConfig import *
from SysPaths import *
from Benchmarks import *
# from Caches import *

def cmd_line_template():
    if options.command_line and options.command_line_file:
        print "Error: --command-line and --command-line-file are " \
              "mutually exclusive"
        sys.exit(1)
    if options.command_line:
        return options.command_line
    if options.command_line_file:
        return open(options.command_line_file).read().strip()
    return None

parser = optparse.OptionParser()
GPUConfig.addGPUOptions(parser)
GPUMemConfig.addMemCtrlOptions(parser)
Options.addCommonOptions(parser)
Options.addFSOptions(parser)

#
# Add the ruby specific and protocol specific options
#
Ruby.define_options(parser)

(options, args) = parser.parse_args()

options.ruby = True

if args:
    print "Error: script doesn't take any positional arguments"
    sys.exit(1)

if buildEnv['TARGET_ISA'] == "arm":
    fatal("gem5-gpu full system mode does not yet work for ARM!")

if buildEnv['TARGET_ISA'] != "x86":
    fatal("gem5-gpu doesn't currently work with non-x86 system!")

#
# CPU type configuration
#
if options.cpu_type != "timing" and options.cpu_type != "TimingSimpleCPU" \
    and options.cpu_type != "detailed" and options.cpu_type != "DerivO3CPU":
    print "Warning: gem5-gpu only known to work with timing and detailed CPUs: Proceed at your own risk!"
(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(options)

# If fast-forwarding, set the fast-forward CPU and mem mode for
# timing rather than atomic
if options.fast_forward:
    assert(CPUClass == AtomicSimpleCPU)
    assert(test_mem_mode == "atomic")
    CPUClass, test_mem_mode = Simulation.getCPUClass("TimingSimpleCPU")

#
# Memory space configuration
#
(cpu_mem_range, gpu_mem_range, total_mem_range) = GPUConfig.configureMemorySpaces(options)

#
# Setup benchmark to be run
#
bm = [SysConfig(disk=options.disk_image)]
bm[0].memsize = '%dB' % cpu_mem_range.size()

# Hard code the cache block width to 128B for now
# TODO: Remove this if/when block size can be different than 128B
if options.cacheline_size != 128:
    print "Warning: Only block size currently supported is 128B. Defaulting to 128."
    options.cacheline_size = 128

#
# Instantiate system
#
system = makeLinuxX86System(test_mem_mode, options.num_cpus, bm[0], True,
                            cmdline=cmd_line_template())
system.cache_line_size = options.cacheline_size
system.voltage_domain = VoltageDomain(voltage = options.sys_voltage)
system.clk_domain = SrcClockDomain(clock = options.sys_clock,
                               voltage_domain = system.voltage_domain)
system.cpu_voltage_domain = VoltageDomain()
system.cpu_clk_domain = SrcClockDomain(clock = options.cpu_clock,
                                voltage_domain = system.cpu_voltage_domain)
system.cpu = [CPUClass(cpu_id = i, clk_domain = system.cpu_clk_domain)
              for i in xrange(options.num_cpus)]

Simulation.setWorkCountOptions(system, options)

if options.kernel is not None:
    system.kernel = binary(options.kernel)

if options.script is not None:
    system.readfile = options.script

#
# Create the GPU
#
system.gpu = GPUConfig.createGPU(options, gpu_mem_range)

#
# Setup Ruby
#
system.ruby_clk_domain = SrcClockDomain(clock = options.ruby_clock,
                                        voltage_domain = system.voltage_domain)
Ruby.create_system(options, True, system, system.iobus, system._dma_ports)

system.gpu.ruby = system.ruby
system.ruby.clk_domain = system.ruby_clk_domain

# connect the PIO bus
system.iobus.master = system.ruby._io_port.slave

if options.split:
    if options.access_backing_store:
        #
        # Reset Ruby's phys_mem to add the device memory range
        #
        system.ruby.phys_mem = SimpleMemory(range=total_mem_range,
                                            in_addr_map=False)

#
# Connect CPU ports
#
for (i, cpu) in enumerate(system.cpu):
    ruby_port = system.ruby._cpu_ports[i]

    cpu.clk_domain = system.cpu_clk_domain
    cpu.createThreads()
    cpu.createInterruptController()
    cpu.interrupts.pio = ruby_port.master
    cpu.interrupts.int_master = ruby_port.slave
    cpu.interrupts.int_slave = ruby_port.master
    #
    # Tie the cpu ports to the correct ruby system ports
    #
    cpu.icache_port = system.ruby._cpu_ports[i].slave
    cpu.dcache_port = system.ruby._cpu_ports[i].slave
    if buildEnv['TARGET_ISA'] == "x86":
        cpu.itb.walker.port = system.ruby._cpu_ports[i].slave
        cpu.dtb.walker.port = system.ruby._cpu_ports[i].slave
    else:
        fatal("Not sure how to connect TLB walker ports in non-x86 system!")

#
# Connect GPU ports
#
GPUConfig.connectGPUPorts(system.gpu, system.ruby, options)

if options.mem_type == "RubyMemoryControl":
    GPUMemConfig.setMemoryControlOptions(system, options)

#
# Finalize setup and run
#

root = Root(full_system = True, system = system)

m5.disableAllListeners()

Simulation.run(options, root, system, FutureClass)
