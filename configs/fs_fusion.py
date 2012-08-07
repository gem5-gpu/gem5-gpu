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
Options.addMemCtrlOptions(parser)
GPUOptions.addGPUOptions(parser)

# Benchmark options
parser.add_option("-o", "--options", default="",
    help='The options to pass to the binary, use " " around the entire string')
parser.add_option("-i", "--input", default="", help="Read stdin from a file.")
parser.add_option("--output", default="", help="Redirect stdout to a file.")
parser.add_option("--errout", default="", help="Redirect stderr to a file.")

#
# Add the ruby specific and protocol specific options
#
Ruby.define_options(parser)

(options, args) = parser.parse_args()
options.ruby = True

if args:
    print "Error: script doesn't take any positional arguments"
    sys.exit(1)

# Clean GPGPU-Sim params
options.sc_l1_assoc = options.l1d_assoc

if options.benchmark:
    try:
        bm = Benchmarks[options.benchmark]
    except KeyError:
        print "Error benchmark %s has not been defined." % options.benchmark
        print "Valid benchmarks are: %s" % DefinedBenchmarks
        sys.exit(1)
else:
    bm = [SysConfig()]

# Check for timing mode because ruby does not support atomic accesses
if not (options.cpu_type == "detailed" or options.cpu_type == "timing"):
    print >> sys.stderr, "Ruby requires TimingSimpleCPU or O3CPU!!"
    sys.exit(1)
(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(options)

CPUClass.clock = options.clock

if buildEnv['TARGET_ISA'] == "alpha":
    system = makeLinuxAlphaRubySystem(test_mem_mode, bm[0])
elif buildEnv['TARGET_ISA'] == "x86":
    system = makeLinuxX86System(test_mem_mode, options.num_cpus, bm[0], True)
    Simulation.setWorkCountOptions(system, options)
else:
    fatal("incapable of building non-alpha or non-x86 full system!")

if options.kernel is not None:
    system.kernel = binary(options.kernel)

if options.script is not None:
    system.readfile = options.script

gpgpusimconfig = GPUOptions.parseGpgpusimConfig(options)

if options.baseline:
        print "Using options based on baseline!"
        print "Remember any command line options may be ignored"
        options.clock = "3GHz"
        options.sc_l1_size = "64kB"

if options.fermi:
    print "Using options based on fermi!"
    print "Remember any command line options may be ignored"
    options.topology = "Crossbar"
    options.clock = "2.6GHz"
    options.cacheline_size = 128
    options.sc_l1_size = "16kB"
    options.sc_l1_assoc = 64
    options.sc_l2_size = "128kB"
    options.sc_l2_assoc = 64
    options.num_dirs = 8
    options.shMemDelay = 30

    #CPU things
    options.l1d_size = "64kB"
    options.l1i_size = "32kB"
    options.l2_size = "256kB"
    options.l1i_assoc = 4
    options.l1d_assoc = 8
    options.l2_assoc = 16

system.cpu = [CPUClass(cpu_id=i) for i in xrange(options.num_cpus)]

system.stream_proc_array = StreamProcessorArray()
system.stream_proc_array.shader_cores = [ShaderCore(id=i) for i in xrange(options.num_sc)]
system.stream_proc_array.ce = SPACopyEngine(driver_delay=5000000)
system.stream_proc_array.shared_mem_delay = options.shMemDelay
system.stream_proc_array.config_path = gpgpusimconfig
system.stream_proc_array.dump_kernel_stats = options.kernel_stats
buildEnv['PROTOCOL'] +=  '_fusion'
Ruby.create_system(options, system, system.piobus, system._dma_ports)
system.stream_proc_array.ruby = system.ruby
system.ruby.block_size_bytes = 128

if options.fermi:
    system.ruby.clock = "2.6GHz" # NOTE: This is the memory clock

for i in xrange(options.num_sc):
    system.stream_proc_array.shader_cores[i].data_port = system.ruby._cpu_ruby_ports[options.num_cpus+i].slave
    system.stream_proc_array.shader_cores[i].inst_port = system.ruby._cpu_ruby_ports[options.num_cpus+i].slave
    if buildEnv['TARGET_ISA'] == "x86":
        system.stream_proc_array.shader_cores[i].host_dtb.walker.port = system.ruby._cpu_ruby_ports[options.num_cpus+i].slave
        system.stream_proc_array.shader_cores[i].device_dtb.walker.port = system.ruby._cpu_ruby_ports[options.num_cpus+i].slave

for (i, cpu) in enumerate(system.cpu):
    #
    # Tie the cpu ports to the correct ruby system ports
    #
    cpu.createInterruptController()
    cpu.icache_port = system.ruby._cpu_ruby_ports[i].slave
    cpu.dcache_port = system.ruby._cpu_ruby_ports[i].slave
    if buildEnv['TARGET_ISA'] == "x86":
        cpu.itb.walker.port = system.ruby._cpu_ruby_ports[i].slave
        cpu.dtb.walker.port = system.ruby._cpu_ruby_ports[i].slave
        cpu.interrupts.pio = system.piobus.master
        cpu.interrupts.int_master = system.piobus.slave
        cpu.interrupts.int_slave = system.piobus.master

if options.baseline:
        # need to do this after ruby created
        for i in xrange(options.num_dirs):
                exec("system.dir_cntrl%d.memBuffer.mem_bus_cycle_multiplier = 5" % i)

if options.fermi:
   for i in xrange(options.num_dirs):
      exec("system.dir_cntrl%d.memBuffer.mem_bus_cycle_multiplier = 1" % i)
      exec("system.dir_cntrl%d.memBuffer.mem_ctl_latency = 1" % i)

#Tie the copy engine port to its cache
system.stream_proc_array.ce.host_port = system.ruby._cpu_ruby_ports[options.num_cpus+options.num_sc].slave
system.stream_proc_array.ce.device_port = system.ruby._cpu_ruby_ports[options.num_cpus+options.num_sc].slave
if buildEnv['TARGET_ISA'] == "x86":
    system.stream_proc_array.ce.host_dtb.walker.port = system.ruby._cpu_ruby_ports[options.num_cpus+options.num_sc].slave
    system.stream_proc_array.ce.device_dtb.walker.port = system.ruby._cpu_ruby_ports[options.num_cpus+options.num_sc].slave

root = Root(full_system = True, system = system)

m5.disableAllListeners()

Simulation.run(options, root, system, FutureClass)
