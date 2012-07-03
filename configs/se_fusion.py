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
addToPath('gpu_protocol')

import GPUOptions
import Options
import Ruby

import Simulation
from cpu2000 import *

# Get paths we might need.  It's expected this file is in m5/configs/example.
config_path = os.path.dirname(os.path.abspath(__file__))
config_root = os.path.join(config_path,"../../configs")
m5_root = os.path.dirname(config_root)

parser = optparse.OptionParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)
Options.addMemCtrlOptions(parser)
GPUOptions.addGPUOptions(parser)

# Benchmark options
parser.add_option("-d", "--detailed", action="store_true", default=True)
parser.add_option("-t", "--timing", action="store_true")

#
# Add the ruby specific and protocol specific options
#
Ruby.define_options(parser)

(options, args) = parser.parse_args()

if args:
    print "Error: script doesn't take any positional arguments"
    sys.exit(1)

if options.bench:
    try:
        if buildEnv['TARGET_ISA'] != 'alpha':
            print >>sys.stderr, "Simpoints code only works for Alpha ISA at this time"
            sys.exit(1)
        exec("workload = %s('alpha', 'tru64', 'ref')" % options.bench)
        process = workload.makeLiveProcess()
    except:
        print >>sys.stderr, "Unable to find workload for %s" % options.bench
        sys.exit(1)
else:
    process = LiveProcess()
    process.executable = options.cmd
    process.cmd = [options.cmd] + options.options.split()


if options.input != "":
    process.input = options.input
if options.output != "":
    process.output = options.output
if options.errout != "":
    process.errout = options.errout

options.sc_l1_assoc = options.l1d_assoc

if options.detailed:
    #check for SMT workload
    workloads = options.cmd.split(';')
    if len(workloads) > 1:
        process = []
        smt_idx = 0
        inputs = []
        outputs = []
        errouts = []

        if options.input != "":
            inputs = options.input.split(';')
        if options.output != "":
            outputs = options.output.split(';')
        if options.errout != "":
            errouts = options.errout.split(';')

        for wrkld in workloads:
            smt_process = LiveProcess()
            smt_process.executable = wrkld
            smt_process.cmd = wrkld + " " + options.options
            if inputs and inputs[smt_idx]:
                smt_process.input = inputs[smt_idx]
            if outputs and outputs[smt_idx]:
                smt_process.output = outputs[smt_idx]
            if errouts and errouts[smt_idx]:
                smt_process.errout = errouts[smt_idx]
            process += [smt_process, ]
            smt_idx += 1

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

#
# Currently ruby only works in timing mode
#
if options.gpu_only:
    class CPUClass(AtomicSimpleCPU): pass
    test_mem_mode = 'timing'
else:
    class CPUClass(DerivO3CPU):
        LQEntries = 128
        SQEntries = 128
    test_mem_mode = 'timing'

FutureClass = None

CPUClass.clock = options.clock

np = options.num_cpus

system = System(cpu = [CPUClass(cpu_id=i) for i in xrange(np)],
                physmem = SimpleMemory(range=AddrRange("1536MB")))

system.stream_proc_array = StreamProcessorArray(gpuTickConv=options.m5_cycles_per_gpu_cycles)
system.stream_proc_array.shader_cores = [ShaderCore(id=i) for i in xrange(options.num_sc)]
system.stream_proc_array.ce = SPACopyEngine(driverDelay=5000000)
system.stream_proc_array.useGem5Mem = options.gpu_ruby
system.stream_proc_array.sharedMemDelay = options.shMemDelay
system.stream_proc_array.nonBlocking = options.gpu_nonblocking
system.stream_proc_array.config_path = gpgpusimconfig
buildEnv['PROTOCOL'] +=  '_fusion'
Ruby.create_system(options, system)
system.stream_proc_array.ruby = system.ruby
system.ruby.block_size_bytes = 128

if options.fermi:
    system.ruby.clock = "2.6GHz" # NOTE: This is the memory clock

for i in xrange(options.num_sc):
   system.stream_proc_array.shader_cores[i].dataPort = system.ruby._cpu_ruby_ports[options.num_cpus+i].slave
   system.stream_proc_array.shader_cores[i].instPort = system.ruby._cpu_ruby_ports[options.num_cpus+i].slave

for (i, cpu) in enumerate(system.cpu):
    ruby_port = system.ruby._cpu_ruby_ports[i]

    cpu.createInterruptController()
    cpu.interrupts.pio = ruby_port.master
    cpu.interrupts.int_master = ruby_port.slave
    cpu.interrupts.int_slave = ruby_port.master
    #
    # Tie the cpu ports to the ruby cpu ports
    #
    if options.gpu_only:
        cpu.icache_port = system.physmem.port
        cpu.dcache_port = system.physmem.port
    else:
        cpu.icache_port = system.ruby._cpu_ruby_ports[i].slave
        cpu.dcache_port = system.ruby._cpu_ruby_ports[i].slave

    '''process = LiveProcess()
    process.executable = options.cmd
    process.cmd = [options.cmd, str(i)]
    '''
    cpu.workload = process

if options.baseline:
    # need to do this after ruby created
    for i in xrange(options.num_dirs):
        exec("system.dir_cntrl%d.memBuffer.mem_bus_cycle_multiplier = 5" % i)

if options.fermi:
   system.ruby.block_size_bytes = 128
   for i in xrange(options.num_dirs):
      exec("system.dir_cntrl%d.memBuffer.mem_bus_cycle_multiplier = 1" % i)
      exec("system.dir_cntrl%d.memBuffer.mem_ctl_latency = 1" % i)

# Tie the copy engine port to its cache
system.stream_proc_array.ce.cePort = system.ruby._cpu_ruby_ports[options.num_cpus+options.num_sc].slave

root = Root(full_system = False, system = system)

Simulation.run(options, root, system, FutureClass)
