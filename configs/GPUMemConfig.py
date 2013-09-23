# Copyright (c) 2013 Mark D. Hill and David A. Wood
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

import m5
from m5.objects import *

def addMemCtrlOptions(parser):
    parser.add_option("--mem_ctl_latency", type="int", default=-1, help="Memory controller latency in cycles")
    parser.add_option("--mem_freq", type="string", default="400MHz", help="Memory controller frequency")
    parser.add_option("--membus_busy_cycles", type="int", default=-1, help="Memory bus busy cycles per data transfer")
    parser.add_option("--membank_busy_time", type="string", default=None, help="Memory bank busy time in ns (CL+tRP+tRCD+CAS)")

def setMemoryControlOptions(system, options):
    from m5.params import Latency
    cpu_mem_ctl_clk = SrcClockDomain(clock = options.mem_freq,
                                     voltage_domain = system.voltage_domain)
    for i in xrange(options.num_dirs):
        cntrl = eval("system.ruby.dir_cntrl%d" % i)
        if options.mem_freq:
            cntrl.memBuffer.clk_domain = cpu_mem_ctl_clk
        if options.mem_ctl_latency >= 0:
            cntrl.memBuffer.mem_ctl_latency = options.mem_ctl_latency
        if options.membus_busy_cycles > 0:
            cntrl.memBuffer.basic_bus_busy_time = options.membus_busy_cycles
        if options.membank_busy_time:
            mem_cycle_seconds = float(cntrl.memBuffer.clk_domain.clock.period)
            bank_latency_seconds = Latency(options.membank_busy_time)
            cntrl.memBuffer.bank_busy_time = long(bank_latency_seconds.period / mem_cycle_seconds)

    if options.split:
        for i in xrange(options.num_dev_dirs):
            cntrl = eval("system.ruby.dev_dir_cntrl%d" % i)
            if options.gpu_mem_freq:
                gpu_mem_ctl_clk = SrcClockDomain(clock = options.gpu_mem_freq,
                                         voltage_domain = system.voltage_domain)
                cntrl.memBuffer.clk_domain = gpu_mem_ctl_clk
            else:
                cntrl.memBuffer.clk_domain = cpu_mem_ctl_clk
            if options.gpu_mem_ctl_latency >= 0:
                cntrl.memBuffer.mem_ctl_latency = options.gpu_mem_ctl_latency
            if options.gpu_membus_busy_cycles > 0:
                cntrl.memBuffer.basic_bus_busy_time = options.gpu_membus_busy_cycles
            if options.gpu_membank_busy_time:
                mem_cycle_seconds = float(cntrl.memBuffer.clk_domain.clock.period)
                bank_latency_seconds = Latency(options.gpu_membank_busy_time)
                cntrl.memBuffer.bank_busy_time = long(bank_latency_seconds.period / mem_cycle_seconds)
