# Copyright (c) 2006-2007 The Regents of The University of Michigan
# Copyright (c) 2009 Advanced Micro Devices, Inc.
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

import math
import m5
from m5.objects import *
from m5.defines import buildEnv
from Cluster import Cluster

#
# Note: the L1 Cache latency is only used by the sequencer on fast path hits
#
class L1Cache(RubyCache):
    latency = 3

#
# Note: the L2 Cache latency is not currently used
#
class L2Cache(RubyCache):
    latency = 15

def create_system(options, system, piobus, dma_devices, ruby_system):

    if not buildEnv['GPGPU_SIM']:
        m5.util.panic("This script requires GPGPU-Sim integration to be built.")

    # Run the original protocol script
    buildEnv['PROTOCOL'] = buildEnv['PROTOCOL'][:-7]
    protocol = buildEnv['PROTOCOL']
    exec "import %s" % protocol
    try:
        (cpu_sequencers, dir_cntrls, cpuCluster) = \
            eval("%s.create_system(options, system, piobus, dma_devices, ruby_system)" % protocol)
    except:
        print "Error: could not create system for ruby protocol inside fusion system %s" % protocol
        raise

    gpuCluster = Cluster()

    #
    # Caches for the stream processors
    #
    l2_bits = int(math.log(options.num_l2caches, 2))
    block_size_bits = int(math.log(options.cacheline_size, 2))

    for i in xrange(options.num_sc):
        #
        # First create the Ruby objects associated with this cpu
        #
        cache = L1Cache(size = options.sc_l1_size,
                            assoc = options.sc_l1_assoc,
                            replacement_policy = "LRU",
                            start_index_bit = block_size_bits,
                            dataArrayBanks = 4,
                            tagArrayBanks = 4,
                            dataAccessLatency = 4,
                            tagAccessLatency = 4,
                            resourceStalls = False)

        l1_cntrl = GPUL1Cache_Controller(version = i,
                                      cntrl_id = len(cpuCluster)+len(gpuCluster)+len(dir_cntrls),
                                      cache = cache,
                                      l2_select_num_bits = l2_bits,
                                      num_l2 = options.num_l2caches,
                                      issue_latency = 30,
                                      number_of_TBEs = options.gpu_l1_buf_depth,
                                      ruby_system = ruby_system)

        cpu_seq = RubySequencer(version = options.num_cpus + i,
                                icache = cache,
                                dcache = cache,
                                access_phys_mem = True,
                                max_outstanding_requests = options.gpu_l1_buf_depth,
                                ruby_system = ruby_system,
                                deadlock_threshold = 2000000)

        l1_cntrl.sequencer = cpu_seq

        if piobus != None:
            cpu_seq.pio_port = piobus.slave

        exec("ruby_system.l1_cntrl_sp%02d = l1_cntrl" % i)

        #
        # Add controllers and sequencers to the appropriate lists
        #
        cpu_sequencers.append(cpu_seq)
        gpuCluster.add(l1_cntrl)

    l2_index_start = block_size_bits + l2_bits

    for i in xrange(options.num_l2caches):
        #
        # First create the Ruby objects associated with this cpu
        #
        l2_cache = L2Cache(size = options.sc_l2_size,
                           assoc = options.sc_l2_assoc,
                           start_index_bit = l2_index_start,
                           replacement_policy = "LRU",
                           dataArrayBanks = 4,
                           tagArrayBanks = 4,
                           dataAccessLatency = 4,
                           tagAccessLatency = 4,
                           resourceStalls = options.gpu_l2_resource_stalls)

        l2_cntrl = GPUL2Cache_Controller(version = i,
                    cntrl_id = len(cpuCluster)+len(gpuCluster)+len(dir_cntrls),
                    L2cache = l2_cache, ruby_system = ruby_system)

        exec("ruby_system.l2_cntrl%d = l2_cntrl" % i)
        gpuCluster.add(l2_cntrl)

    ######################################################################################
    #copy engine cache (make as small as possible, ideally 0)
    cache = L1Cache(size = "4096B", assoc = 2)

    #
    # Only one unified L1 cache exists.  Can cache instructions and data.
    #
    cpu_seq = RubySequencer(version = options.num_cpus+options.num_sc,
                               icache = cache,
                               dcache = cache,
                               access_phys_mem = True,
                               max_outstanding_requests = 64,
                               support_inst_reqs = False,
                               ruby_system = ruby_system)

    ce_cntrl = GPUCopyDMA_Controller(version = 0,
                                    cntrl_id = len(cpuCluster)+len(gpuCluster)+len(dir_cntrls),
                                    sequencer = cpu_seq,
                                    number_of_TBEs = 256,
                                    ruby_system = ruby_system)

    ruby_system.l1_cntrl_ce = ce_cntrl
    cpu_sequencers.append(cpu_seq)

    mainCluster = Cluster()
    mainCluster.add(ce_cntrl)
    mainCluster.add(cpuCluster)
    mainCluster.add(gpuCluster)

    for cntrl in dir_cntrls:
        mainCluster.add(cntrl)

    return (cpu_sequencers, dir_cntrls, mainCluster)
