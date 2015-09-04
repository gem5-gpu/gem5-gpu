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
from Ruby import create_topology

class L1Cache(RubyCache): pass
class L2Cache(RubyCache): pass

def create_system(options, full_system, system, dma_ports, ruby_system):

    if not buildEnv['GPGPU_SIM']:
        m5.util.panic("This script requires GPGPU-Sim integration to be built.")


    options.access_backing_store = True

    print "Creating system for GPU"

    # Run the original protocol script
    buildEnv['PROTOCOL'] = buildEnv['PROTOCOL'][:-7]
    protocol = buildEnv['PROTOCOL']
    exec "import %s" % protocol
    try:
        (cpu_sequencers, dir_cntrl_nodes, topology) = \
            eval("%s.create_system(options, full_system, system, dma_ports, ruby_system)" % protocol)
    except:
        print "Error: could not create system for ruby protocol inside fusion system %s" % protocol
        raise

    #
    # Must create the individual controllers before the network to ensure the
    # controller constructors are called before the network constructor
    #
    block_size_bits = int(math.log(options.cacheline_size, 2))

    cntrl_count = 0

    for i in xrange(options.num_sc):
        #
        # First create the Ruby objects associated with this cpu
        #
        l1i_cache = L1Cache(size = options.l1i_size,
                            assoc = options.l1i_assoc,
                            start_index_bit = block_size_bits,
                            is_icache = True)
        l1d_cache = L1Cache(size = options.l1d_size,
                            assoc = options.l1d_assoc,
                            start_index_bit = block_size_bits)
        l2_cache = L2Cache(size = options.l2_size,
                           assoc = options.l2_assoc,
                           start_index_bit = block_size_bits)

        l1_cntrl = L1Cache_Controller(version = options.num_cpus+i,
                                      L1Icache = l1i_cache,
                                      L1Dcache = l1d_cache,
                                      L2cache = l2_cache,
                                      no_mig_atomic = not \
                                        options.allow_atomic_migration,
                                      send_evictions = False,
                                      transitions_per_cycle = options.ports,
                                      ruby_system = ruby_system)

        cpu_seq = RubySequencer(version = options.num_cpus + i,
                                icache = l1i_cache,
                                dcache = l1d_cache,
                                max_outstanding_requests = options.gpu_l1_buf_depth,
                                ruby_system = ruby_system,
                                connect_to_io = False)

        l1_cntrl.sequencer = cpu_seq
        if options.recycle_latency:
            l1_cntrl.recycle_latency = options.recycle_latency

        exec("ruby_system.l1_cntrl_sp%02d = l1_cntrl" % i)

        #
        # Add controllers and sequencers to the appropriate lists
        #
        cpu_sequencers.append(cpu_seq)
        topology.addController(l1_cntrl)

        # Connect the L1 controller and the network
        # Connect the buffers from the controller to network
        l1_cntrl.requestFromCache = MessageBuffer()
        l1_cntrl.requestFromCache.master = ruby_system.network.slave
        l1_cntrl.responseFromCache = MessageBuffer()
        l1_cntrl.responseFromCache.master = ruby_system.network.slave
        l1_cntrl.unblockFromCache = MessageBuffer()
        l1_cntrl.unblockFromCache.master = ruby_system.network.slave

        l1_cntrl.triggerQueue = MessageBuffer()

        # Connect the buffers from the network to the controller
        l1_cntrl.mandatoryQueue = MessageBuffer()
        l1_cntrl.forwardToCache = MessageBuffer()
        l1_cntrl.forwardToCache.slave = ruby_system.network.master
        l1_cntrl.responseToCache = MessageBuffer()
        l1_cntrl.responseToCache.slave = ruby_system.network.master

        cntrl_count += 1

    if not options.split:
        ########################################################################
        # Pagewalk cache
        # NOTE: We use a CPU L1 cache controller here. This is to facilatate MMU
        #       cache coherence (as the GPU L1 caches are incoherent without
        #       flushes. The L2 cache is small, and should have minimal affect
        #       on the performance (see Section 6.2 of Power et al. HPCA 2014).
        pwd_cache = L1Cache(size = options.pwc_size,
                                assoc = 16, # 64 is fully associative @ 8kB
                                replacement_policy = LRUReplacementPolicy(),
                                start_index_bit = block_size_bits,
                                resourceStalls = False)
        # Small cache since CPU L1 requires I and D
        pwi_cache = L1Cache(size = "512B",
                                assoc = 2,
                                replacement_policy = LRUReplacementPolicy(),
                                start_index_bit = block_size_bits,
                                resourceStalls = False)
        # Small cache since CPU L1 controller requires L2
        l2_cache = L2Cache(size = "512B",
                               assoc = 2,
                               start_index_bit = block_size_bits,
                               resourceStalls = False)

        l1_cntrl = L1Cache_Controller(version = options.num_cpus + \
                                                options.num_sc,
                                      L1Icache = pwi_cache,
                                      L1Dcache = pwd_cache,
                                      L2cache = l2_cache,
                                      send_evictions = False,
                                      transitions_per_cycle = options.ports,
                                      cache_response_latency = 1,
                                      l2_cache_hit_latency = 1,
                                      number_of_TBEs = options.gpu_l1_buf_depth,
                                      ruby_system = ruby_system)

        cpu_seq = RubySequencer(version = options.num_cpus + options.num_sc,
                                # Never get data from pwi_cache
                                icache = pwd_cache,
                                dcache = pwd_cache,
                                icache_hit_latency = 8,
                                dcache_hit_latency = 8,
                                max_outstanding_requests = \
                                    options.gpu_l1_buf_depth,
                                ruby_system = ruby_system,
                                deadlock_threshold = 2000000,
                                connect_to_io = False)

        l1_cntrl.sequencer = cpu_seq


        ruby_system.l1_pw_cntrl = l1_cntrl
        cpu_sequencers.append(cpu_seq)

        topology.addController(l1_cntrl)

        # Connect the L1 controller and the network
        # Connect the buffers from the controller to network
        l1_cntrl.requestFromCache = MessageBuffer()
        l1_cntrl.requestFromCache.master = ruby_system.network.slave
        l1_cntrl.responseFromCache = MessageBuffer()
        l1_cntrl.responseFromCache.master = ruby_system.network.slave
        l1_cntrl.unblockFromCache = MessageBuffer()
        l1_cntrl.unblockFromCache.master = ruby_system.network.slave

        l1_cntrl.triggerQueue = MessageBuffer()

        # Connect the buffers from the network to the controller
        l1_cntrl.mandatoryQueue = MessageBuffer()
        l1_cntrl.forwardToCache = MessageBuffer()
        l1_cntrl.forwardToCache.slave = ruby_system.network.master
        l1_cntrl.responseToCache = MessageBuffer()
        l1_cntrl.responseToCache.slave = ruby_system.network.master

    # Copy engine cache (make as small as possible, ideally 0)
    l1i_cache = L1Cache(size = "2kB", assoc = 2)
    l1d_cache = L1Cache(size = "2kB", assoc = 2)
    l2_cache = L2Cache(size = "2kB",
                        assoc = 2,
                        start_index_bit = block_size_bits)

    l1_cntrl = L1Cache_Controller(version = options.num_cpus+options.num_sc+1,
                                      L1Icache = l1i_cache,
                                      L1Dcache = l1d_cache,
                                      L2cache = l2_cache,
                                      no_mig_atomic = not \
                                        options.allow_atomic_migration,
                                      send_evictions = False,
                                      transitions_per_cycle = options.ports,
                                      ruby_system = ruby_system)

    #
    # Only one unified L1 cache exists.  Can cache instructions and data.
    #
    cpu_seq = RubySequencer(version = options.num_cpus + options.num_sc + 1,
                            icache = l1i_cache,
                            dcache = l1d_cache,
                            max_outstanding_requests = 64,
                            ruby_system = ruby_system,
                            connect_to_io = False)

    l1_cntrl.sequencer = cpu_seq

    ruby_system.ce_cntrl = l1_cntrl

    cpu_sequencers.append(cpu_seq)
    topology.addController(l1_cntrl)

    # Connect the L1 controller and the network
    # Connect the buffers from the controller to network
    l1_cntrl.requestFromCache = MessageBuffer()
    l1_cntrl.requestFromCache.master = ruby_system.network.slave
    l1_cntrl.responseFromCache = MessageBuffer()
    l1_cntrl.responseFromCache.master = ruby_system.network.slave
    l1_cntrl.unblockFromCache = MessageBuffer()
    l1_cntrl.unblockFromCache.master = ruby_system.network.slave

    l1_cntrl.triggerQueue = MessageBuffer()

    # Connect the buffers from the network to the controller
    l1_cntrl.mandatoryQueue = MessageBuffer()
    l1_cntrl.forwardToCache = MessageBuffer()
    l1_cntrl.forwardToCache.slave = ruby_system.network.master
    l1_cntrl.responseToCache = MessageBuffer()
    l1_cntrl.responseToCache.slave = ruby_system.network.master

    return (cpu_sequencers, dir_cntrl_nodes, topology)
