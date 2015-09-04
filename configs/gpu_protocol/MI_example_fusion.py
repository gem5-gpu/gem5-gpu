# Copyright (c) 2006-2007 The Regents of The University of Michigan
# Copyright (c) 2009 Advanced Micro Devices, Inc.
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
# Authors: Brad Beckmann

import math
import m5
from m5.objects import *
from m5.defines import buildEnv

class L1Cache(RubyCache): pass

def create_system(options, full_system, system, dma_devices, ruby_system):

    if not buildEnv['GPGPU_SIM']:
        m5.util.panic("This script requires GPGPU-Sim integration to be built.")

    options.access_backing_store = True

    # Run the original protocol script
    buildEnv['PROTOCOL'] = buildEnv['PROTOCOL'][:-7]
    protocol = buildEnv['PROTOCOL']
    exec "import %s" % protocol
    try:
        (cpu_sequencers, dir_cntrls, topology) = \
            eval("%s.create_system(options, full_system, system, dma_devices, ruby_system)" % protocol)
    except:
        print "Error: could not create system for ruby protocol inside fusion system %s" % protocol
        raise

    #
    # The ruby network creation expects the list of nodes in the system to be
    # consistent with the NetDest list.  Therefore the l1 controller nodes must be
    # listed before the directory nodes and directory nodes before dma nodes, etc.
    #
    l1_cntrl_nodes = []

    block_size_bits = int(math.log(options.cacheline_size, 2))

    #
    # Caches for the stream processors
    #
    for i in xrange(options.num_sc):
        # First create the Ruby objects associated with this cpu
        # Only one cache exists for this protocol, so by default use the L1D
        # config parameters.
        #
        cache = L1Cache(size = options.sc_l1_size,
                        assoc = options.sc_l1_assoc,
                        replacement_policy = LRUReplacementPolicy(),
                        start_index_bit = block_size_bits)


        l1_cntrl = L1Cache_Controller(version = options.num_cpus + i,
                                      cacheMemory = cache,
                                      send_evictions = False,
                                      transitions_per_cycle = options.ports,
                                      ruby_system = ruby_system)

        #
        # Only one unified L1 cache exists.  Can cache instructions and data.
        #
        cpu_seq = RubySequencer(version = options.num_cpus + i,
                                icache = cache,
                                dcache = cache,
                                max_outstanding_requests = options.gpu_l1_buf_depth,
                                ruby_system = ruby_system,
                                connect_to_io = False)

        l1_cntrl.sequencer = cpu_seq

        exec("ruby_system.l1_cntrl_sp%02d = l1_cntrl" % i)
        #
        # Add controllers and sequencers to the appropriate lists
        #
        cpu_sequencers.append(cpu_seq)
        topology.addController(l1_cntrl)

        # Connect the L1 controllers and the network
        l1_cntrl.mandatoryQueue = MessageBuffer()
        l1_cntrl.requestFromCache = MessageBuffer(ordered = True)
        l1_cntrl.requestFromCache.master = ruby_system.network.slave
        l1_cntrl.responseFromCache = MessageBuffer(ordered = True)
        l1_cntrl.responseFromCache.master = ruby_system.network.slave
        l1_cntrl.forwardToCache = MessageBuffer(ordered = True)
        l1_cntrl.forwardToCache.slave = ruby_system.network.master
        l1_cntrl.responseToCache = MessageBuffer(ordered = True)
        l1_cntrl.responseToCache.slave = ruby_system.network.master

    ############################################################################
    # Pagewalk cache
    # NOTE: We use a CPU L1 cache controller here. This is to facilatate MMU
    #       cache coherence (as the GPU L1 caches are incoherent without flushes
    #       The L2 cache is small, and should have minimal affect on the
    #       performance (see Section 6.2 of Power et al. HPCA 2014).
    pw_cache = L1Cache(size = options.pwc_size,
                       assoc = 16, # 64 is fully associative @ 8kB
                       replacement_policy = LRUReplacementPolicy(),
                       start_index_bit = block_size_bits,
                       resourceStalls = False)

    prefetcher = RubyPrefetcher.Prefetcher()

    l1_cntrl = L1Cache_Controller(version = options.num_cpus + options.num_sc,
                                  cacheMemory = pw_cache,
                                  send_evictions = False,
                                  transitions_per_cycle = options.ports,
                                  ruby_system = ruby_system)

    cpu_seq = RubySequencer(version = options.num_cpus + options.num_sc,
                            icache = pw_cache,
                            dcache = pw_cache,
                            icache_hit_latency = 8,
                            dcache_hit_latency = 8,
                            max_outstanding_requests = options.gpu_l1_buf_depth,
                            ruby_system = ruby_system,
                            connect_to_io = False)

    l1_cntrl.sequencer = cpu_seq


    ruby_system.l1_pw_cntrl = l1_cntrl
    cpu_sequencers.append(cpu_seq)

    topology.addController(l1_cntrl)

    # Connect the L1 controllers and the network
    l1_cntrl.mandatoryQueue = MessageBuffer()
    l1_cntrl.requestFromCache = MessageBuffer(ordered = True)
    l1_cntrl.requestFromCache.master = ruby_system.network.slave
    l1_cntrl.responseFromCache = MessageBuffer(ordered = True)
    l1_cntrl.responseFromCache.master = ruby_system.network.slave
    l1_cntrl.forwardToCache = MessageBuffer(ordered = True)
    l1_cntrl.forwardToCache.slave = ruby_system.network.master
    l1_cntrl.responseToCache = MessageBuffer(ordered = True)
    l1_cntrl.responseToCache.slave = ruby_system.network.master

    #copy engine cache (make as small as possible, ideally 0)
    cache = L1Cache(size = "4kB", assoc = 2)

    l1_cntrl = L1Cache_Controller(version = \
                                      options.num_cpus + options.num_sc + 1,
                                  cacheMemory = cache,
                                  send_evictions = False,
                                  transitions_per_cycle = options.ports,
                                  ruby_system = ruby_system)

    #
    # Only one unified L1 cache exists.  Can cache instructions and data.
    #
    cpu_seq = RubySequencer(version = options.num_cpus + options.num_sc + 1,
                            icache = cache,
                            dcache = cache,
                            max_outstanding_requests = 64,
                            ruby_system = ruby_system,
                            connect_to_io = False)

    l1_cntrl.sequencer = cpu_seq

    ruby_system.ce_cntrl = l1_cntrl

    cpu_sequencers.append(cpu_seq)
    topology.addController(l1_cntrl)

    # Connect the L1 controllers and the network
    l1_cntrl.mandatoryQueue = MessageBuffer()
    l1_cntrl.requestFromCache = MessageBuffer(ordered = True)
    l1_cntrl.requestFromCache.master = ruby_system.network.slave
    l1_cntrl.responseFromCache = MessageBuffer(ordered = True)
    l1_cntrl.responseFromCache.master = ruby_system.network.slave
    l1_cntrl.forwardToCache = MessageBuffer(ordered = True)
    l1_cntrl.forwardToCache.slave = ruby_system.network.master
    l1_cntrl.responseToCache = MessageBuffer(ordered = True)
    l1_cntrl.responseToCache.slave = ruby_system.network.master

    return cpu_sequencers, dir_cntrls, topology
