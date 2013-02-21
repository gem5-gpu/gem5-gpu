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

#
# Note: the L1 Cache latency is only used by the sequencer on fast path hits
#
class L1Cache(RubyCache):
    latency = 2

#
# Note: the L2 Cache latency is not currently used
#
class L2Cache(RubyCache):
    latency = 10

def create_system(options, system, piobus, dma_ports, ruby_system):

    if not buildEnv['GPGPU_SIM']:
        m5.util.panic("This script requires GPGPU-Sim integration to be built.")

    # Run the original protocol script
    buildEnv['PROTOCOL'] = buildEnv['PROTOCOL'].replace('split', 'fusion')
    protocol = buildEnv['PROTOCOL']
    exec "import %s" % protocol
    try:
        (cpu_sequencers, dir_cntrl_nodes, topology) = \
            eval("%s.create_system(options, system, piobus, dma_ports, ruby_system)" % protocol)
    except:
        print "Error: could not create system for ruby protocol inside fusion system %s" % protocol
        raise

    # Faking things to build the rest of the system
    print "Warning!"
    print "Warning: Faking split MOESI_hammer protocol; collecting checkpoints?"
    print "Warning!"
    options.num_dev_dirs = 0

    #
    # Create controller for the copy engine to connect to in GPU cluster
    # Cache is unused by controller
    #
    block_size_bits = int(math.log(options.cacheline_size, 2))
    l1i_cache = L1Cache(size = "2kB", assoc = 2)
    l1d_cache = L1Cache(size = "2kB", assoc = 2)
    l2_cache = L2Cache(size = "2kB",
                        assoc = 2,
                        start_index_bit = block_size_bits)

    l1_cntrl = L1Cache_Controller(version = options.num_cpus + options.num_sc + 1,
                                      cntrl_id = len(topology),
                                      L1IcacheMemory = l1i_cache,
                                      L1DcacheMemory = l1d_cache,
                                      L2cacheMemory = l2_cache,
                                      no_mig_atomic = not \
                                          options.allow_atomic_migration,
                                      send_evictions = (
                                          options.cpu_type == "detailed"),
                                      ruby_system = ruby_system)

    gpu_ce_seq = RubySequencer(version = options.num_cpus + options.num_sc,
                            icache = l1i_cache,
                            dcache = l1d_cache,
                            access_phys_mem = True,
                            max_outstanding_requests = 64,
                            ruby_system = ruby_system)

    l1_cntrl.sequencer = gpu_ce_seq

    if piobus != None:
        gpu_ce_seq.pio_port = piobus.slave

    ruby_system.l1_cntrl_gpuce = l1_cntrl

    cpu_sequencers.append(gpu_ce_seq)
    topology.addController(l1_cntrl)

    return (cpu_sequencers, dir_cntrl_nodes, topology)
