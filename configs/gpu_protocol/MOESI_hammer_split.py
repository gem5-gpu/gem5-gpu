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
import MemConfig

class L1Cache(RubyCache): pass
class L2Cache(RubyCache): pass
class ProbeFilter(RubyCache): pass

def create_system(options, full_system, system, dma_ports, ruby_system):

    if not buildEnv['GPGPU_SIM']:
        m5.util.panic("This script requires GPGPU-Sim integration to be built.")

    options.access_backing_store = True

    # Run the original protocol script
    buildEnv['PROTOCOL'] = buildEnv['PROTOCOL'].replace('split', 'fusion')
    protocol = buildEnv['PROTOCOL']
    exec "import %s" % protocol
    try:
        (cpu_sequencers, dir_cntrl_nodes, topology) = \
            eval("%s.create_system(options, full_system, system, dma_ports, ruby_system)" % protocol)
    except:
        print "Error: could not create system for ruby protocol inside fusion system %s" % protocol
        raise

    # Faking things to build the rest of the system
    print "Warning!"
    print "Warning: Faking split MOESI_hammer protocol; collecting checkpoints?"
    print "Warning!"

    if options.num_dev_dirs > 0:
        block_size_bits = int(math.log(options.cacheline_size, 2))
        gpu_phys_mem_size = system.gpu.gpu_memory_range.size()
        mem_module_size = gpu_phys_mem_size / options.num_dev_dirs

        #
        # determine size and index bits for probe filter
        # By default, the probe filter size is configured to be twice the
        # size of the L2 cache.
        #
        pf_size = MemorySize(options.sc_l2_size)
        pf_size.value = pf_size.value * 2
        dir_bits = int(math.log(options.num_dev_dirs, 2))
        pf_bits = int(math.log(pf_size.value, 2))
        if options.numa_high_bit:
            if options.pf_on or options.dir_on:
                # if numa high bit explicitly set, make sure it does not overlap
                # with the probe filter index
                assert(options.numa_high_bit - dir_bits > pf_bits)

            # set the probe filter start bit to just above the block offset
            pf_start_bit = block_size_bits
        else:
            if dir_bits > 0:
                pf_start_bit = dir_bits + block_size_bits - 1
            else:
                pf_start_bit = block_size_bits

        dev_dir_cntrls = []
        dev_mem_ctrls = []
        num_cpu_dirs = len(dir_cntrl_nodes)
        for i in xrange(options.num_dev_dirs):
            #
            # Create the Ruby objects associated with the directory controller
            #

            dir_version = i + num_cpu_dirs

            dir_size = MemorySize('0B')
            dir_size.value = mem_module_size

            pf = ProbeFilter(size = pf_size, assoc = 4,
                             start_index_bit = pf_start_bit)

            dev_dir_cntrl = Directory_Controller(version = dir_version,
                                 directory = \
                                 RubyDirectoryMemory( \
                                            version = dir_version,
                                            size = dir_size,
                                            numa_high_bit = \
                                            options.numa_high_bit,
                                            device_directory = True),
                                 probeFilter = pf,
                                 probe_filter_enabled = options.pf_on,
                                 full_bit_dir_enabled = options.dir_on,
                                 transitions_per_cycle = options.ports,
                                 ruby_system = ruby_system)

            if options.recycle_latency:
                dev_dir_cntrl.recycle_latency = options.recycle_latency

            exec("ruby_system.dev_dir_cntrl%d = dev_dir_cntrl" % i)
            dev_dir_cntrls.append(dev_dir_cntrl)

            # Connect the directory controller to the network
            dev_dir_cntrl.forwardFromDir = MessageBuffer()
            dev_dir_cntrl.forwardFromDir.master = ruby_system.network.slave
            dev_dir_cntrl.responseFromDir = MessageBuffer()
            dev_dir_cntrl.responseFromDir.master = ruby_system.network.slave
            dev_dir_cntrl.dmaResponseFromDir = MessageBuffer(ordered = True)
            dev_dir_cntrl.dmaResponseFromDir.master = ruby_system.network.slave

            dev_dir_cntrl.triggerQueue = MessageBuffer(ordered = True)

            dev_dir_cntrl.unblockToDir = MessageBuffer()
            dev_dir_cntrl.unblockToDir.slave = ruby_system.network.master
            dev_dir_cntrl.responseToDir = MessageBuffer()
            dev_dir_cntrl.responseToDir.slave = ruby_system.network.master
            dev_dir_cntrl.requestToDir = MessageBuffer()
            dev_dir_cntrl.requestToDir.slave = ruby_system.network.master
            dev_dir_cntrl.dmaRequestToDir = MessageBuffer(ordered = True)
            dev_dir_cntrl.dmaRequestToDir.slave = ruby_system.network.master
            dev_dir_cntrl.responseFromMemory = MessageBuffer()

            dev_mem_ctrl = MemConfig.create_mem_ctrl(
                MemConfig.get(options.mem_type), system.gpu.gpu_memory_range,
                i, options.num_dev_dirs, int(math.log(options.num_dev_dirs, 2)),
                options.cacheline_size)
            dev_mem_ctrl.port = dev_dir_cntrl.memory
            dev_mem_ctrls.append(dev_mem_ctrl)

            topology.addController(dev_dir_cntrl)

        system.dev_mem_ctrls = dev_mem_ctrls

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

    l1_cntrl = L1Cache_Controller(version = options.num_cpus + options.num_sc,
                                      L1Icache = l1i_cache,
                                      L1Dcache = l1d_cache,
                                      L2cache = l2_cache,
                                      no_mig_atomic = not \
                                          options.allow_atomic_migration,
                                      send_evictions = False,
                                      transitions_per_cycle = options.ports,
                                      ruby_system = ruby_system)

    gpu_ce_seq = RubySequencer(version = options.num_cpus + options.num_sc,
                               icache = l1i_cache,
                               dcache = l1d_cache,
                               max_outstanding_requests = 64,
                               ruby_system = ruby_system,
                               connect_to_io = False)

    l1_cntrl.sequencer = gpu_ce_seq

    ruby_system.dev_ce_cntrl = l1_cntrl

    cpu_sequencers.append(gpu_ce_seq)
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
