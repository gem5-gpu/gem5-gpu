# Copyright (c) 2012 Mark D. Hill and David A. Wood
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


from MemObject import MemObject
from ShaderTLB import ShaderTLB
from m5.params import *

class ShaderLSQ(MemObject):
    type = 'ShaderLSQ'
    cxx_class = 'ShaderLSQ'
    cxx_header = "gpu/shader_lsq.hh"

    cache_port = MasterPort("The data cache port for this LSQ")

    lane_port = VectorSlavePort("the ports back to the shader core")

    data_tlb = Param.ShaderTLB(ShaderTLB(), "Data TLB")

    control_port = SlavePort("The control port for this LSQ")

    inject_width = Param.Int(1, "Max requests sent to L1 per cycle")
    eject_width = Param.Int(1, "Max cache lines to receive per cycle")

    warp_size = Param.Int(32, "Size of the warp")
    cache_line_size = Param.Int("Cache line size in bytes")
    subline_bytes = Param.Int(32, "Bytes per cache subline (e.g. Fermi = 32")
    warp_contexts = Param.Int(48, "Number of warps possible per GPU core")
    num_warp_inst_buffers = Param.Int(64, "Maximum number of in-flight warp instructions")
    atoms_per_subline = Param.Int(3, "Maximum atomic ops to send per cache subline in a single access (Fermi = 3)")

    # Notes: Fermi back-to-back dependent warp load L1 hits are 19 SM cycles
    # GPGPU-Sim models 5 cycles between LSQ completion and next issued load
    latency = Param.Cycles(14, "Cycles of latency for single uncontested L1 hit")
    l1_tag_cycles = Param.Cycles(4, "Cycles of latency L1 tag access")

    # currently only VI_hammer cache protocol supports flushing.
    # In VI_hammer only the L1 is flushed.
    forward_flush = Param.Bool("Issue a flush all to caches whenever the LSQ is flushed")
