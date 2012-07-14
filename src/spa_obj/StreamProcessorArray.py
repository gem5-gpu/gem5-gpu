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


from m5.SimObject import SimObject
from m5.defines import buildEnv
from m5.params import *
from m5.proxy import *

class StreamProcessorArray(SimObject):
   type = 'StreamProcessorArray'
   sys = Param.System(Parent.any, "system sp will run on")
   gpuTickConv = Param.Float(1.0, "number of gpgpu ticks per m5 tick")
   useGem5Mem = Param.Bool(True, "flag to enable ruby and disable gpgpu-sim's internal mem");
   sharedMemDelay = Param.Int(1, "Delay to access shared memory in gpgpu-sim ticks");
   nonBlocking = Param.Bool(False, "flag to choose whether GPGPU kernels are nonblocking or blocking");
   launchDelay = Param.Float(0.000005645904, "Kernel launch delay in seconds");
   returnDelay = Param.Float(0.000002217222, "Kernel return delay in seconds");

   ruby = Param.RubySystem(Parent.any, "ruby system")

   ce = Param.SPACopyEngine(Parent.any, "copy engine")

   if buildEnv['TARGET_ISA'] == 'x86':
      from X86TLB import X86TLB
      dtb = Param.X86TLB(X86TLB(), "Data TLB")
      itb = Param.X86TLB(X86TLB(), "Instruction TLB")
   else:
      print "Don't know how to do gpgpusim with %s" % \
         buildEnv['TARGET_ISA']
      sys.exit(1)
   stats_filename = Param.String("gpu_stats.txt",
         "file to which gpgpu-sim dumps its stats")
   config_path = Param.String('gpgpusim.config', "file to which gpgpu-sim dumps its stats")
   dump_kernel_stats = Param.Bool(False, "Dump and reset simulator statistics at the beginning and end of kernels")
