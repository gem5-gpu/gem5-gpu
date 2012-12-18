

from m5.params import *
from m5.SimObject import SimObject

class FusionProfiler(SimObject):
    type = 'FusionProfiler'
    cxx_class = 'FusionProfiler'
    cxx_header = "mem/ruby/FusionProfiler.py"

    num_sc = Param.Int("number of Shader cores in the GPU")
    ruby_system = Param.RubySystem('')

    bandwidth_interval = Param.Tick(1000000, "Interval to measure bandwidth between")
