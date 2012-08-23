

from m5.params import *
from m5.SimObject import SimObject

class FusionProfiler(SimObject):
	type = 'FusionProfiler'

	ruby_system = Param.RubySystem('')
