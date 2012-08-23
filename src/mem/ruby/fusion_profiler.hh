

#ifndef __MEM_FUSION_PROFILER_PROFILER_HH__
#define __MEM_FUSION_PROFILER_PROFILER_HH__


#include "base/statistics.hh"
#include "mem/protocol/RubyRequestType.hh"
#include "mem/ruby/common/TypeDefines.hh"
#include "params/FusionProfiler.hh"
#include "sim/sim_object.hh"

class RubySystem;

class FusionProfiler : public SimObject
{
	typedef FusionProfilerParams Params;

public:
    FusionProfiler(const Params *);

	static FusionProfiler *singletonProfiler;
    static FusionProfiler* getProfiler() {
    	return singletonProfiler;
    }

    void rubyCallback(Time issued, Time initResp, Time fwdResp, Time firstResp, Time now, RubyRequestType type, bool isGPU);

    void regStats();

private:
	RubySystem* ruby_system;

	Stats::Histogram gpuReadLatency;
	Stats::Histogram gpuWriteLatency;
	Stats::Histogram gpuIfetchLatency;
	Stats::Histogram gpuOtherLatency;
	Stats::Histogram cpuReadLatency;
	Stats::Histogram cpuWriteLatency;
	Stats::Histogram cpuIfetchLatency;
	Stats::Histogram cpuOtherLatency;

};

#endif
