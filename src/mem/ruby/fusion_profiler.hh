

#ifndef __MEM_FUSION_PROFILER_PROFILER_HH__
#define __MEM_FUSION_PROFILER_PROFILER_HH__


#include "base/statistics.hh"
#include "mem/protocol/GenericMachineType.hh"
#include "mem/protocol/RubyRequestType.hh"
#include "mem/ruby/common/TypeDefines.hh"
#include "mem/ruby/system/CacheMemory.hh"
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

    void rubyCallback(Time issued, Time initResp, Time fwdResp, Time firstResp,
    				  Time now, RubyRequestType type, GenericMachineType mach, 
    				  bool isGPU);

    void regStats();

    Tick getBandwithInterval() { return bandwidthInterval; }

	Stats::Histogram gpuReadLatency;
	Stats::Histogram gpuWriteLatency;
	Stats::Histogram gpuIfetchLatency;
	Stats::Histogram gpuOtherLatency;
	Stats::Histogram cpuReadLatency;
	Stats::Histogram cpuWriteLatency;
	Stats::Histogram cpuIfetchLatency;
	Stats::Histogram cpuOtherLatency;
    Stats::Histogram readRequestPerWarp;
    Stats::Histogram totalWarpReadLatency;
    Stats::Histogram interWarpReadLatency;
    Stats::Histogram writeRequestPerWarp;
    Stats::Histogram totalWarpWriteLatency;
    Stats::Histogram interWarpWriteLatency;

    Stats::Vector gpuL1ReadHits;
    Stats::Vector gpuL1ReadMisses;
    Stats::Vector gpuL1WriteHits;
    Stats::Vector gpuL1WriteMisses;
    Stats::Scalar gpuL2ReadHits;
    Stats::Scalar gpuL2ReadMisses;
    Stats::Scalar gpuL2WriteHits;
    Stats::Scalar gpuL2WriteMisses;

    Stats::Formula gpuL1ReadHitRate;
    Stats::Formula gpuL1WriteHitRate;
    Stats::Formula gpuL2ReadHitRate;
    Stats::Formula gpuL2WriteHitRate;
    Stats::Formula gpuL1HitRate;
    Stats::Formula gpuL2HitRate;


private:
	RubySystem* ruby_system;
	int numSC;

  Tick bandwidthInterval;

};

class WarpMemRequest
{
public:
	WarpMemRequest();

	void addRequest(Tick time) {
		if (start == 0) {
			start = time;
		}
		outstandingRequests++;
	}

	// Returns true if this is the last request
	bool requestFinish(Tick time, bool isRead) {
		assert(outstandingRequests > 0);
		if (firstFinish == 0) {
			firstFinish = time;
			if (isRead) {
				FusionProfiler::getProfiler()->readRequestPerWarp.sample(outstandingRequests);
			} else {
				FusionProfiler::getProfiler()->writeRequestPerWarp.sample(outstandingRequests);
			}
		}
		outstandingRequests--;
		if (outstandingRequests == 0) {
			lastFinish = time;
			if (isRead) {
				FusionProfiler::getProfiler()->totalWarpReadLatency.sample((lastFinish - start)/freq);
				FusionProfiler::getProfiler()->interWarpReadLatency.sample((lastFinish - firstFinish)/freq);
			} else {
				FusionProfiler::getProfiler()->totalWarpWriteLatency.sample((lastFinish - start)/freq);
				FusionProfiler::getProfiler()->interWarpWriteLatency.sample((lastFinish - firstFinish)/freq);
			}
			return true;
		}
		return false;
	}

private:
	Tick start;
	Tick firstFinish;
	Tick lastFinish;
	unsigned outstandingRequests;

	Tick freq;
};

void profileGPUL1Access(bool isRead, bool isHit, int version);
void profileGPUL2Access(bool isRead, bool isHit);
void profileGPUL2WriteMiss(GenericMachineType mach);

#endif
