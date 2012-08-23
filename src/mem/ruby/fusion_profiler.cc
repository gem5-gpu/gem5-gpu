

#include "mem/ruby/fusion_profiler.hh"
#include "mem/ruby/system/System.hh"

FusionProfiler* FusionProfiler::singletonProfiler = NULL;

FusionProfiler::FusionProfiler(const Params *p)
    : SimObject(p), ruby_system(p->ruby_system)
{
	assert(singletonProfiler == NULL);
	singletonProfiler = this;
	ruby_system->registerFusionProfiler(this);
}

void FusionProfiler::rubyCallback(Time issued, Time initResp, Time fwdResp, Time firstResp, Time now, RubyRequestType type, bool isGPU) {
	if (type == RubyRequestType_LD) {
		if (isGPU) {
			gpuReadLatency.sample(now - issued);
		} else {
			cpuReadLatency.sample(now - issued);
		}
	} else if (type == RubyRequestType_ST) {
		if (isGPU) {
			gpuWriteLatency.sample(now - issued);
		} else {
			cpuWriteLatency.sample(now - issued);
		}
	} else if (type == RubyRequestType_IFETCH) {
		if (isGPU) {
			gpuIfetchLatency.sample(now - issued);
		} else {
			cpuIfetchLatency.sample(now - issued);
		}
	} else {
		if (isGPU) {
			gpuOtherLatency.sample(now - issued);
		} else {
			cpuOtherLatency.sample(now - issued);
		}
	}
}

void FusionProfiler::regStats()
{
	gpuReadLatency
		.init(32)
		.name("FusionProfiler.gpuReadLatency")
		.desc("Latency of all reads to GPU caches");

	gpuWriteLatency
		.init(32)
		.name("FusionProfiler.gpuWriteLatency")
		.desc("Latency of all Writes to GPU caches");

	gpuIfetchLatency
		.init(32)
		.name("FusionProfiler.gpuIfetchLatency")
		.desc("Latency of all Ifetchs to GPU caches");

	gpuOtherLatency
		.init(32)
		.name("FusionProfiler.gpuOtherLatency")
		.desc("Latency of all Others to GPU caches");

	cpuReadLatency
		.init(32)
		.name("FusionProfiler.cpuReadLatency")
		.desc("Latency of all reads to CPU caches");

	cpuWriteLatency
		.init(32)
		.name("FusionProfiler.cpuWriteLatency")
		.desc("Latency of all Writes to CPU caches");

	cpuIfetchLatency
		.init(32)
		.name("FusionProfiler.cpuIfetchLatency")
		.desc("Latency of all Ifetchs to CPU caches");

	cpuOtherLatency
		.init(32)
		.name("FusionProfiler.cpuOtherLatency")
		.desc("Latency of all Others to CPU caches");
}

FusionProfiler *FusionProfilerParams::create() {
    return new FusionProfiler(this);
}
