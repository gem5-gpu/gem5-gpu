

#include "mem/ruby/fusion_profiler/fusion_profiler.hh"
#include "mem/ruby/system/System.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"

FusionProfiler* FusionProfiler::singletonProfiler = NULL;

FusionProfiler::FusionProfiler(const Params *p)
    : SimObject(p), ruby_system(p->ruby_system), numSC(p->num_sc),
      bandwidthInterval(p->bandwidth_interval)
{
    assert(singletonProfiler == NULL);
    singletonProfiler = this;
    //ruby_system->registerFusionProfiler(this);
}

void
FusionProfiler::rubyCallback(Time issued, Time initResp, Time fwdResp,
                             Time firstResp, Time now, RubyRequestType type,
                             GenericMachineType mach, bool isGPU)
{
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


Tick
FusionProfiler::getBandwithInterval()
{
    return bandwidthInterval;
}

void
FusionProfiler::regStats()
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

    readRequestPerWarp
        .init(16)
        .name("FusionProfiler.readRequestPerWarp")
        .desc("Total requests required for each warp")
        ;
    totalWarpReadLatency
        .init(16)
        .name("FusionProfiler.totalWarpReadLatency")
        .desc("Latency in cycles to finish all memory requests for the warp")
        ;
    interWarpReadLatency
        .init(16)
        .name("FusionProfiler.interWarpReadLatency")
        .desc("Latency in cycles because of differing request latency within the warp")
        ;
    writeRequestPerWarp
        .init(16)
        .name("FusionProfiler.writeRequestPerWarp")
        .desc("Total requests required for each warp")
        ;
    totalWarpWriteLatency
        .init(16)
        .name("FusionProfiler.totalWarpWriteLatency")
        .desc("Latency in cycles to finish all memory requests for the warp")
        ;
    interWarpWriteLatency
        .init(16)
        .name("FusionProfiler.interWarpWriteLatency")
        .desc("Latency in cycles because of differing request latency within the warp")
        ;

    gpuL1ReadHits
        .init(numSC)
        .name("FusionProfiler.gpuL1ReadHits")
        .desc("Number of read hits for the L1 cache")
        ;
    gpuL1ReadMisses
        .init(numSC)
        .name("FusionProfiler.gpuL1ReadMisses")
        .desc("Number of read misses for the L1 cache")
        ;
    gpuL1WriteHits
        .init(numSC)
        .name("FusionProfiler.gpuL1WriteHits")
        .desc("Number of writes to valid lines for the L1 cache")
        ;
    gpuL1WriteMisses
        .init(numSC)
        .name("FusionProfiler.gpuL1WriteMisses")
        .desc("Number of writes to invalid lines for the L1 cache")
        ;
    gpuL2ReadHits
        .name("FusionProfiler.gpuL2ReadHits")
        .desc("Number of read hits for the L2 cache")
        ;
    gpuL2ReadMisses
        .name("FusionProfiler.gpuL2ReadMisses")
        .desc("Number of read misses for the L2 cache")
        ;
    gpuL2WriteHits
        .name("FusionProfiler.gpuL2WriteHits")
        .desc("Number of write hits for the L2 cache")
        ;
    gpuL2WriteMisses
        .name("FusionProfiler.gpuL2WriteMisses")
        .desc("Number of write misses for the L2 cache")
        ;
    gpuL1ReadHitRate
        .name("FusionProfiler.gpuL1ReadHitRate")
        .desc("Hits / total accesses")
        ;
    gpuL1WriteHitRate
        .name("FusionProfiler.gpuL1WriteHitRate")
        .desc("Hits / total accesses")
        ;
    gpuL2ReadHitRate
        .name("FusionProfiler.gpuL2ReadHitRate")
        .desc("Hits / total accesses")
        ;
    gpuL2WriteHitRate
        .name("FusionProfiler.gpuL2WriteHitRate")
        .desc("Hits / total accesses")
        ;
    gpuL1HitRate
        .name("FusionProfiler.gpuL1HitRate")
        .desc("Hits / total accesses")
        ;
    gpuL2HitRate
        .name("FusionProfiler.gpuL2HitRate")
        .desc("Hits / total accesses")
        ;

    gpuL1ReadHitRate = gpuL1ReadHits / (gpuL1ReadHits + gpuL1ReadMisses);
    gpuL1WriteHitRate = gpuL1WriteHits / (gpuL1WriteHits + gpuL1WriteMisses);
    gpuL2ReadHitRate = gpuL2ReadHits / (gpuL2ReadHits + gpuL2ReadMisses);
    gpuL2WriteHitRate = gpuL2WriteHits / (gpuL2WriteHits + gpuL2WriteMisses);
    gpuL1HitRate = (gpuL1ReadHits + gpuL1WriteHits) /
                   (gpuL1ReadHits + gpuL1ReadMisses + gpuL1WriteHits + gpuL1WriteMisses);
    gpuL2HitRate = (gpuL2ReadHits + gpuL2WriteHits) /
                   (gpuL2ReadHits + gpuL2ReadMisses + gpuL2WriteHits + gpuL2WriteMisses);
}

FusionProfiler*
FusionProfilerParams::create()
{
    return new FusionProfiler(this);
}


WarpMemRequest::WarpMemRequest()
{
    start = 0;
    outstandingRequests = 0;
    firstFinish = 0;
    lastFinish = 0;
    // TODO: The FusionProfiler needs to flexible for multiple GPUs...
    freq = 0;
}

void
profileGPUL1Access(bool isRead, bool isHit, int version)
{
    if (isRead) {
        if (isHit) {
            FusionProfiler::getProfiler()->gpuL1ReadHits[version]++;
        } else {
            FusionProfiler::getProfiler()->gpuL1ReadMisses[version]++;
        }
    } else {
        if (isHit) {
            FusionProfiler::getProfiler()->gpuL1WriteHits[version]++;
        } else {
            FusionProfiler::getProfiler()->gpuL1WriteMisses[version]++;
        }
    }
}

void
profileGPUL2Access(bool isRead, bool isHit)
{
    if (isRead) {
        if (isHit) {
            FusionProfiler::getProfiler()->gpuL2ReadHits++;
        } else {
            FusionProfiler::getProfiler()->gpuL2ReadMisses++;
        }
    } else {
        if (isHit) {
            FusionProfiler::getProfiler()->gpuL2WriteHits++;
        } else {
            FusionProfiler::getProfiler()->gpuL2WriteMisses++;
        }
    }
}

void
profileGPUL2WriteMiss(GenericMachineType mach)
{
}
