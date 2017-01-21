/*
 * Copyright (c) 2012-2013 Mark D. Hill and David A. Wood
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Joel Hestness, Jason Power
 *
 */

#ifndef __GPU_SHADER_LSQ_HH__
#define __GPU_SHADER_LSQ_HH__

#include <queue>
#include <list>
#include <vector>

#include "base/statistics.hh"
#include "cpu/translation.hh"
#include "gpu/lsq_warp_inst_buffer.hh"
#include "gpu/shader_tlb.hh"
#include "mem/mem_object.hh"
#include "mem/port.hh"
#include "params/ShaderLSQ.hh"

/**
 * The ShaderLSQ models the load-store queue for GPU shader cores. The LSQ
 * contains a pool of warp instruction buffers, and manages the progress of
 * active warp instructions.
 *
 * Each warp instruction flows through effectively 4 ShaderLSQ stages:
 *  1) Coalesce + address translations [Always 1 cycle given TLB hits]
 *  2) Translations complete, queued waiting to issue accesses [0+ cycles]
 *     - Extra latency to represent L1 tag and data array access of real
 *       hardware is added before accesses are issued to the L1 cache
 *  3) L1 access [1 cycle given uncontended L1 hit]
 *  4) Warp instruction commits [1 cycle given no commit contention]
 *
 * The ShaderLSQ responsibilities include handling requests that come from the
 * GPU core through the LanePorts, signaling the warp instruction buffers to
 * coalesce requests into cache accesses, issuing translations for the coalesced
 * accesses, injecting the accesses into the cache hierarchy through the
 * CachePort, ejecting the access responses, and sending the completed warp
 * memory instruction back to the GPU core for commit.
 *
 * This LSQ is capable of maintaining sequential consistency. It does not allow
 * writes to be read before being sent to the cache hierarchy, so it can
 * support write atomicity as long as the cache coherence protocol also
 * enforces write atomicity. Program order consistency is enforced by ordering
 * warp instructions using per-warp instruction buffer queues.
 *
 * This LSQ has been validated to perform comparably to NVidia Fermi (GTX4XX,
 * GTX5XX) hardware
 */
class ShaderLSQ : public MemObject
{
  protected:
    typedef ShaderLSQParams Params;

  private:
    /**
     * Port which receives requests from the shader core on a per-lane basis
     * and sends replies to the shader core.
     */
    class LanePort : public SlavePort
    {
        ShaderLSQ* lsq;
        int laneId;

      public:
        LanePort(const std::string &_name, ShaderLSQ *owner, int lane_id)
            : SlavePort(_name, owner), lsq(owner), laneId(lane_id) {}

        ~LanePort() {}

      protected:
        virtual bool recvTimingReq(PacketPtr pkt);
        virtual Tick recvAtomic(PacketPtr pkt);
        virtual void recvFunctional(PacketPtr pkt);
        virtual void recvRespRetry();
        virtual AddrRangeList getAddrRanges() const;

    };
    // One lane port for each lane in the shader core
    std::vector<LanePort*> lanePorts;

    class ControlPort : public SlavePort
    {
        ShaderLSQ* lsq;

      public:
        ControlPort(const std::string &_name, ShaderLSQ *owner)
            : SlavePort(_name, owner), lsq(owner) {}

        ~ControlPort() {}

      protected:
        virtual bool recvTimingReq(PacketPtr pkt);
        virtual Tick recvAtomic(PacketPtr pkt);
        virtual void recvFunctional(PacketPtr pkt);
        virtual void recvRespRetry();
        virtual AddrRangeList getAddrRanges() const;

    };
    ControlPort controlPort;

    // A variable to track whether the writeback stage of the core is blocked
    // If so, must block warp instruction commit
    bool writebackBlocked;

    /**
     * Port which sends the coalesced requests to the ruby port
     */
    class CachePort : public MasterPort
    {
      private:
        ShaderLSQ* lsq;

      public:
        CachePort(const std::string &_name, ShaderLSQ *owner)
            : MasterPort(_name, owner), lsq(owner) {}

        bool recvTimingResp(PacketPtr pkt);
        void recvReqRetry();
    };
    CachePort cachePort;

    // Maximum number of lanes (threads) per warp
    unsigned warpSize;

    // Maximum number of warps that can be executing on GPU core
    unsigned maxNumWarpsPerCore;

    // Maximum number of atomic operations to send per subline per access
    unsigned atomsPerSubline;

    // TODO: When adding support for membars, this should be updated to track
    // flushing status on a per-warp basis
    // For SM-wide flush handling
    bool flushing;
    PacketPtr flushingPkt;
    bool forwardFlush;

    // The complete pool of buffers that hold warp instructions in-flight in
    // the LSQ. Other buffers are just pointers to this physical pool.
    WarpInstBuffer** warpInstBufPool;

    // The size of the pool of warp instruction buffers
    unsigned warpInstBufPoolSize;

    // Holds pointers to buffers that are currently unoccupied
    std::queue<WarpInstBuffer*> availableWarpInstBufs;

    // The warp instruction buffer pointers for different stages of the LSQ:
    // Currently, GPGPU-Sim only supports dispatching a single warp instruction
    // to the LSQ per cycle. This pointer holds the warp instruction currently
    // being dispatched by the core
    WarpInstBuffer *dispatchWarpInstBuf;
    // After dispatch, warp instruction buffers are pushed into per-warp
    // queues that maintain warp instruction ordering within each warp,
    // ensuring the program order portion of the consistency model. Only the
    // warp instruction at the head of each queue is allowed to inject accesses
    // into cache hierarchy. Once all accesses have been injected, the warp
    // instruction is removed from the head of the queue.
    std::vector<std::queue<WarpInstBuffer*> > perWarpInstructionQueues;
    // Track the number of outstanding memory accesses from each warp for
    // enforcing memory fence boundaries between instructions
    std::vector<unsigned> perWarpOutstandingAccesses;

    // LSQ latencies:
    // This is specified as a parameter to the LSQ and represents the dispatch
    // to commit latency of a warp instruction that results in a single L1 hit.
    Cycles overallLatencyCycles;
    // This is specified as a parameter to the LSQ and represents the number of
    // cycles during which a coalesced request is accessing the L1 tag array
    Cycles l1TagAccessCycles;
    // This latency is derived from the overall latency, cycles for pipeline
    // latency and the L1 tag access latency. It represents the delay cycles
    // after ejection of the last memory access for a warp instruction in order
    // to model the overall request latency correctly
    Cycles completeCycles;

    // Data TLB to translate coalesced virtual to physical addresses
    ShaderTLB *tlb;

    // Use this cycle specifier to block inject for variable issue latency
    // e.g. Fermi and Maxwell store issue is 1 cycle per cache subline
    unsigned sublineBytes;
    Cycles nextAllowedInject;
    // Number of accesses that can be injected into L1 cache per cycle
    unsigned injectWidth;
    // Buffer to hold accesses to be sent to the cache
    std::deque<WarpInstBuffer::CoalescedAccess*> injectBuffer;

    // Stores whether a cache line is currently blocked by a prior access
    std::map<Addr, bool> blockedLineAddrs;
    // Emulate MSHR queuing of accesses to lines with outstanding accesses
    std::map<Addr, std::queue<WarpInstBuffer::CoalescedAccess*> > blockedAccesses;
    // Block when there are no available MSHRs to forward the request to lower
    // levels of the cache hierarchy
    bool mshrsFull;
    // Track the number of cycles during which all MSHRs are full
    Cycles mshrsFullStarted;

    // The maximum number of memory accesses that the LSQ can accept from the
    // cache hierarchy per cycle
    unsigned ejectWidth;
    // Buffer to hold accesses when received from the caches until they can
    // be used to update the appropriate WarpInstBuffer
    std::queue<WarpInstBuffer::CoalescedAccess*> ejectBuffer;

    // Buffer to queue warp instruction completions to be sent to core
    std::queue<WarpInstBuffer*> commitInstBuffer;

    unsigned cacheLineAddrMaskBits;
    inline Addr addrToLine(Addr addr) {
        return addr & (((Addr)-1) << cacheLineAddrMaskBits);
    }

    // Helper functions and variables for tracking active warp instruction
    // buffers for stats
    void incrementActiveWarpInstBuffers();
    void decrementActiveWarpInstBuffers();
    Tick lastWarpInstBufferChange;
    unsigned numActiveWarpInstBuffers;

  public:

    ShaderLSQ(Params *params);
    ~ShaderLSQ();

    // Required for implementing MemObject
    virtual BaseMasterPort& getMasterPort(const std::string &if_name, PortID idx = -1);
    virtual BaseSlavePort& getSlavePort(const std::string &if_name, PortID idx = -1);
    bool isSquashed() { return false; }
    void finishTranslation(WholeTranslationState *state);

  private:

    // Accept warp instruction and flush requests from the shader core into LSQ
    bool addFlushRequest(PacketPtr pkt);
    bool addLaneRequest(int lane_id, PacketPtr pkt);

    // LSQ Pipeline Stage 1:
    // Process the dispatchWarpInstBuf, which is holding requests received
    // during the previous cycle. This includes coalescing requests into cache
    // accesses and issuing translations for lines accessed
    void dispatchWarpInst();
    void issueWarpInstTranslations(WarpInstBuffer *warp_inst);
    void pushToInjectBuffer(WarpInstBuffer::CoalescedAccess *mem_request);

    // LSQ Pipeline Stage 2:
    // After coalescing and translating addresses for cache accesses, they
    // can be injected into the cache hierarchy
    void injectCacheAccesses();
    void scheduleRetryInject();

    // LSQ Pipeline Stage 3:
    // Accept cache access responses and queue them for ejection. Ejection
    // consists of rejoining the access with its WarpInstBuffer to update
    // threads affected by the access
    bool recvResponsePkt(PacketPtr pkt);
    void ejectAccessResponses();

    // LSQ Pipeline Stage 4:
    // Once a WarpInstBuffer has received responses for all cache accesses, it
    // can be committed by signaling back to the shader core
    void clearFenceAtQueueHead(int warp_id);
    void pushToCommitBuffer(WarpInstBuffer *warp_inst);
    void commitWarpInst();
    void retryCommitWarpInst();

    // Flush handling functions
    void processFlush();
    void finalizeFlush();

    // Events that trigger warp instruction dispatch, cache accesses injection
    // and ejection, and warp instruction commit pipeline stages, respectively
    EventWrapper<ShaderLSQ, &ShaderLSQ::dispatchWarpInst> dispatchInstEvent;
    EventWrapper<ShaderLSQ, &ShaderLSQ::injectCacheAccesses> injectAccessesEvent;
    EventWrapper<ShaderLSQ, &ShaderLSQ::ejectAccessResponses> ejectAccessesEvent;
    EventWrapper<ShaderLSQ, &ShaderLSQ::commitWarpInst> commitInstEvent;

    // Stats
    Stats::Histogram activeWarpInstBuffers;
    Stats::Average accessesOutstandingToCache;
    Stats::Scalar writebackBlockedCycles;
    Stats::Scalar mshrHitQueued;
    Stats::Scalar mshrsFullCycles;
    Stats::Scalar mshrsFullCount;

    Stats::Histogram warpCoalescedAccesses;
    Stats::Histogram warpLatencyRead;
    Stats::Histogram warpLatencyWrite;
    Stats::Histogram warpLatencyFence;
    Stats::Histogram warpLatencyAtomic;
    Stats::Histogram tlbMissLatency;
    void regStats();

};

#endif
