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

#ifndef __LSQ_WARP_INST_BUFFER_HH__
#define __LSQ_WARP_INST_BUFFER_HH__

#include "mem/packet.hh"

/**
 * The WarpInstBuffer class represents a hardware buffer to hold a warp
 * instruction that is in-flight in a GPU load-store queue. It tracks the
 * current state of memory requests from each thread executing the warp
 * instruction, and it is responsible for functionally coalescing those
 * requests into cache accesses, represented by the CoalescedAccesses class.
 *
 * For coding convenience, the WarpInstBuffer class tracks all cache accesses
 * through coalescing, translation, injection and ejection pipeline stages.
 * This may not be consistent with actual hardware designs that likely queue
 * accesses in various hardware LSQ buffers after coalescing.
 */
class WarpInstBuffer {
  private:
    // An enumeration to track the current state of the warp instruction
    //  EMPTY: This buffer does not contain a valid warp instruction
    //  PRECOALESCE: This buffer has accepted warp instruction requests, but
    //               has not yet coalesced the requests into cache accesses
    //  COALESCED: Requests have been coalesced into cache accesses, which will
    //             first be translated, then sent to the cache hierarchy
    enum BufferState { EMPTY, PRECOALESCE, COALESCED };

    int warpId;
    const unsigned laneCount;
    const unsigned warpParts;
    BufferState state;
    // Track whether this warp instruction is a load or store
    // NOTE: If other instruction types are introduced, this may need to be
    // expanded to an enum type iterating possible memory instructions
    bool isLoadInst;
    unsigned requestDataSize;
    // Tick values to track latency of warp instructions
    Tick startTick;
    Tick firstCycleTick;
    Tick completeCycleTick;
    MasterID masterId;
    // An array to hold warp instruction requests per lane (thread) while
    // they are coalesced and access the caches
    PacketPtr* laneRequestPkts;
    Addr pc;

    // Coalesce requests into cache accesses
    void coalesce();
    // Called from coalesce() to instantiate the CoalescedAccess
    void generateCoalescedAccesses(Addr addr, size_t size,
                                   std::list<unsigned> &active_lanes);

    Addr getLaneAddr(unsigned lane_id)
    {
        PacketPtr lane_pkt = laneRequestPkts[lane_id];
        assert(lane_pkt);
        return lane_pkt->req->getVaddr();
    }

    uint8_t* getLaneData(unsigned lane_id)
    {
        assert(lane_id < laneCount);
        PacketPtr lane_pkt = laneRequestPkts[lane_id];
        assert(lane_pkt);
        return lane_pkt->getPtr<uint8_t>();
    }

  public:

    // CoalescedAccesses are generated through the request coalescing process.
    // After coalescing and translation, these accesses are sent to the
    // cache hierarchy. Note that a CoalescedAccess descends from
    // Packet::SenderState, so it can be tagged on a Request using the standard
    // interface for translation. It is descendant from Packet, so it can be
    // sent directly to the caches using the standard ports interface.
    class CoalescedAccess : public Packet, public Packet::SenderState {
      private:
        // The warp instruction that generated this access
        WarpInstBuffer *warpInst;
        uint8_t *pktData;
        // The lanes of the warp that are participating in this access
        std::list<unsigned> activeLanes;
        Cycles injectTime;

      public:
        CoalescedAccess(RequestPtr _req, MemCmd _cmd, WarpInstBuffer *warp_inst,
                    std::list<unsigned> active_lanes, uint8_t *pkt_data = NULL)
            : Packet(_req, _cmd), warpInst(warp_inst), pktData(pkt_data),
              activeLanes(active_lanes), injectTime(0) {}

        ~CoalescedAccess()
        {
            assert(activeLanes.empty());
            if (pktData) delete [] pktData;
            if (req) delete req;
        }

        WarpInstBuffer *getWarpBuffer() { return warpInst; }
        uint8_t* getPktData() { return pktData; }
        int getWarpId() { return warpInst->getWarpId(); }
        std::list<unsigned> *getActiveLanes() { return &activeLanes; };

        void setInjectCycle(Cycles inject_time) { injectTime = inject_time; }
        Cycles getInjectCycle() { return injectTime; }

        Cycles tlbStartCycle;
    };

  private:
    // Buffers for convenience of tracking accesses for this warp instruction:

    // Buffer to track accesses generated by coalescing stage for this warp
    // instruction. Accesses are held in this buffer until injected into the
    // cache hierarchy
    std::list<CoalescedAccess*> coalescedAccesses;
    // Buffer to hold accesses that have been translated. Accesses are held in
    // this buffer until ejected from the cache hierarchy
    std::list<CoalescedAccess*> translatedAccesses;

    void removeTranslated(CoalescedAccess *mem_access)
    {
        translatedAccesses.remove(mem_access);
    }

  public:
    WarpInstBuffer(unsigned lane_count, unsigned warp_parts = 1)
        : warpId(-1), laneCount(lane_count), warpParts(warp_parts),
          state(EMPTY), isLoadInst(false)
    {
        laneRequestPkts = new PacketPtr[laneCount];
        for (int i = 0; i < laneCount; i++) {
            laneRequestPkts[i] = NULL;
        }
    }

    ~WarpInstBuffer()
    {
        if (!coalescedAccesses.empty()) {
            std::list<CoalescedAccess*>::iterator iter =
                                                coalescedAccesses.begin();
            for (; iter != coalescedAccesses.end(); iter++) {
                delete (*iter);
            }
        }
    }

    int getWarpId() { return warpId; }
    void initializeInstBuffer(PacketPtr pkt)
    {
        assert(state == EMPTY);
        state = PRECOALESCE;
        startTick = curTick();
        // If adding support for request types other than reads and writes,
        // this assertion will need to be removed or modified accordingly
        isLoadInst = pkt->isRead();
        if (!isLoadInst) assert(pkt->isWrite());
        warpId = pkt->req->threadId();
        requestDataSize = pkt->getSize();
        pc = pkt->req->getPC();
        masterId = pkt->req->masterId();
    }
    bool isLoad() { return isLoadInst; }
    bool addLaneRequest(unsigned lane_id, PacketPtr pkt);

    void coalesceMemRequests()
    {
        assert(state == PRECOALESCE);
        firstCycleTick = curTick();
        // Functionally coalesce
        coalesce();
        state = COALESCED;
    }

    void removeCoalesced(CoalescedAccess *mem_access)
    {
        coalescedAccesses.remove(mem_access);
    }

    unsigned coalescedAccessesSize()
    {
        return coalescedAccesses.size();
    }

    const std::list<CoalescedAccess*>* getCoalescedAccesses()
    {
        return &coalescedAccesses;
    }

    void setTranslated(CoalescedAccess *mem_access)
    {
        translatedAccesses.push_back(mem_access);
    }

    const std::list<CoalescedAccess*>* getTranslatedAccesses()
    {
        return &translatedAccesses;
    }

    PacketPtr* getLaneRequestPkts() { return laneRequestPkts; }
    void setCompleteTick(Tick time) { completeCycleTick = time; }
    Tick getCompleteTick() { return completeCycleTick; }
    Tick getLatency() { return curTick() - firstCycleTick; }

    // When a memory access is complete, update the lane requests accordingly
    // and signal to the caller whether the warp instruction is complete
    bool finishAccess(CoalescedAccess *mem_access);
    void resetState()
    {
        assert(state == COALESCED);
        assert(coalescedAccesses.empty());
        assert(translatedAccesses.empty());
        warpId = -1;
        state = EMPTY;
    }
};

#endif
