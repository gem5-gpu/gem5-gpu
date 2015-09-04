/*
 * Copyright (c) 2013 Mark D. Hill and David A. Wood
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
 * Authors: Jason Power
 */

#ifndef SHADER_MMU_HH_
#define SHADER_MMU_HH_

#include <list>
#include <map>
#include <queue>
#include <set>

#include "arch/tlb.hh"
#include "base/statistics.hh"
#include "debug/ShaderMMU.hh"
#include "params/ShaderMMU.hh"
#include "gpu/shader_tlb.hh"
#include "sim/clocked_object.hh"
#include "sim/faults.hh"
#include "arch/generic/tlb.hh"

class ShaderMMU : public ClockedObject
{
private:
    std::vector<TheISA::TLB*> pagewalkers;
    std::map<TheISA::TLB*, unsigned> pagewalkerIndices;
    std::vector<bool> activeWalkers;
#if THE_ISA == ARM_ISA
    TheISA::Stage2MMU *stage2MMU;
#endif

    class TranslationRequest : public BaseTLB::Translation
    {
    public:
        ShaderMMU *mmu;
        ShaderTLB *origTLB;
        TheISA::TLB *pageWalker;
        BaseTLB::Translation *wrappedTranslation;
        RequestPtr req;
        BaseTLB::Mode mode;
        ThreadContext *tc;
        Addr vpBase;
        Cycles beginFault;
        Cycles beginWalk;
        Tick startTick;
        bool prefetch;

    public:
        TranslationRequest(ShaderMMU *_mmu, ShaderTLB *_tlb,
                           BaseTLB::Translation *translation, RequestPtr _req,
                           BaseTLB::Mode _mode, ThreadContext *_tc,
                           Tick start_tick, bool prefetch = false);
        Tick getStartTick() { return startTick; }
        void markDelayed() { wrappedTranslation->markDelayed(); }
        void finish(const Fault &fault, RequestPtr _req, ThreadContext *_tc,
                    BaseTLB::Mode _mode)
        {
            assert(_mode == mode);
            assert(_req == req);
            assert(_tc == tc);
            mmu->finishWalk(this, fault);
        }
    };

    // Latency for requests to reach the MMU from the L1 TLBs
    Cycles latency;

    class TLBMissEvent : public Event
    {
        ShaderMMU *mmu;
    public:
        TLBMissEvent(ShaderMMU *_mmu) : mmu(_mmu) {}
        void process() {
            mmu->handleTLBMiss();
        }
    };

    TLBMissEvent startMissEvent;
    std::queue<TranslationRequest*> startMisses;

    class StartPagewalkEvent : public Event
    {
        ShaderMMU *mmu;
        TheISA::TLB *walker;
        TranslationRequest *translation;
    public:
        StartPagewalkEvent(ShaderMMU *_mmu, TheISA::TLB *_walker) :
            mmu(_mmu), walker(_walker), translation(NULL) {}
        void setTranslation(TranslationRequest *_translation) {
            assert(!translation);
            translation = _translation;
        }
        void process() {
            assert(translation);
            TranslationRequest *starting_translation = translation;
            translation = NULL;
            mmu->walk(walker, starting_translation);
        }
    };

    std::vector<StartPagewalkEvent*> pagewalkEvents;

    /**
     * A timeout event for raising page faults to the CPU. Currently, this is
     * used as a deadlock detection mechanism in the event that a page fault
     * is raised but never serviced (e.g. CPU thread has swapped or been
     * suspended, or a simulator bug drops the PF).
     *
     * Schedule this fault timeout for faultTimeoutCycles after raising
     * a page fault. faultTimeoutCycles defaults to 1,000,000 cycles to be
     * consistent with the ShaderLSQ deadlock: More LSQ activity can occur
     * while a page fault is in flight, so the page fault timeout should
     * trigger before the LSQ deadlock in the event that it is dropped.
     *
     * NOTE: GPU-triggered segmentation faults often print CPU output which is
     * often helpful for debugging benchmark memory bugs. Two methods are
     * useful for fixing these: Run in SE mode, which catches segfaults but not
     * page faults, or in FS mode, increase timeouts and deadlock checks to at
     * least 30M GPU cycles to allow the CPU thread to print segfault
     * information.
     */
    class FaultTimeoutEvent : public Event
    {
        ShaderMMU *mmu;
        ThreadContext *tc;
    public:
        FaultTimeoutEvent(ShaderMMU *_mmu) : mmu(_mmu), tc(NULL) {}
        void setTC(ThreadContext *_tc) {
            tc = _tc;
        }
        void process() {
            mmu->faultTimeout(tc);
        }
    };

    FaultTimeoutEvent faultTimeoutEvent;
    Cycles faultTimeoutCycles;

    TLBMemory *tlb;

    enum FaultStatus {
        None, // No outstanding faults
        InKernel, // Waiting for the kernel to handle the pf
        Retrying // Retrying the pagetable walk. May not be complete yet.
    };

    std::queue<TranslationRequest*> pendingWalks;
    std::map<Addr, std::list<TranslationRequest*> > outstandingWalks;
    std::queue<TranslationRequest*> pendingFaults;

    FaultStatus outstandingFaultStatus;
    TranslationRequest *outstandingFaultInfo;

    unsigned int curOutstandingWalks;

    std::map<Addr, GPUTlbEntry> prefetchBuffer;
    int prefetchBufferSize;
    int prefetchAheadDistance;

    void finalizeTranslation(TranslationRequest *translation);

    /// Handle a page fault from a shader TLB
    void handlePageFault(TranslationRequest *translation);

    void setWalkerFree(TheISA::TLB *walker);
    TheISA::TLB *getFreeWalker();
    void schedulePagewalk(TheISA::TLB *walker, TranslationRequest *translation)
    {
        // Start the page walk in the next cycle
        unsigned pw_id = pagewalkerIndices[walker];
        assert(activeWalkers[pw_id]);
        StartPagewalkEvent *spe = pagewalkEvents[pw_id];
        spe->setTranslation(translation);
        schedule(spe, nextCycle());
    }

    // Log the vp base address of the access. If we detect a pattern issue the
    // prefetch. This is currently just a simple 1-ahead prefetcher
    void tryPrefetch(Addr vp_base, ThreadContext *tc);

    // Insert prefetch into prefetch buffer
    void insertPrefetch(Addr vp_base, Addr pp_base);

public:
    /// Constructor
    typedef ShaderMMUParams Params;
    ShaderMMU(const Params *p);
    ~ShaderMMU();

    /// Called from TLBMissEvent after latency cycles has passed since
    /// beginTLBMiss
    void handleTLBMiss();

    /// Called when a shader tlb has a miss
    void beginTLBMiss(ShaderTLB *req_tlb, BaseTLB::Translation *translation,
                      RequestPtr req, BaseTLB::Mode mode, ThreadContext *tc);

    /// Called from a start pagewalk event
    void walk(TheISA::TLB *walker, TranslationRequest *translation) {
        assert(translation->pageWalker == NULL);
        assert(walker != NULL);
        translation->beginWalk = curCycle();
        translation->pageWalker = walker;
        numPagewalks++;
        walker->translateTiming(translation->req, translation->tc, translation,
                                translation->mode);
    }

    // Called after the pagetable walk from TranslationRequest
    void finishWalk(TranslationRequest *translation, Fault fault);

    // Return whether there is a pending GPU fault for decisions about CPU
    // thread handling
    bool isFaultInFlight(ThreadContext *tc);

    // Raise the page fault to the CPU if everything is ready
    void raisePageFaultInterrupt(ThreadContext *tc);

    // Page fault timed out, so crash and/or cleanup
    void faultTimeout(ThreadContext *tc);

    /// Handle a page fault once it's done (called from CUDA API via CudaGPU)
    void handleFinishPageFault(ThreadContext *tc);

    void regStats();

    Stats::Scalar numPagefaults;
    Stats::Scalar numPagewalks;
    Stats::Scalar totalRequests;
    Stats::Scalar l2hits;
    Stats::Scalar prefetchHits;
    Stats::Scalar numPrefetches;
    Stats::Scalar prefetchFaults;

    Stats::Histogram pagefaultLatency;
    Stats::Histogram concurrentWalks;
    Stats::Histogram pagewalkLatency;
};

#endif // SHADER_MMU_HH_
