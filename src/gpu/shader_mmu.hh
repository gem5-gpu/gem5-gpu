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
#include "sim/tlb.hh"

class ShaderMMU : public ClockedObject
{
private:
    std::vector<TheISA::TLB*> pagewalkers;
    std::vector<bool> activeWalkers;

    // Latency for requests to reach the MMU from the L1 TLBs
    Cycles latency;

    class TLBMissEvent : public Event
    {
        ShaderMMU *mmu;
        ShaderTLB *tlb;
        BaseTLB::Translation *translation;
        RequestPtr req;
        BaseTLB::Mode mode;
        ThreadContext *tc;
    public:
        TLBMissEvent(ShaderMMU *_mmu, ShaderTLB *_tlb,
                     BaseTLB::Translation *_translation, RequestPtr _req,
                     BaseTLB::Mode _mode, ThreadContext *_tc) :
            mmu(_mmu), tlb(_tlb), translation(_translation), req(_req),
            mode(_mode), tc(_tc)
        {
            setFlags(Event::AutoDelete);
        }
        void process() {
            mmu->handleTLBMiss(tlb, translation, req, mode, tc);
        }
    };

    TLBMemory *tlb;

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
        Addr vpn;
        Cycles beginFault;
        Cycles beginWalk;
        bool prefetch;

    public:
        TranslationRequest(ShaderMMU *_mmu, ShaderTLB *_tlb,
                           BaseTLB::Translation *translation, RequestPtr _req,
                           BaseTLB::Mode _mode, ThreadContext *_tc,
                           bool prefetch=false);
        void markDelayed() { wrappedTranslation->markDelayed(); }
        void finish(const Fault &fault, RequestPtr _req, ThreadContext *_tc,
                    BaseTLB::Mode _mode)
        {
            assert(_mode == mode);
            assert(_req == req);
            assert(_tc == tc);
            mmu->finishWalk(this, fault);
        }
        void walk(TheISA::TLB *walker) {
            beginWalk = mmu->curCycle();
            assert(walker != NULL);
            pageWalker = walker;
            mmu->numPagewalks++;
            pageWalker->translateTiming(req, tc, this, mode);
        }
    };

    enum FaultStatus {
        None, // No outstanding faults
        Pending, // Waiting until CPU is not in kernel mode anymore.
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

    // Log the vpn of the access. If we detect a pattern issue the prefetch
    // This is currently just a simple 1-ahead prefetcher
    void tryPrefetch(Addr vpn, ThreadContext *tc);

    // Insert prefetch into prefetch buffer
    void insertPrefetch(Addr vpn, Addr ppn);

public:
    /// Constructor
    typedef ShaderMMUParams Params;
    ShaderMMU(const Params *p);
    ~ShaderMMU();

    /// Called from TLBMissEvent after latency cycles has passed since
    /// beginTLBMiss
    void handleTLBMiss(ShaderTLB *req_tlb, BaseTLB::Translation *translation,
                       RequestPtr req, BaseTLB::Mode mode, ThreadContext *tc);

    /// Called when a shader tlb has a miss
    void beginTLBMiss(ShaderTLB *req_tlb, BaseTLB::Translation *translation,
                      RequestPtr req, BaseTLB::Mode mode, ThreadContext *tc);

    // Called after the pagetable walk from TranslationRequest
    void finishWalk(TranslationRequest *translation, Fault fault);

    /// Handle a page fault once it's done (called from CUDA API via CudaGPU)
    void handleFinishPageFault(ThreadContext *tc);

    void regStats();

    Stats::Scalar numPagefaults;
    Stats::Scalar numPagewalks;
    Stats::Scalar totalRequests;
    Stats::Scalar retryFailures;
    Stats::Scalar l2hits;
    Stats::Scalar prefetchHits;
    Stats::Scalar numPrefetches;
    Stats::Scalar prefetchFaults;

    Stats::Histogram pagefaultLatency;
    Stats::Histogram concurrentWalks;
    Stats::Histogram pagewalkLatency;
};

#endif // SHADER_MMU_HH_
