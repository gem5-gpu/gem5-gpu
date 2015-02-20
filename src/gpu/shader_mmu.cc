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

#include <list>

#include "arch/isa.hh"
#include "cpu/base.hh"
#include "debug/ShaderMMU.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "gpu/shader_mmu.hh"
#include "params/ShaderMMU.hh"
#include "sim/full_system.hh"

#if THE_ISA == ARM_ISA
    // TODO: To enable full-system mode ARM interrupts may require including
    // an ARM instruction with a GPU interrupt handler
#elif THE_ISA == X86_ISA
    #include "arch/x86/generated/decoder.hh"
#else
    #error Currently gem5-gpu is only known to support x86 and ARM
#endif

using namespace std;
using namespace TheISA;

ShaderMMU::ShaderMMU(const Params *p) :
    ClockedObject(p), pagewalkers(p->pagewalkers), latency(p->latency),
    outstandingFaultStatus(None), curOutstandingWalks(0),
    prefetchBufferSize(p->prefetch_buffer_size)
{
    activeWalkers.resize(pagewalkers.size());
    if (p->l2_tlb_entries > 0) {
        tlb = new TLBMemory(p->l2_tlb_entries, p->l2_tlb_assoc);
    } else {
        tlb = NULL;
    }
}

ShaderMMU::~ShaderMMU()
{
    if (tlb) {
        delete tlb;
    }
}

void
ShaderMMU::beginTLBMiss(ShaderTLB *req_tlb, BaseTLB::Translation *translation,
                        RequestPtr req, BaseTLB::Mode mode, ThreadContext *tc)
{
    TLBMissEvent *e = new TLBMissEvent(this,
                                       req_tlb, translation, req, mode, tc);
    schedule(e, clockEdge(Cycles(latency)));
}

void
ShaderMMU::handleTLBMiss(ShaderTLB *req_tlb, BaseTLB::Translation *translation,
                         RequestPtr req, BaseTLB::Mode mode, ThreadContext *tc)
{
    Addr ppn;
    Addr offset = req->getVaddr() % TheISA::PageBytes;
    Addr vpn = req->getVaddr() - offset;

    // Check the L2 TLB
    if (tlb && tlb->lookup(vpn, ppn, req_tlb)) {
        // Found in the L2 TLB
        l2hits++;
        req->setPaddr(ppn + offset);
        req_tlb->insert(vpn, ppn);
        translation->finish(NoFault, req, tc, mode);
        return;
    }

    // Check for a hit in the prefetch buffers
    auto it = prefetchBuffer.find(vpn);
    if (it != prefetchBuffer.end()) {
        // Hit in the prefetch buffer
        prefetchHits++;
        ppn = it->second.ppn;
        if (tlb) {
            tlb->insert(vpn, ppn);
        }
        req->setPaddr(ppn + offset);
        req_tlb->insert(vpn, ppn);
        translation->finish(NoFault, req, tc, mode);
        // Remove from prefetchBuffer
        prefetchBuffer.erase(it);
        // This was a hit in the prefetch buffer, so we must have done the
        // right thing, Let's see if we get lucky again.
        tryPrefetch(vpn, tc);
        return;
    }

    // Wrap the translation in another class so we can catch the insertion
    TranslationRequest *wrappedTranslation =
            new TranslationRequest(this, req_tlb, translation, req, mode, tc);

    DPRINTF(ShaderMMU, "Inserting request for vpn %#x. %d outstanding\n", vpn,
            outstandingWalks[vpn].size());
    outstandingWalks[vpn].push_back(wrappedTranslation);
    totalRequests++;

    if (outstandingWalks[vpn].size() == 1) {
        DPRINTF(ShaderMMU, "Walking for %#x\n", req->getVaddr());
        TLB *walker = getFreeWalker();
        if (walker == NULL) {
            pendingWalks.push(wrappedTranslation);
        } else {
            wrappedTranslation->walk(walker);
            // Try to prefetch on demand misses (but wait until the demand
            // walk has started.)
            tryPrefetch(vpn, tc);
        }
    }
}

void
ShaderMMU::finishWalk(TranslationRequest *translation, Fault fault)
{
    pagewalkLatency.sample(curCycle() - translation->beginWalk);
    setWalkerFree(translation->pageWalker);

    if (!pendingWalks.empty()) {
        TLB *walker = getFreeWalker();
        TranslationRequest *t = pendingWalks.front();
        t->walk(walker);
        pendingWalks.pop();
    }

    RequestPtr req = translation->req;

    // Handling for after the OS satisfies a page fault
    if (outstandingFaultStatus == Retrying &&
        req->getVaddr() == outstandingFaultInfo->req->getVaddr()) {
        DPRINTF(ShaderMMU, "Walk finished for retry of %#x\n", req->getVaddr());
        if (fault != NoFault) {
            DPRINTF(ShaderMMU, "Got another fault!\n");
            outstandingFaultStatus = InKernel;
            // No need to invoke another page fault if this wasn't satisfied
            // We could have been notified at the end of some other interrrupt
            // Or the kernel could be in the middle of segfault, etc.
            retryFailures++;
            return;
        } else {
            DPRINTF(ShaderMMU, "Retry successful\n");
            outstandingFaultStatus = None;
            ThreadContext *tc = translation->tc;
            GPUFaultReg faultReg = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT);
            faultReg.inFault = 0;
            // HACK! Setting CPU registers is a convenient way to communicate
            // page fault information to the CPU rather than implementing full
            // memory-mapped device registers. However, setting registers can
            // cause pipeline flushes that upset the proper execution of thread
            // control instructions such as suspend operations. gem5-gpu
            // gem5-patches contain a fix for this bug in the O3CPU, but beware
            // of other thread control problems when handling GPU page faults.
            tc->setMiscRegNoEffect(MISCREG_GPU_FAULT, faultReg);
            if (!pendingFaults.empty()) {
                TranslationRequest *pending = pendingFaults.front();
                DPRINTF(ShaderMMU, "Invoking pending fault %#x\n",
                        pending->req->getVaddr());
                handlePageFault(pending);
                pendingFaults.pop();
            } else {
                DPRINTF(ShaderMMU, "No pending faults\n");
            }
        }
    }

    if (fault == NoFault) {
        DPRINTF(ShaderMMU, "Walk successful for vaddr %#x. Paddr %#x\n",
            req->getVaddr(), req->getPaddr());
        finalizeTranslation(translation);
    } else {
        DPRINTF(ShaderMMU, "Walk for vaddr %#x: fault!\n", req->getVaddr());
        // NOTE: fault.get()->errorCode should be 0x4 or 0x6 (from x86 guide)
        //       0x4 & 0x6 are user-mode reads/writes and this is all the GPU
        //       should issue.
        handlePageFault(translation);
    }
}

void
ShaderMMU::finalizeTranslation(TranslationRequest *translation)
{
    RequestPtr req = translation->req;
    Addr vpn = translation->vpn;
    Addr ppn = req->getPaddr() - req->getPaddr() % TheISA::PageBytes;

    DPRINTF(ShaderMMU, "Walk complete for VPN %#x to PPN %#x\n", vpn, ppn);

    list<TranslationRequest*>::iterator it;
    list<TranslationRequest*> &walks = outstandingWalks[vpn];
    DPRINTF(ShaderMMU, "Walk satifies %d outstanding reqs\n", walks.size());
    for (it = walks.begin(); it != walks.end(); it++) {
        TranslationRequest *t = (*it);

        RequestPtr match_req = t->req;
        if (match_req != translation->req) {
            Addr offset = match_req->getVaddr() % TheISA::PageBytes;
            match_req->setPaddr(ppn + offset);
        }

        if (t->prefetch && match_req == translation->req) {
            // Only insert into pf buffer if no other requests were made to this
            // vpn before the prefetch completed
            if (walks.size() == 1) {
                insertPrefetch(vpn, ppn);
            }
            delete t->req;
        } else {
            // insert the mapping into the TLB
            if (tlb) {
                tlb->insert(vpn, ppn);
            }
            // Insert into L1 TLB
            t->origTLB->insert(vpn, ppn);
            // Forward the translation on
            t->wrappedTranslation->finish(NoFault, match_req, t->tc,
                                          t->mode);
        }
        delete t;
    }
    outstandingWalks.erase(vpn);
}

void
ShaderMMU::handlePageFault(TranslationRequest *translation)
{
    if (!FullSystem) {
        panic("Page fault handling (addr: %#x, pc: %#x) not available in SE "
              "mode: No interrupt handler!\n", translation->vpn,
              translation->req->getPC());
    }
    if (translation->prefetch) {
        DPRINTF(ShaderMMU, "Ignoring since fault on prefetch\n");
        prefetchFaults++;
        TranslationRequest *new_translation = NULL;
        list<TranslationRequest*> &walks = outstandingWalks[translation->vpn];
        if (walks.size() != 1) {
            DPRINTF(ShaderMMU, "Well this is complicated. Prefetch fault for"
                                "real request.\n");
            walks.remove(translation);
            new_translation = walks.front();
            delete translation->req;
            delete translation;
        } else {
            outstandingWalks.erase(translation->vpn);
            delete translation->req;
            delete translation;
            return;
        }
        translation = new_translation;
        assert(translation != NULL);
    }

    ThreadContext *tc = translation->tc;
    assert(tc == CudaGPU::getCudaGPU(0)->getThreadContext());

    if (outstandingFaultStatus != None) {
        pendingFaults.push(translation);
        DPRINTF(ShaderMMU, "Outstanding fault. %d faults pending \n",
                pendingFaults.size());
        return;
    }

    numPagefaults++;

    outstandingFaultStatus = InKernel;
    outstandingFaultInfo = translation;
    DPRINTF(ShaderMMU, "fault for %#x\n", translation->req->getVaddr());

    outstandingFaultInfo->beginFault = curCycle();

    GPUFaultReg faultReg = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT);
    assert(faultReg.inFault == 0);

    GPUFaultCode code = 0;
    code.write = (translation->mode == BaseTLB::Write);
    code.user = 1;
    faultReg.inFault = 1;

    // HACK! Setting CPU registers is a convenient way to communicate page
    // fault information to the CPU rather than implementing full memory-mapped
    // device registers. However, setting registers can cause pipeline flushes
    // that upset the proper execution of thread control instructions such as
    // suspend operations. gem5-gpu gem5-patches contain a fix for this bug in
    // the O3CPU, but beware of other thread control problems when handling
    // GPU page faults.
    tc->setMiscRegNoEffect(MISCREG_GPU_FAULT, faultReg);
    tc->setMiscRegNoEffect(MISCREG_GPU_FAULTADDR, translation->req->getVaddr());
    tc->setMiscRegNoEffect(MISCREG_GPU_FAULTCODE, code);

#if THE_ISA == ARM_ISA
    panic("You must be executing in FullSystem mode with ARM:\n"
          "ShaderMMU cannot yet handle ARM page faults");
    // TODO: Add interrupt called "triggerGPUInterrupt()" to the ARM
    // interrupts device
#elif THE_ISA == X86_ISA
    // Delay the fault if the thread is in kernel mode
    HandyM5Reg m5reg = tc->readMiscRegNoEffect(MISCREG_M5_REG);
    if (m5reg.cpl != 3) {
        DPRINTF(ShaderMMU, "Not invoking fault in kernel mode. Waiting.\n");
        outstandingFaultStatus = Pending;
        return;
    }

    Interrupts *interrupts = tc->getCpuPtr()->getInterruptController();
    interrupts->triggerGPUInterrupt();
#endif
}

void
ShaderMMU::handleFinishPageFault(ThreadContext *tc)
{
    assert(((GPUFaultReg)tc->readMiscRegNoEffect(MISCREG_GPU_FAULT)).inFault == 1);

    DPRINTF(ShaderMMU, "Handling a finish page fault event\n");

    assert(outstandingFaultStatus != None);

#if THE_ISA == ARM_ISA
    panic("You must be executing in FullSystem mode with ARM ISA:\n"
          "ShaderMMU cannot yet handle ARM page faults");
    // TODO: Add interrupt called "triggerGPUInterrupt()" to the ARM
    // interrupts device
    // TODO: Add sanity check for correct page table
#elif THE_ISA == X86_ISA
    if (outstandingFaultStatus == Pending) {
        DPRINTF(ShaderMMU, "Invoking the pending fault\n");
        outstandingFaultStatus = InKernel;
        Interrupts *interrupts = tc->getCpuPtr()->getInterruptController();
        interrupts->triggerGPUInterrupt();
        return;
    }

    // Sanity check the CR2 register
    Addr cr2 = tc->readMiscRegNoEffect(MISCREG_CR2);
    if (cr2 != outstandingFaultInfo->req->getVaddr()) {
        warn("Handle finish page fault with wrong CR2\n");
        return;
    }
#endif

    if (outstandingFaultStatus == Retrying) {
        DPRINTF(ShaderMMU, "Already retrying. Maybe queue another?'\n");
        return;
    }

    DPRINTF(ShaderMMU, "Retying pagetable walk for %#x\n",
                        outstandingFaultInfo->req->getVaddr());
    outstandingFaultStatus = Retrying;

    pagefaultLatency.sample(curCycle() - outstandingFaultInfo->beginFault);

    DPRINTF(ShaderMMU, "Walking for %#x\n",
                        outstandingFaultInfo->req->getVaddr());

    TLB *walker = getFreeWalker();
    if (walker == NULL) {
        // May want to push this to the front in the future to decrease latency
        pendingWalks.push(outstandingFaultInfo);
    } else {
        outstandingFaultInfo->walk(walker);
    }
}

void
ShaderMMU::setWalkerFree(TLB *walker)
{
    int i;
    for (i=0; i<pagewalkers.size(); i++) {
        if (pagewalkers[i] == walker) {
            DPRINTF(ShaderMMU, "Setting walker %d free\n", i);
            assert(activeWalkers[i] == true);
            activeWalkers[i] = false;
            break;
        }
    }
    if (i == pagewalkers.size()) {
        panic("Could not find walker!");
    }
    curOutstandingWalks--;
}

TLB *
ShaderMMU::getFreeWalker()
{
    TLB * walker = NULL;
    for (int i=0; i<activeWalkers.size(); i++) {
        if (walker == NULL && !activeWalkers[i]) {
            DPRINTF(ShaderMMU, "Using walker %d\n", i);
            activeWalkers[i] = true;
            walker = pagewalkers[i];
            break;
        }
    }
    if (walker != NULL) {
        concurrentWalks.sample(curOutstandingWalks);
        curOutstandingWalks++;
    }
    return walker;
}

void
ShaderMMU::tryPrefetch(Addr vpn, ThreadContext *tc)
{
    // If not using a prefetcher, skip this function.
    if (prefetchBufferSize == 0) {
        return;
    }

    // If this address has already been prefetched, skip
    auto it = prefetchBuffer.find(vpn);
    if (it != prefetchBuffer.end()) {
        return;
    }
    if (curOutstandingWalks >= pagewalkers.size()) {
        // Not issuing a pagewalk since we already have the max outstanding
        return;
    }

    Addr next_vpn = vpn + TheISA::PageBytes;
    Addr ppn;
    if (tlb && tlb->lookup(next_vpn, ppn, false)) {
        // This vpn already in the TLB, no need to prefetch
        return;
    }

    if (outstandingWalks.find(next_vpn) != outstandingWalks.end()) {
        // Already walking for this vpn, no need to prefetch
        return;
    }

    numPrefetches++;

    // Prefetch the next PTE into the TLB.
    Request::Flags flags;
    RequestPtr req = new Request(0, next_vpn, 4, flags, 0, 0, 0, 0);
    TranslationRequest *translation = new TranslationRequest(this, NULL, NULL,
                                        req, BaseTLB::Read, tc, true);
    outstandingWalks[next_vpn].push_back(translation);
    TLB *walker = getFreeWalker();
    assert(walker != NULL); // Should never try to issue a prefetch in this case

    DPRINTF(ShaderMMU, "Prefetching translation for %#x.\n", next_vpn);
    translation->walk(walker);
}

void
ShaderMMU::insertPrefetch(Addr vpn, Addr ppn)
{
    DPRINTF(ShaderMMU, "Inserting %#x->%#x into pf buffer\n", vpn, ppn);
    assert(vpn % TheISA::PageBytes == 0);
    // Insert into prefetch buffer
    if (prefetchBuffer.size() >= prefetchBufferSize) {
        // evict unused entry from prefetch buffer
        auto min = prefetchBuffer.begin();
        Tick minTick = curTick();
        for (auto it=prefetchBuffer.begin(); it!=prefetchBuffer.end(); it++) {
            if (it->second.mruTick < minTick) {
                minTick = it->second.mruTick;
                min = it;
            }
        }
        assert(minTick != curTick() && min != prefetchBuffer.end());
        prefetchBuffer.erase(min);
    }
    GPUTlbEntry &e = prefetchBuffer[vpn];
    e.vpn = vpn;
    e.ppn = ppn;
    e.setMRU();
    assert(prefetchBuffer.size() <= prefetchBufferSize);
}

void
ShaderMMU::regStats()
{
    numPagefaults
        .name(name()+".numPagefaults")
        .desc("Number of Pagefaults")
        ;
    numPagewalks
        .name(name()+".numPagewalks")
        .desc("Number of Pagewalks")
        ;
    totalRequests
        .name(name()+".totalRequests")
        .desc("Total number of requests")
        ;
    retryFailures
        .name(name()+".retryFailures")
        .desc("Times the retry walk after a pagefault failed")
        ;
    l2hits
        .name(name()+".l2hits")
        .desc("Hits in the shared L2")
        ;

    prefetchHits
        .name(name() + ".prefetchHits")
        .desc("Number of prefetch hits")
        ;
    numPrefetches
        .name(name() + ".numPrefetches")
        .desc("Number of prefetchs")
        ;
    prefetchFaults
        .name(name() + ".prefetchFaults")
        .desc("Number of faults caused by prefetches")
        ;

    pagefaultLatency
        .name(name()+".pagefaultLatency")
        .desc("Latency to complete the pagefault")
        .init(32)
        ;

    pagewalkLatency
        .name(name()+".pagewalkLatency")
        .desc("Latency to complete the pagewalk")
        .init(32)
        ;

    concurrentWalks
        .name(name()+".concurrentWalks")
        .desc("Number of outstanding walks")
        .init(16)
        ;
}

ShaderMMU::TranslationRequest::TranslationRequest(ShaderMMU *_mmu,
    ShaderTLB *_tlb, BaseTLB::Translation *translation,
    RequestPtr _req, BaseTLB::Mode _mode, ThreadContext *_tc, bool prefetch)
            : mmu(_mmu), origTLB(_tlb),
              wrappedTranslation(translation), req(_req), mode(_mode), tc(_tc),
              beginFault(0), prefetch(prefetch)
{
    vpn = req->getVaddr() - req->getVaddr() % TheISA::PageBytes;
}

ShaderMMU *ShaderMMUParams::create() {
    return new ShaderMMU(this);
}

#if THE_ISA == ARM_ISA
    // TODO: Need to define an analogous function to be called from an ARM
    // instruction at the end of the interrupt handler
#elif THE_ISA == X86_ISA

// Global function which the x86 microop gpufinishfault calls.
namespace X86ISAInst {
void
gpuFinishPageFault(int gpuId, ThreadContext *tc)
{
    CudaGPU::getCudaGPU(gpuId)->handleFinishPageFault(tc);
}
}

#endif
