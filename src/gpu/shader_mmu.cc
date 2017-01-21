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
    ClockedObject(p), pagewalkers(p->pagewalkers),
#if THE_ISA == ARM_ISA
    stage2MMU(p->stage2_mmu),
#endif
    latency(p->latency), startMissEvent(this), faultTimeoutEvent(this),
    faultTimeoutCycles(1000000), outstandingFaultStatus(None),
    outstandingFaultInfo(NULL), curOutstandingWalks(0),
    prefetchBufferSize(p->prefetch_buffer_size)
{
    activeWalkers.resize(pagewalkers.size());
    if (p->l2_tlb_entries > 0) {
        tlb = new TLBMemory(p->l2_tlb_entries, p->l2_tlb_assoc);
    } else {
        tlb = NULL;
    }

    pagewalkEvents.resize(pagewalkers.size());
    for (unsigned pw_id = 0; pw_id < pagewalkers.size(); pw_id++) {
        activeWalkers[pw_id] = false;
        TheISA::TLB* pw = pagewalkers[pw_id];
        pagewalkerIndices[pw] = pw_id;
        pagewalkEvents[pw_id] = new StartPagewalkEvent(this, pw);
#if THE_ISA == ARM_ISA
        pw->setMMU(stage2MMU, pw_id);
#endif
    }
}

ShaderMMU::~ShaderMMU()
{
    if (tlb) {
        delete tlb;
    }
    for (unsigned pw_id = 0; pw_id < pagewalkers.size(); pw_id++) {
        delete pagewalkEvents[pw_id];
    }
}

void
ShaderMMU::beginTLBMiss(ShaderTLB *req_tlb, BaseTLB::Translation *translation,
                        RequestPtr req, BaseTLB::Mode mode, ThreadContext *tc)
{
    // Wrap the translation in another class so we can catch the insertion
    TranslationRequest *wrapped_translation = new TranslationRequest(this,
              req_tlb, translation, req, mode, tc, clockEdge(Cycles(latency)));

    startMisses.push(wrapped_translation);
    if (!startMissEvent.scheduled()) {
        schedule(startMissEvent, startMisses.front()->getStartTick());
    }
}

void
ShaderMMU::handleTLBMiss()
{
    assert(!startMisses.empty());

    TranslationRequest *translation_request = startMisses.front();
    startMisses.pop();

    assert(!startMissEvent.scheduled());
    if (!startMisses.empty()) {
        schedule(startMissEvent, startMisses.front()->getStartTick());
    }

    ShaderTLB *req_tlb = translation_request->origTLB;
    BaseTLB::Translation *translation = translation_request->wrappedTranslation;
    RequestPtr req = translation_request->req;
    BaseTLB::Mode mode = translation_request->mode;
    ThreadContext *tc = translation_request->tc;

    Addr pp_base;
    Addr offset = req->getVaddr() % TheISA::PageBytes;
    Addr vp_base = req->getVaddr() - offset;

    // Check the L2 TLB
    if (tlb && tlb->lookup(vp_base, pp_base, req_tlb)) {
        // Found in the L2 TLB
        l2hits++;
        req->setPaddr(pp_base + offset);
        req_tlb->insert(vp_base, pp_base);
        translation->finish(NoFault, req, tc, mode);
        delete translation_request;
        return;
    }

    // Check for a hit in the prefetch buffers
    auto it = prefetchBuffer.find(vp_base);
    if (it != prefetchBuffer.end()) {
        // Hit in the prefetch buffer
        prefetchHits++;
        pp_base = it->second.ppBase;
        if (tlb) {
            tlb->insert(vp_base, pp_base);
        }
        req->setPaddr(pp_base + offset);
        req_tlb->insert(vp_base, pp_base);
        translation->finish(NoFault, req, tc, mode);
        // Remove from prefetchBuffer
        prefetchBuffer.erase(it);
        // This was a hit in the prefetch buffer, so we must have done the
        // right thing, Let's see if we get lucky again.
        tryPrefetch(vp_base, tc);
        delete translation_request;
        return;
    }

    DPRINTF(ShaderMMU, "Inserting request for vp base %#x. %d outstanding\n",
            vp_base, outstandingWalks[vp_base].size());
    outstandingWalks[vp_base].push_back(translation_request);
    totalRequests++;

    if (outstandingWalks[vp_base].size() == 1) {
        DPRINTF(ShaderMMU, "Walking for %#x\n", req->getVaddr());
        TLB *walker = getFreeWalker();
        if (walker == NULL) {
            pendingWalks.push(translation_request);
        } else {
            schedulePagewalk(walker, translation_request);
            // Try to prefetch on demand misses (but wait until the demand
            // walk has started.)
            tryPrefetch(vp_base, tc);
        }
    }
}

void
ShaderMMU::finishWalk(TranslationRequest *translation, Fault fault)
{
    pagewalkLatency.sample(curCycle() - translation->beginWalk);
    setWalkerFree(translation->pageWalker);
    translation->pageWalker = NULL;

    if (!pendingWalks.empty()) {
        TLB *walker = getFreeWalker();
        TranslationRequest *t = pendingWalks.front();
        schedulePagewalk(walker, t);
        pendingWalks.pop();
    }

    RequestPtr req = translation->req;

    // Handling for after the OS satisfies a page fault
    if (outstandingFaultStatus == Retrying &&
        req->getVaddr() == outstandingFaultInfo->req->getVaddr()) {
        DPRINTF(ShaderMMU, "Walk finished for retry of %#x\n", req->getVaddr());
        if (fault != NoFault) {
            panic("GPU encountered another fault for faulted address.\n"
                  "      Likely a GPU-triggered segfault for: %#x, pc: %#x",
                  req->getVaddr(), req->getPC());
        } else {
            DPRINTF(ShaderMMU, "Retry successful\n");
            outstandingFaultStatus = None;
            outstandingFaultInfo = NULL;
            ThreadContext *tc = translation->tc;
            GPUFaultReg fault_reg = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT);
            fault_reg.inFault = 0;
            // HACK! Setting CPU registers is a convenient way to communicate
            // page fault information to the CPU rather than implementing full
            // memory-mapped device registers. However, setting registers can
            // cause erratic CPU behavior, such as pipeline flushes. Use extreme
            // care/testing when changing these.
            tc->setMiscRegActuallyNoEffect(MISCREG_GPU_FAULT, fault_reg);
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
    Addr vp_base = translation->vpBase;
    Addr pp_base = req->getPaddr() - req->getPaddr() % TheISA::PageBytes;

    DPRINTF(ShaderMMU, "Walk complete for VP %#x to PP %#x\n", vp_base, pp_base);

    list<TranslationRequest*> &walks = outstandingWalks[vp_base];
    assert(walks.front() == translation);
    walks.pop_front();

    // First, complete the walked translation
    if (translation->prefetch) {
        // Only insert into pf buffer if no other requests were made to this
        // virtual page before the prefetch completed
        if (walks.size() == 0) {
            insertPrefetch(vp_base, pp_base);
        }
        delete translation->req;
    } else {
        // Insert the mapping into the TLB. This only needs to happen once
        if (tlb) {
            tlb->insert(vp_base, pp_base);
        }
        // Insert into L1 TLB
        translation->origTLB->insert(vp_base, pp_base);
        // Forward the translation on
        translation->wrappedTranslation->finish(NoFault, translation->req,
                                           translation->tc, translation->mode);
    }

    // Next, complete any queued translations for this same page
    DPRINTF(ShaderMMU, "Walk satisfies %d other requests\n", walks.size());
    list<TranslationRequest*>::iterator it;
    for (it = walks.begin(); it != walks.end(); it++) {
        TranslationRequest *t = (*it);
        // Prefetches should not have been queued
        assert(!t->prefetch);
        assert(t != translation);

        // Set the physical address to complete the translation
        Addr offset = t->req->getVaddr() % TheISA::PageBytes;
        t->req->setPaddr(pp_base + offset);

        // Insert into L1 TLB
        t->origTLB->insert(vp_base, pp_base);
        // Forward the translation on
        t->wrappedTranslation->finish(NoFault, t->req, t->tc, t->mode);

        delete t;
    }
    delete translation;
    outstandingWalks.erase(vp_base);
}

void
ShaderMMU::raisePageFaultInterrupt(ThreadContext *tc)
{
    // NOTE: This function must run through to completion. Otherwise, the
    // outstanding fault may never get raised, and thus, the waiting GPU
    // may deadlock.
    assert(outstandingFaultInfo);
    assert(outstandingFaultStatus == InKernel);

    if (tc != CudaGPU::getCudaGPU(0)->getThreadContext()) {
        warn("Host TC changed! Updating outstanding fault: Old: %p, New: %p\n",
             tc, CudaGPU::getCudaGPU(0)->getThreadContext());
        tc = CudaGPU::getCudaGPU(0)->getThreadContext();
        outstandingFaultInfo->tc = tc;
    }

    // Do ISA-specific checks
#if THE_ISA == X86_ISA
    Addr running_pt_base = CudaGPU::getCudaGPU(0)->getRunningPTBase();
    Addr current_pt_base = tc->readMiscRegNoEffect(MISCREG_CR3);
    if (current_pt_base != running_pt_base) {
        // NOTE: This is an indicator that the CPU thread is either in kernel
        // mode or the thread context has been swapped. In the former case,
        // the interrupts device should make sure that the PF is raised as
        // appropriate. In the latter, the PF may fail, since the CPU thread
        // has been moved.
        warn("Raising GPU PF with incorrect CR3! (running: %p, current: %p)\n",
             running_pt_base, current_pt_base);
    }
#endif

    DPRINTF(ShaderMMU, "Raising interrupt for page fault at addr: %#x\n",
            outstandingFaultInfo->req->getVaddr());

    GPUFaultReg fault_reg = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT);
    assert(fault_reg.inFault == 0);
    fault_reg.inFault = 1;

    GPUFaultCode code = 0;
    code.write = (outstandingFaultInfo->mode == BaseTLB::Write);
    code.user = 1;

    GPUFaultRSPReg fault_rsp = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT_RSP);
    fault_rsp = 0;

    // HACK! Setting CPU registers is a convenient way to communicate page
    // fault information to the CPU rather than implementing full memory-mapped
    // device registers. However, setting registers can cause erratic CPU
    // behavior, such as pipeline flushes. Use extreme care/testing when
    // changing these.
    tc->setMiscRegActuallyNoEffect(MISCREG_GPU_FAULT, fault_reg);
    tc->setMiscRegActuallyNoEffect(MISCREG_GPU_FAULTADDR,
                                   outstandingFaultInfo->req->getVaddr());
    tc->setMiscRegActuallyNoEffect(MISCREG_GPU_FAULTCODE, code);
    tc->setMiscRegActuallyNoEffect(MISCREG_GPU_FAULT_RSP, fault_rsp);

#if THE_ISA == ARM_ISA
    panic("You must be executing in FullSystem mode with ARM:\n"
          "ShaderMMU cannot yet handle ARM page faults");
    // TODO: Add interrupt called "triggerGPUInterrupt()" to the ARM
    // interrupts device
#elif THE_ISA == X86_ISA
    Interrupts *interrupts = tc->getCpuPtr()->getInterruptController();
    interrupts->triggerGPUInterrupt();
#endif

    // Schedule a timeout event to ensure that it does not get randomly
    // dropped. Currently, this is for debugging purposes only (e.g. CPU thread
    // gets swapped or descheduled, or a simulator bug drops the fault).
    assert(!faultTimeoutEvent.scheduled());
    faultTimeoutEvent.setTC(tc);
    schedule(faultTimeoutEvent, clockEdge(faultTimeoutCycles));
}

void
ShaderMMU::faultTimeout(ThreadContext *tc)
{
    warn("Fault is timing out!");
    if (outstandingFaultStatus == InKernel) {
        warn("Fault status: InKernel");
    } else if (outstandingFaultStatus == Retrying) {
        warn("Fault status: Retrying");
    } else {
        warn("Fault status: None");
    }
    panic("Fault outstanding for %d cycles!\n"
          "TC: tc: %p, runningTC: %p, fault reg: %x, fault addr: %x, fault code: %x, fault RSP: %x\n",
          faultTimeoutCycles, tc, CudaGPU::getCudaGPU(0)->getThreadContext(),
          tc->readMiscRegNoEffect(MISCREG_GPU_FAULT),
          tc->readMiscRegNoEffect(MISCREG_GPU_FAULTADDR),
          tc->readMiscRegNoEffect(MISCREG_GPU_FAULTCODE),
          tc->readMiscRegNoEffect(MISCREG_GPU_FAULT_RSP));
}

void
ShaderMMU::handlePageFault(TranslationRequest *translation)
{
    if (!FullSystem) {
        panic("Page fault handling (addr: %#x, pc: %#x) not available in SE "
              "mode: No interrupt handler!\n", translation->vpBase,
              translation->req->getPC());
    }
    if (translation->prefetch) {
        DPRINTF(ShaderMMU, "Ignoring since fault on prefetch\n");
        prefetchFaults++;
        TranslationRequest *new_translation = NULL;
        list<TranslationRequest*> &walks = outstandingWalks[translation->vpBase];
        if (walks.size() != 1) {
            DPRINTF(ShaderMMU, "Well this is complicated. Prefetch fault for"
                                "real request.\n");
            walks.remove(translation);
            new_translation = walks.front();
            delete translation->req;
            delete translation;
        } else {
            outstandingWalks.erase(translation->vpBase);
            delete translation->req;
            delete translation;
            return;
        }
        translation = new_translation;
        assert(translation != NULL);
    }

    ThreadContext *tc = translation->tc;
    if (tc != CudaGPU::getCudaGPU(0)->getThreadContext()) {
        warn("Host TC changed! Old: %p, New: %p. Changing translation\n",
             tc, CudaGPU::getCudaGPU(0)->getThreadContext());
        tc = CudaGPU::getCudaGPU(0)->getThreadContext();
        translation->tc = tc;
    }

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

    raisePageFaultInterrupt(tc);
}

void
ShaderMMU::handleFinishPageFault(ThreadContext *tc)
{
    if (!faultTimeoutEvent.scheduled()) {
        panic("faultTimeoutEvent not scheduled!\n");
    }
    deschedule(faultTimeoutEvent);

    assert(outstandingFaultStatus != None);

    if (tc != CudaGPU::getCudaGPU(0)->getThreadContext()) {
        warn("Finishing: Host TC changed! Old: %p, New: %p. Changing...\n",
             tc, CudaGPU::getCudaGPU(0)->getThreadContext());
        tc = CudaGPU::getCudaGPU(0)->getThreadContext();
    }

    // The CPU sets inFault = 2 when it begins the page fault handler. If this
    // register is not set to 2, the CPU has triggered this handler incorrectly
    // somehow. Fail and print the incorrect value.
    GPUFaultReg fault_reg = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT);
    if (fault_reg.inFault != 2) {
        panic("GPU page fault register status incorrect (%u)!\n",
              fault_reg.inFault);
    }

    Addr returning_rsp = 0;

    // Do ISA-specific register reads and sanity checks
#if THE_ISA == ARM_ISA
    panic("You must be executing in FullSystem mode with ARM ISA:\n"
          "ShaderMMU cannot yet handle ARM page faults");
    // TODO: Add interrupt called "triggerGPUInterrupt()" to the ARM
    // interrupts device
    // TODO: Add sanity check for correct page table
#elif THE_ISA == X86_ISA
    // Sanity check the CR2 register
    Addr cr2 = tc->readMiscRegNoEffect(MISCREG_CR2);
    if (cr2 != outstandingFaultInfo->req->getVaddr()) {
        warn("Handle finish page fault with wrong CR2\n");
        return;
    }

    returning_rsp = tc->readIntReg(INTREG_RSP);
#endif

    // TODO: This test may be ISA-specific to x86. Test this when we get to
    // that point.
    Addr faulting_rsp = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT_RSP);
    if (returning_rsp != faulting_rsp) {
        warn("RSPs do not match... probably segfault! Returning RSP: %x, "
             "Fault RSP: %x", returning_rsp, faulting_rsp);
    }

    DPRINTF(ShaderMMU,
            "Handling a finish page fault event. SP: %x\n", returning_rsp);

    if (outstandingFaultStatus == Retrying) {
        DPRINTF(ShaderMMU, "Already retrying. Maybe queue another?\n");
        return;
    }

    DPRINTF(ShaderMMU, "Retrying pagetable walk for %#x\n",
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
        schedulePagewalk(walker, outstandingFaultInfo);
    }
}

bool
ShaderMMU::isFaultInFlight(ThreadContext *tc)
{
    GPUFaultReg fault_reg = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT);
    return (fault_reg.inFault != 0) && (outstandingFaultStatus == InKernel);
}

void
ShaderMMU::setWalkerFree(TLB *walker)
{
    unsigned pw_id = pagewalkerIndices[walker];
    assert(pagewalkers[pw_id] == walker);
    assert(activeWalkers[pw_id] == true);
    activeWalkers[pw_id] = false;

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
ShaderMMU::tryPrefetch(Addr vp_base, ThreadContext *tc)
{
    // If not using a prefetcher, skip this function.
    if (prefetchBufferSize == 0) {
        return;
    }

    // If this address has already been prefetched, skip
    auto it = prefetchBuffer.find(vp_base);
    if (it != prefetchBuffer.end()) {
        return;
    }
    if (curOutstandingWalks >= pagewalkers.size()) {
        // Not issuing a pagewalk since we already have the max outstanding
        return;
    }

    Addr next_vp_base = vp_base + TheISA::PageBytes;
    Addr pp_base;
    if (tlb && tlb->lookup(next_vp_base, pp_base, false)) {
        // This vp already in the TLB, no need to prefetch
        return;
    }

    if (outstandingWalks.find(next_vp_base) != outstandingWalks.end()) {
        // Already walking for this vp, no need to prefetch
        return;
    }

    numPrefetches++;

    // Prefetch the next PTE into the TLB.
    Request::Flags flags;
    RequestPtr req = new Request(0, next_vp_base, 4, flags, 0, 0, 0, 0);
    TranslationRequest *translation = new TranslationRequest(this, NULL, NULL,
                                        req, BaseTLB::Read, tc, true);
    outstandingWalks[next_vp_base].push_back(translation);
    TLB *walker = getFreeWalker();
    assert(walker != NULL); // Should never try to issue a prefetch in this case

    DPRINTF(ShaderMMU, "Prefetching translation for %#x.\n", next_vp_base);
    schedulePagewalk(walker, translation);
}

void
ShaderMMU::insertPrefetch(Addr vp_base, Addr pp_base)
{
    DPRINTF(ShaderMMU, "Inserting %#x->%#x into pf buffer\n", vp_base, pp_base);
    assert(vp_base % TheISA::PageBytes == 0);
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
    GPUTlbEntry &e = prefetchBuffer[vp_base];
    e.vpBase = vp_base;
    e.ppBase = pp_base;
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
    RequestPtr _req, BaseTLB::Mode _mode, ThreadContext *_tc, Tick start_tick,
    bool prefetch)
            : mmu(_mmu), origTLB(_tlb), pageWalker(NULL),
              wrappedTranslation(translation), req(_req), mode(_mode), tc(_tc),
              beginFault(0), beginWalk(0), startTick(start_tick),
              prefetch(prefetch)
{
    vpBase = req->getVaddr() - req->getVaddr() % TheISA::PageBytes;
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
