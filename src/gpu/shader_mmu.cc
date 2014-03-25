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

#include "debug/ShaderMMU.hh"
#include "arch/x86/generated/decoder.hh"
#include "arch/x86/regs/misc.hh"
#include "cpu/base.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "gpu/shader_mmu.hh"
#include "params/ShaderMMU.hh"

using namespace std;

ShaderMMU::ShaderMMU(const Params *p) :
    SimObject(p), outstandingFaultStatus(None)
{
}

void
ShaderMMU::handleTLBMiss(X86ISA::TLB *pw_wrapper, ShaderTLB *req_tlb,
              BaseTLB::Translation *translation, RequestPtr req,
              BaseTLB::Mode mode, ThreadContext *tc)
{
    // Wrap the translation in another class so we can catch the insertion
    TranslationRequest *wrappedTranslation =
            new TranslationRequest(this, req_tlb, pw_wrapper, translation, req,
                                   mode, tc);

    DPRINTF(ShaderMMU, "Walking for %#x\n", req->getVaddr());

    numPagewalks++;
    outstandingWalks[req_tlb]++;
    concurrentWalks.sample(outstandingWalks[req_tlb]);

    wrappedTranslation->walk();
}

void
ShaderMMU::finishWalk(TranslationRequest *translation, Fault fault)
{
    RequestPtr req = translation->req;

    if (outstandingFaultStatus == Retrying &&
        req->getVaddr() == outstandingFaultInfo->req->getVaddr()) {
        DPRINTF(ShaderMMU, "Walk finished for retry of %#x\n", req->getVaddr());
        if (fault != NoFault) {
            DPRINTF(ShaderMMU, "Got another fault!\n");
            outstandingFaultStatus = InKernel;
            // No need to invoke another page fault if this wasn't satisfied
            // We could have been notified at the end of some other interrrupt
            // Or the kernel could be in the middle of segfault, etc.
            return;
        } else {
            DPRINTF(ShaderMMU, "Retry successful\n");
            outstandingFaultStatus = None;
            using namespace X86ISA;
            ThreadContext *tc = translation->tc;
            GPUFaultReg faultReg = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT);
            faultReg.inFault = 0;
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

    outstandingWalks[translation->origTLB]--;

    if (fault == NoFault) {
        DPRINTF(ShaderMMU, "Walk successful for vaddr %#x. Paddr %#x\n",
            req->getVaddr(), req->getPaddr());
        finalizeTranslation(translation);
    } else {
        DPRINTF(ShaderMMU, "Walk for vaddr %#x: fault!\n", req->getVaddr());
        // REMOVE this before checkin (and change X86FaultBase back)!!
        uint64_t ec = ((X86ISA::X86FaultBase*)fault.get())->errorCode;
        if (ec != 0x4 && ec != 0x6) {
            DPRINTF(ShaderMMU, "Error code does not match: %#x\n", ec);
        }
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

    translation->origTLB->insert(vpn, ppn);
    translation->wrappedTranslation->finish(NoFault, req, translation->tc,
                                             translation->mode);
}

void
ShaderMMU::handlePageFault(TranslationRequest *translation)
{
    using namespace X86ISA;
    ThreadContext *tc = translation->tc;

    assert(tc == CudaGPU::getCudaGPU(0)->getThreadContext());

    if (outstandingFaultStatus != None) {
        if (outstandingFaultInfo->vpn == translation->vpn) {
            coalescedFaults.push_back(translation);
            DPRINTF(ShaderMMU, "Fault for same page. %d faults pending\n",
                               coalescedFaults.size());
        } else {
            pendingFaults.push(translation);
            DPRINTF(ShaderMMU, "Outstanding fault. %d faults pending \n",
                                pendingFaults.size());
        }
        return;
    }

    numPagefaults++;

    outstandingFaultStatus = InKernel;
    outstandingFaultInfo = translation;
    DPRINTF(ShaderMMU, "invoking fault for %#x\n",
                       translation->req->getVaddr());

    outstandingFaultInfo->beginFault = CudaGPU::getCudaGPU(0)->curCycle();

    GPUFaultReg faultReg = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT);
    assert(faultReg.inFault == 0);

    GPUFaultCode code = 0;
    code.write = (translation->mode == BaseTLB::Write);
    code.user = 1;
    faultReg.inFault = 1;

    tc->setMiscRegNoEffect(MISCREG_GPU_FAULT, faultReg);
    tc->setMiscRegNoEffect(MISCREG_GPU_FAULTADDR, translation->req->getVaddr());
    tc->setMiscRegNoEffect(MISCREG_GPU_FAULTCODE, code);

    HandyM5Reg m5reg = tc->readMiscRegNoEffect(MISCREG_M5_REG);
    if (m5reg.cpl != 3) {
        DPRINTF(ShaderMMU, "Not invoking fault in kernel mode. Waiting.\n");
        outstandingFaultStatus = Pending;
        return;
    }

    Interrupts *interrupts = tc->getCpuPtr()->getInterruptController();
    interrupts->triggerGPUInterrupt();
}

void
ShaderMMU::handleFinishPageFault(ThreadContext *tc)
{
    using namespace X86ISA;
    HandyM5Reg m5reg = tc->readMiscRegNoEffect(MISCREG_M5_REG);
    GPUFaultReg faultReg = tc->readMiscRegNoEffect(MISCREG_GPU_FAULT);
    assert(faultReg.inFault == 1);

    DPRINTF(ShaderMMU, "Handling a finish page fault event\n");

    assert(outstandingFaultStatus != None);

    if (outstandingFaultStatus == Pending) {
        DPRINTF(ShaderMMU, "Invoking the pending fault\n");
        outstandingFaultStatus = InKernel;
        Interrupts *interrupts = tc->getCpuPtr()->getInterruptController();
        interrupts->triggerGPUInterrupt();
        return;
    }

    Addr cr2 = tc->readMiscRegNoEffect(MISCREG_CR2);
    if (cr2 != outstandingFaultInfo->req->getVaddr()) {
        warn("Handle finish page fault with wrong CR2\n");
        return;
    }

    if (outstandingFaultStatus == Retrying) {
        DPRINTF(ShaderMMU, "Already retrying. Maybe queue another?'\n");
        return;
    }

    DPRINTF(ShaderMMU, "Retying pagetable walk for %#x\n",
                        outstandingFaultInfo->req->getVaddr());
    outstandingFaultStatus = Retrying;

    pagefaultLatency.sample(CudaGPU::getCudaGPU(0)->curCycle() -
                            outstandingFaultInfo->beginFault);

    DPRINTF(ShaderMMU, "Walking for %#x\n",
                        outstandingFaultInfo->req->getVaddr());

    numPagewalks++;
    outstandingWalks[outstandingFaultInfo->origTLB]++;
    concurrentWalks.sample(outstandingWalks[outstandingFaultInfo->origTLB]);

    outstandingFaultInfo->walk();
    for (auto t=coalescedFaults.begin(); t!=coalescedFaults.end(); t++) {
        numPagewalks++;
        outstandingWalks[(*t)->origTLB]++;
        concurrentWalks.sample(outstandingWalks[(*t)->origTLB]);
        (*t)->walk();
    }
    coalescedFaults.clear();
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

    pagefaultLatency
        .name(name()+".pagefaultLatency")
        .desc("Latency to complete the pagefault")
        .init(32)
        ;

    concurrentWalks
        .name(name()+".concurrentWalks")
        .desc("Number of ourstanding walks to a single TLB")
        .init(32)
        ;
}

ShaderMMU::TranslationRequest::TranslationRequest(ShaderMMU *_mmu,
    ShaderTLB *_tlb, X86ISA::TLB *pw, BaseTLB::Translation *translation,
    RequestPtr _req, BaseTLB::Mode _mode, ThreadContext *_tc)
            : mmu(_mmu), origTLB(_tlb), pageWalker(pw),
              wrappedTranslation(translation), req(_req), mode(_mode), tc(_tc),
              beginFault(0)
{
    vpn = req->getVaddr() - req->getVaddr() % TheISA::PageBytes;
}

ShaderMMU *ShaderMMUParams::create() {
    return new ShaderMMU(this);
}

// Global function which the x86 microop gpufinishfault calls.
namespace X86ISAInst {
void
gpuFinishPageFault(int gpuId, ThreadContext *tc)
{
    CudaGPU::getCudaGPU(gpuId)->handleFinishPageFault(tc);
}
}
