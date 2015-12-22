/*
 * Copyright (c) 2011 Mark D. Hill and David A. Wood
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
 */

#include <map>

#include "arch/isa.hh"
#include "debug/ShaderTLB.hh"
#include "gpu/shader_tlb.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"

#if THE_ISA == ARM_ISA
    // May need to include appropriate files for fault handling
#elif THE_ISA == X86_ISA
    #include "arch/x86/insts/microldstop.hh"
    #include "arch/x86/regs/misc.hh"
#else
    #error Currently gem5-gpu is only known to support x86 and ARM
#endif

using namespace std;
using namespace TheISA;

ShaderTLB::ShaderTLB(const Params *p) :
    BaseTLB(p), numEntries(p->entries), hitLatency(p->hit_latency),
    cudaGPU(p->gpu), accessHostPageTable(p->access_host_pagetable)
{
    if (numEntries > 0) {
        tlbMemory = new TLBMemory(p->entries, p->associativity);
    } else {
        tlbMemory = new InfiniteTLBMemory();
    }
    mmu = cudaGPU->getMMU();
}

void
ShaderTLB::unserialize(CheckpointIn &cp)
{
    // Intentionally left blank to keep from trying to read shader header from
    // checkpoint files. Allows for restore into any number of shader cores.
    // NOTE: Cannot checkpoint during kernels
}

void
ShaderTLB::beginTranslateTiming(RequestPtr req,
                                BaseTLB::Translation *translation,
                                BaseTLB::Mode mode)
{
    if (accessHostPageTable) {
        translateTiming(req, cudaGPU->getThreadContext(), translation, mode);
    } else {
        // The below code implements a perfect TLB with instant access to the
        // device page table.
        // TODO: We can shift this around, maybe to memory, maybe hierarchical TLBs
        assert(numEntries == 0);
        Addr vaddr = req->getVaddr();
        Addr page_vaddr = cudaGPU->getGPUPageTable()->addrToPage(vaddr);
        Addr offset = vaddr - page_vaddr;
        Addr page_paddr;
        if (cudaGPU->getGPUPageTable()->lookup(page_vaddr, page_paddr)) {
            DPRINTF(ShaderTLB, "Translation found for vaddr %x = paddr %x\n",
                                vaddr, page_paddr + offset);
            req->setPaddr(page_paddr + offset);
            translation->finish(NoFault, req, NULL, mode);
        } else {
            panic("ShaderTLB missing translation for vaddr: %p! @pc: %p",
                    vaddr, req->getPC());
        }
    }
}

void
ShaderTLB::translateTiming(RequestPtr req, ThreadContext *tc,
                           Translation *translation, Mode mode)
{

#if THE_ISA == ARM_ISA
    // @TODO: Currently, translateTiming should only be called for translating
    // the copy engine's host-side addresses under ARM. These should not raise
    // page faults under SE mode, but it would still be good to check that the
    // CPU thread's state is correct for handling the translation
    warn_once("Should add sanity check for access-host-pagetable under ARM!\n");

    // For some reason, this request flag must be set to verify that data
    // accesses are aligned properly (note: not required for inst fetches)
    req->setFlags(TLB::MustBeOne);
#elif THE_ISA == X86_ISA

    // Include some sanity checking
    uint32_t flags = req->getFlags();

    // If this is true, we're dealing with a request to a non-memory address
    // space.
    if ((flags & SegmentFlagMask) == SEGMENT_REG_MS) {
        panic("GPU TLB cannot deal with non-memory addresses");
    }

    // Cannot deal with unprotected mode
    assert(((HandyM5Reg)tc->readMiscRegNoEffect(MISCREG_M5_REG)).prot);
    // must be in long mode
    assert(((HandyM5Reg)tc->readMiscRegNoEffect(MISCREG_M5_REG)).mode == LongMode);
    // Assuming 64-bit mode
    assert(((HandyM5Reg)tc->readMiscRegNoEffect(MISCREG_M5_REG)).submode == SixtyFourBitMode);
    // Paging better be enabled!
    assert(((HandyM5Reg)tc->readMiscRegNoEffect(MISCREG_M5_REG)).paging);

#endif

    Addr vaddr = req->getVaddr();
    DPRINTF(ShaderTLB, "Translating vaddr %#x.\n", vaddr);
    Addr offset = vaddr % TheISA::PageBytes;
    Addr vp_base = vaddr - offset;
    Addr pp_base;

    if (tlbMemory->lookup(vp_base, pp_base)) {
        DPRINTF(ShaderTLB, "TLB hit. Phys addr %#x.\n", pp_base + offset);
        hits++;
        req->setPaddr(pp_base + offset);
        translation->finish(NoFault, req, tc, mode);
    } else {
        // TLB miss! Let the TLB handle the walk, etc
        DPRINTF(ShaderTLB, "TLB miss for addr %#x\n", vaddr);
        misses++;
        translation->markDelayed();

        mmu->beginTLBMiss(this, translation, req, mode, tc);
    }
}

void
ShaderTLB::insert(Addr vp_base, Addr pp_base)
{
    tlbMemory->insert(vp_base, pp_base);
}

void
ShaderTLB::demapPage(Addr addr, uint64_t asn)
{
    DPRINTF(ShaderTLB, "Demapping %#x.\n", addr);
    panic("Demap addr unimplemented");
}

void
ShaderTLB::flushAll()
{
    panic("Flush all unimplemented");
}

bool
TLBMemory::lookup(Addr vp_base, Addr& pp_base, bool set_mru)
{
    int way = (vp_base / TheISA::PageBytes) % ways;
    for (int i=0; i < sets; i++) {
        if (entries[way][i].vpBase == vp_base && !entries[way][i].free) {
            pp_base = entries[way][i].ppBase;
            assert(entries[way][i].mruTick > 0);
            if (set_mru) {
                entries[way][i].setMRU();
            }
            entries[way][i].hits++;
            return true;
        }
    }
    pp_base = Addr(0);
    return false;
}

void
TLBMemory::insert(Addr vp_base, Addr pp_base)
{
    Addr a;
    if (lookup(vp_base, a)) {
        return;
    }
    int way = (vp_base / TheISA::PageBytes) % ways;
    GPUTlbEntry* entry = NULL;
    Tick minTick = curTick();
    for (int i=0; i < sets; i++) {
        if (entries[way][i].free) {
            entry = &entries[way][i];
            break;
        } else if (entries[way][i].mruTick <= minTick) {
            minTick = entries[way][i].mruTick;
            entry = &entries[way][i];
        }
    }
    assert(entry);
    if (!entry->free) {
        DPRINTF(ShaderTLB, "Evicting entry for vp %#x\n", entry->vpBase);
    }

    entry->vpBase = vp_base;
    entry->ppBase = pp_base;
    entry->free = false;
    entry->setMRU();
}

void
ShaderTLB::regStats()
{
    hits
        .name(name()+".hits")
        .desc("Number of hits in this TLB")
        ;
    misses
        .name(name()+".misses")
        .desc("Number of misses in this TLB")
        ;
    hitRate
        .name(name()+".hitRate")
        .desc("Hit rate for this TLB")
        ;

    hitRate = hits / (hits + misses);
}

ShaderTLB *
ShaderTLBParams::create()
{
    return new ShaderTLB(this);
}
