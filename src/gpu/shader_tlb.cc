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

#include "arch/x86/insts/microldstop.hh"
#include "arch/x86/regs/misc.hh"
#include "arch/x86/regs/msr.hh"
#include "arch/x86/faults.hh"
#include "arch/x86/pagetable_walker.hh"
#include "debug/ShaderTLB.hh"
#include "gpu/shader_tlb.hh"

using namespace std;
using namespace X86ISA;

ShaderTLB::ShaderTLB(const Params *p) :
    BaseTLB(p), x86tlb(p->x86tlb), numEntries(p->entries),
    hitLatency(p->hit_latency), cudaGPU(p->gpu),
    accessHostPageTable(p->access_host_pagetable)
{
    if (numEntries > 0) {
        tlbMemory = new TLBMemory(p->entries, p->associativity);
    } else {
        tlbMemory = new InfiniteTLBMemory();
    }
    mmu = cudaGPU->getMMU();
}

void
ShaderTLB::unserialize(Checkpoint *cp, const std::string &section)
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
    uint32_t flags = req->getFlags();

    // If this is true, we're dealing with a request to a non-memory address
    // space.
    if ((flags & SegmentFlagMask) == SEGMENT_REG_MS) {
        panic("GPU TLB cannot deal with non-memory addresses");
    }

    Addr vaddr = req->getVaddr();
    DPRINTF(ShaderTLB, "Translating vaddr %#x.\n", vaddr);

    HandyM5Reg m5Reg = tc->readMiscRegNoEffect(MISCREG_M5_REG);

    assert(m5Reg.prot); // Cannot deal with unprotected mode
    assert(m5Reg.mode == LongMode); // must be in long mode
    assert(m5Reg.submode == SixtyFourBitMode); // Assuming 64-bit mode
    assert(m5Reg.paging); // Paging better be enabled!

    Addr offset = vaddr % TheISA::PageBytes;
    Addr vpn = vaddr - offset;
    Addr ppn;

    if (tlbMemory->lookup(vpn, ppn)) {
        DPRINTF(ShaderTLB, "TLB hit. Phys addr %#x.\n", ppn + offset);
        hits++;
        req->setPaddr(ppn + offset);
        translation->finish(NoFault, req, tc, mode);
    } else {
        // TLB miss! Let the x86 TLB handle the walk, etc
        DPRINTF(ShaderTLB, "TLB miss for addr %#x\n", vaddr);
        misses++;
        translation->markDelayed();

        mmu->handleTLBMiss(x86tlb, this, translation, req, mode, tc);
    }
}

void
ShaderTLB::insert(Addr vpn, Addr ppn)
{
    tlbMemory->insert(vpn, ppn);
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
TLBMemory::lookup(Addr vpn, Addr& ppn)
{
    int way = (vpn / TheISA::PageBytes) % ways;
    for (int i=0; i < sets; i++) {
        if (entries[way][i].vpn == vpn && !entries[way][i].free) {
            ppn = entries[way][i].ppn;
            assert(entries[way][i].mruTick > 0);
            entries[way][i].setMRU();
            return true;
        }
    }
    ppn = Addr(0);
    return false;
}

void
TLBMemory::insert(Addr vpn, Addr ppn)
{
    Addr a;
    if (lookup(vpn, a)) {
        return;
    }
    int way = (vpn / TheISA::PageBytes) % ways;
    TLBEntry* entry = NULL;
    Tick minTick = curTick();
    for (int i=0; i < sets; i++) {
        if (entries[way][i].free) {
            entry = &entries[way][i];
            break;
        } else if (entries[way][i].mruTick < minTick) {
            minTick = entries[way][i].mruTick;
            entry = &entries[way][i];
        }
    }
    assert(entry);
    if (!entry->free) {
        DPRINTF(ShaderTLB, "Evicting entry for vpn %#x\n", entry->vpn);
    }

    entry->vpn = vpn;
    entry->ppn = ppn;
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
