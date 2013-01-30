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

#include "debug/ShaderTLB.hh"
#include "gpu/shader_tlb.hh"

ShaderTLB::ShaderTLB(const Params *p) :
    X86ISA::TLB(p), spa(p->spa), accessHostPageTable(p->access_host_pagetable)
{
}

void
ShaderTLB::unserialize(Checkpoint *cp, const std::string &section)
{
    // Intentionally left blank to keep from trying to read shader header from
    // checkpoint files. Allows for restore into any number of shader cores.
    // NOTE: Cannot checkpoint during kernels
}

void
ShaderTLB::beginTranslateTiming(RequestPtr req, BaseTLB::Translation *translation, BaseTLB::Mode mode)
{
    if (accessHostPageTable) {
        translateTiming(req, spa->getThreadContext(), translation, mode);
    } else {
        // The below code implements a perfect TLB with instant access to the
        // device page table.
        // TODO: We can shift this around, maybe to memory, maybe hierarchical TLBs
        Addr vaddr = req->getVaddr();
        Addr page_vaddr = spa->getGPUPageTable()->addrToPage(vaddr);
        Addr offset = vaddr - page_vaddr;
        Addr page_paddr;
        if (spa->getGPUPageTable()->lookup(page_vaddr, page_paddr)) {
            DPRINTF(ShaderTLB, "Translation found for vaddr %x = paddr %x\n", vaddr, page_paddr + offset);
            req->setPaddr(page_paddr + offset);
            translation->finish(NoFault, req, NULL, mode);
        } else {
            panic("ShaderTLB missing translation for vaddr %x!", vaddr);
        }
    }
}

ShaderTLB *
ShaderTLBParams::create()
{
    return new ShaderTLB(this);
}
