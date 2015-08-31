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

#ifndef SHADER_TLB_HH_
#define SHADER_TLB_HH_

#include <map>
#include <set>

#include "base/statistics.hh"
#include "params/ShaderTLB.hh"
#include "arch/generic/tlb.hh"

class ShaderMMU;
class CudaGPU;

class GPUTlbEntry {
public:
    Addr vpBase;
    Addr ppBase;
    bool free;
    Tick mruTick;
    uint32_t hits;
    GPUTlbEntry() : vpBase(0), ppBase(0), free(true), mruTick(0), hits(0) {}
    void setMRU() { mruTick = curTick(); }
};

class BaseTLBMemory {
public:
    virtual bool lookup(Addr vp_base, Addr& pp_base, bool set_mru=true) = 0;
    virtual void insert(Addr vp_base, Addr pp_base) = 0;
};

class TLBMemory : public BaseTLBMemory {
    int numEntries;
    int sets;
    int ways;

    GPUTlbEntry **entries;

protected:
    TLBMemory() {}

public:
    TLBMemory(int _numEntries, int associativity) :
        numEntries(_numEntries), sets(associativity)
    {
        if (sets == 0) {
            sets = numEntries;
        }
        assert(numEntries % sets == 0);
        ways = numEntries/sets;
        entries = new GPUTlbEntry*[ways];
        for (int i=0; i < ways; i++) {
            entries[i] = new GPUTlbEntry[sets];
        }
    }
    virtual ~TLBMemory()
    {
        for (int i=0; i < sets; i++) {
            delete[] entries[i];
        }
        delete[] entries;
    }

    virtual bool lookup(Addr vp_base, Addr& pp_base, bool set_mru=true);
    virtual void insert(Addr vp_base, Addr pp_base);
};

class InfiniteTLBMemory : public BaseTLBMemory {
    std::map<Addr, Addr> entries;
public:
    InfiniteTLBMemory() {}
    ~InfiniteTLBMemory() {}

    bool lookup(Addr vp_base, Addr& pp_base, bool set_mru=true)
    {
        auto it = entries.find(vp_base);
        if (it != entries.end()) {
            pp_base = it->second;
            return true;
        } else {
            pp_base = Addr(0);
            return false;
        }
    }
    void insert(Addr vp_base, Addr pp_base)
    {
        entries[vp_base] = pp_base;
    }
};

class ShaderTLB : public BaseTLB
{
private:
    unsigned numEntries;

    Cycles hitLatency;

    // Pointer to the SPA to access the page table
    CudaGPU* cudaGPU;
    bool accessHostPageTable;

    BaseTLBMemory *tlbMemory;

    void translateTiming(RequestPtr req, ThreadContext *tc,
                         Translation *translation, Mode mode);

    ShaderMMU *mmu;

public:
    typedef ShaderTLBParams Params;
    ShaderTLB(const Params *p);

    // For checkpoint restore (empty unserialize)
    virtual void unserialize(CheckpointIn &cp);

    void beginTranslateTiming(RequestPtr req, BaseTLB::Translation *translation,
                              BaseTLB::Mode mode);

    void finishTranslation(Fault fault, RequestPtr req, ThreadContext *tc,
                           Mode mode, Translation* origTranslation);

    void demapPage(Addr addr, uint64_t asn);
    void flushAll();

    void takeOverFrom(BaseTLB *_tlb) {}

    void insert(Addr vp_base, Addr pp_base);

    void regStats();

    Stats::Scalar hits;
    Stats::Scalar misses;
    Stats::Formula hitRate;
};

#endif /* SHADER_TLB_HH_ */
