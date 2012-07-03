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

#ifndef __GPGPU_COPY_ENGINE_HH__
#define __GPGPU_COPY_ENGINE_HH__

#include "../gpgpu-sim/stream_manager.h"
#include "arch/types.hh"
#include "config/the_isa.hh"
#include "cpu/translation.hh"
#include "mem/ruby/system/RubyPort.hh"
#include "mem/mem_object.hh"
#include "params/SPACopyEngine.hh"
#include "params/ShaderCore.hh"
#include "sim/process.hh"
#include "sp_array.hh"

class SPACopyEngine : public MemObject
{
private:
    typedef SPACopyEngineParams Params;

    class CEPort : public MasterPort
    {
        friend class SPACopyEngine;

    private:
        SPACopyEngine *engine;

        /// holds packets that failed to send for retry
        PacketPtr outstandingPkt;

    public:
        CEPort(const std::string &_name, SPACopyEngine *_proc, int _idx)
        : MasterPort(_name, _proc), engine(_proc), idx(_idx)
        {
            outstandingPkt = NULL;
        }

        int idx;

    protected:
        virtual bool recvTimingResp(PacketPtr pkt);
        virtual void recvRetry();
        virtual Tick recvAtomic(PacketPtr pkt);
        virtual void recvFunctional(PacketPtr pkt);
    };

    CEPort port;

    class TickEvent : public Event
    {
        friend class SPACopyEngine;

    private:
        SPACopyEngine *ce;

    public:
        TickEvent(SPACopyEngine *c) : Event(CPU_Tick_Pri), ce(c) {}
        void process() { ce->tick(); }
        virtual const char *description() const { return "SPACopyEngine tick"; }
    };

    const Params * params() const { return dynamic_cast<const Params *>(_params);	}

    TickEvent tickEvent;
    MasterID masterId;

private:
    friend class StreamProcessorArray;

    StreamProcessorArray *spa;

    const SPACopyEngineParams *_params;

    void tick();

    int driverDelay;

    TheISA::TLB *dtb;
    TheISA::TLB *itb;

    ThreadContext *tc;

    bool stallOnRetry;
    bool needToRead;
    bool needToWrite;
    Addr currentReadAddr;
    Addr currentWriteAddr;
    Addr beginAddr;
    unsigned long long writeLeft;
    unsigned long long writeDone;
    unsigned long long readLeft;
    unsigned long long readDone;
    unsigned long long totalLength;

    uint8_t *curData;
    bool *readsDone;
    bool running;
    struct CUstream_st *stream;

    bool scheduledTickEvent;

    void initialize(ThreadContext *_tc, StreamProcessorArray* _spa) { tc = _tc; spa = _spa;}

    void tryRead();
    void tryWrite();

    bool sendPkt(PacketPtr pkt);

    unsigned long long memCpyStartTime;
    std::vector<unsigned long long> memCpyTimes;

public:

    SPACopyEngine(const Params *p);
    virtual MasterPort& getMasterPort(const std::string &if_name, int idx = -1);
    void finishTranslation(WholeTranslationState *state);

    int memcpy(Addr src, Addr dst, size_t length, struct CUstream_st *_stream);

    void cePrintStats(std::ostream& out);
};


class CEExitCallback : public Callback
{
private:
    std::string stats_filename;
    SPACopyEngine *ce_obj;

public:
    virtual ~CEExitCallback() {}

    CEExitCallback(SPACopyEngine *_ce_obj, const std::string& _stats_filename)
    {
        stats_filename = _stats_filename;
        ce_obj = _ce_obj;
    }

    virtual void process();
};

#endif
