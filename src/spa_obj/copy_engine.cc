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

#include <iostream>

#include "arch/tlb.hh"
#include "arch/utility.hh"
#include "base/output.hh"
#include "config/the_isa.hh"
#include "debug/SPACopyEngine.hh"
#include "mem/page_table.hh"
#include "params/SPACopyEngine.hh"
#include "copy_engine.hh"

#define READ_AMOUNT 128

using namespace TheISA;
using namespace std;

SPACopyEngine::SPACopyEngine(const Params *p) :
        MemObject(p),  port("cePort", this, 0), tickEvent(this),
        masterId(p->sys->getMasterId(name())), _params(p),
        driverDelay(p->driverDelay), dtb(p->dtb), itb(p->itb)
{
    DPRINTF(SPACopyEngine, "Created copy engine\n");

    stallOnRetry = false;
    needToRead = false;
    needToWrite = false;
    scheduledTickEvent = false;
    running = false;

    CEExitCallback* ceExitCB = new CEExitCallback(this, p->stats_filename);
    registerExitCallback(ceExitCB);
}

Tick SPACopyEngine::CEPort::recvAtomic(PacketPtr pkt)
{
    panic("SPACopyEngine::CEPort::recvAtomic() not implemented!\n");
    return 0;
}


bool SPACopyEngine::CEPort::recvTiming(PacketPtr pkt)
{
    // packet done

    if (pkt->isRead()) {
        DPRINTF(SPACopyEngine, "done with a read addr: 0x%x, size: %d\n", pkt->req->getVaddr(), pkt->getSize());
        pkt->writeData(engine->curData + (pkt->req->getVaddr() - engine->beginAddr));

        // set the addresses we just got as done
        for (int i=pkt->req->getVaddr() - engine->beginAddr;
                i<pkt->req->getVaddr() - engine->beginAddr + pkt->getSize(); i++) {
            engine->readsDone[i] = true;
        }

        DPRINTF(SPACopyEngine, "Data is: %d\n", *((int*) (engine->curData + (pkt->req->getVaddr() - engine->beginAddr))));
        if (engine->readDone < engine->totalLength) {
            DPRINTF(SPACopyEngine, "Trying to write\n");
            engine->needToWrite = true;
            if (!engine->scheduledTickEvent) {
                engine->scheduledTickEvent = true;
                engine->schedule(engine->tickEvent, curTick()+1);
            }
        }

        // mark readDone as only the contiguous region
        while (engine->readsDone[engine->readDone]) {
            engine->readDone++;
        }

        if (engine->readDone >= engine->totalLength) {
            DPRINTF(SPACopyEngine, "done reading!!\n");
            engine->needToRead = false;
        }
    } else {
        DPRINTF(SPACopyEngine, "done with a write addr: 0x%x\n", pkt->req->getVaddr());
        engine->writeDone += pkt->getSize();
        if (!(engine->writeDone < engine->totalLength)) {
            // we are done!
            DPRINTF(SPACopyEngine, "done writing, completely done!!!!\n");
            engine->needToWrite = false;
            delete engine->curData;
            // unblock the cpu
            engine->running = false;
            engine->stream->record_next_done();
            DPRINTF(SPACopyEngine, "Total time was: %llu\n", curTick() - engine->memCpyStartTime);
            engine->memCpyTimes.push_back(curTick() - engine->memCpyStartTime);
            engine->spa->streamRequestTick(1);
            engine->tc->activate();
        } else {
            if (!engine->scheduledTickEvent) {
                engine->scheduledTickEvent = true;
                engine->schedule(engine->tickEvent, curTick()+1);
            }
        }
    }

    return true;
}

void SPACopyEngine::CEPort::recvRetry() {
    assert(outstandingPkt != NULL);

    DPRINTF(SPACopyEngine, "Got a retry...\n");
    if(sendTiming(outstandingPkt)) {
        DPRINTF(SPACopyEngine, "unblocked moving on.\n");
        outstandingPkt = NULL;
        engine->stallOnRetry = false;
        if (!engine->scheduledTickEvent) {
            engine->scheduledTickEvent = true;
            engine->schedule(engine->tickEvent, curTick()+1);
        }
    } else {
        //DPRINTF(SPACopyEngine, "Still blocked\n");
    }
}

void SPACopyEngine::tryRead()
{
    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;
    //unsigned block_size = port.peerBlockSize();

    if (readLeft <= 0) {
        DPRINTF(SPACopyEngine, "WHY ARE WE HERE?\n");
        return;
    }

    int size;
    if (currentReadAddr % READ_AMOUNT) {
        size = READ_AMOUNT - (currentReadAddr % READ_AMOUNT);
        DPRINTF(SPACopyEngine, "Aligning\n");
    } else {
        size = READ_AMOUNT;
    }
    size = readLeft > size-1 ? size : readLeft;
    req->setVirt(asid, currentReadAddr, size, flags, masterId, pc);

    DPRINTF(SPACopyEngine, "trying read addr: 0x%x, %d bytes\n", currentReadAddr, size);

    BaseTLB::Mode mode = BaseTLB::Read;

    WholeTranslationState *state =
            new WholeTranslationState(req, new uint8_t[req->getSize()], NULL, mode);
    DataTranslation<SPACopyEngine*> *translation
            = new DataTranslation<SPACopyEngine*>(this, state);

    dtb->translateTiming(req, tc, translation, mode);

    currentReadAddr += size;

    readLeft -= size;

    if (!(readLeft > 0) && !scheduledTickEvent) {
        scheduledTickEvent = true;
        schedule(tickEvent, curTick()+1);
    }
    if (!(readLeft > 0)) {
        needToRead = false;
    }
}

void SPACopyEngine::tryWrite()
{
    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;
    //unsigned block_size = port.peerBlockSize();

    if (writeLeft <= 0) {
        DPRINTF(SPACopyEngine, "WHY ARE WE HERE (write)?\n");
        return;
    }

    int size;
    if (currentWriteAddr % READ_AMOUNT) {
        size = READ_AMOUNT - (currentWriteAddr % READ_AMOUNT);
        DPRINTF(SPACopyEngine, "Aligning\n");
    } else {
        size = READ_AMOUNT;
    }
    size = writeLeft > size-1 ? size : writeLeft;

    if (readDone < size+(totalLength-writeLeft)) {
        // haven't read enough yet
        DPRINTF(SPACopyEngine, "Tried to write when we haven't read enough\n");
        return;
    }

    req->setVirt(asid, currentWriteAddr, size, flags, masterId, pc);

    assert(	(totalLength-writeLeft +size) <= readDone);
    uint8_t *data = new uint8_t[size];
    std::memcpy(data, &curData[totalLength-writeLeft], size);
    req->setExtraData((uint64_t)data);

    DPRINTF(SPACopyEngine, "trying write addr: 0x%x, %d bytes, data %d\n", currentWriteAddr, size, *((int*)(&curData[totalLength-writeLeft])));

    BaseTLB::Mode mode = BaseTLB::Write;

    WholeTranslationState *state =
            new WholeTranslationState(req, new uint8_t[req->getSize()], NULL, mode);
    DataTranslation<SPACopyEngine*> *translation
            = new DataTranslation<SPACopyEngine*>(this, state);

    dtb->translateTiming(req, tc, translation, mode);

    currentWriteAddr += size;

    writeLeft -= size;

    if (!(writeLeft > 0) && !scheduledTickEvent) {
        scheduledTickEvent = true;
        schedule(tickEvent, curTick()+1);
    }
}

void SPACopyEngine::tick()
{
    scheduledTickEvent = false;
    if (stallOnRetry) {
        DPRINTF(SPACopyEngine, "Stalled\n");
    }

    if (needToRead && !stallOnRetry) {
        DPRINTF(SPACopyEngine, "trying read\n");
        tryRead();
    }

    if (needToWrite && !stallOnRetry && ((totalLength-writeLeft) < readDone)) {
        DPRINTF(SPACopyEngine, "trying write\n");
        tryWrite();
    }
}


int SPACopyEngine::memcpy(Addr src, Addr dst, size_t length, struct CUstream_st *_stream)
{
    stream = _stream;

    assert(!running);
    running = true;
    if(length > 0) {
        DPRINTF(SPACopyEngine, "Initiating copy of %d bytes from 0x%x to 0x%x\n", length, src, dst);
        memCpyStartTime = curTick();

        needToRead = true;
        needToWrite = false;

        currentReadAddr = src;
        currentWriteAddr = dst;

        beginAddr = src;

        readLeft = length;
        writeLeft = length;

        totalLength = length;

        readDone = 0;
        writeDone = 0;

        curData = new uint8_t[length];
        readsDone = new bool[length];

        schedule(tickEvent, curTick()+driverDelay);
    }

    return 0;
}


void SPACopyEngine::finishTranslation(WholeTranslationState *state)
{
    DPRINTF(SPACopyEngine, "Finished translation of addr 0x%x...\n", state->mainReq->getVaddr());
    if (state->mode == BaseTLB::Read) {
        PacketPtr pkt = new Packet(state->mainReq, MemCmd::ReadReq, Packet::Broadcast);
        pkt->allocate();
        if (!sendPkt(pkt)) {
            stallOnRetry = true;
        }
    } else {
        PacketPtr pkt = new Packet(state->mainReq, MemCmd::WriteReq, Packet::Broadcast);
        uint8_t *pkt_data = (uint8_t *)state->mainReq->getExtraData();
        pkt->dataDynamicArray(pkt_data);
        if (!sendPkt(pkt)) {
            stallOnRetry = true;
        }
    }
}

bool SPACopyEngine::sendPkt(PacketPtr pkt) {
    if (!port.sendTiming(pkt)) {
        DPRINTF(SPACopyEngine, "sendTiming failed in sendPkt (pkt->req->getVaddr()=0x%x)\n", (unsigned int)pkt->req->getVaddr());
        port.outstandingPkt = pkt;
        return false;
    }
    return true;
}


Port *SPACopyEngine::getPort(const std::string &if_name, int idx)
{
    return &port;
}


SPACopyEngine *SPACopyEngineParams::create() {
    return new SPACopyEngine(this);
}


void SPACopyEngine::cePrintStats(std::ostream& out) {
    int i = 0;
    unsigned long long totalMemCpyTime=0;
    vector<unsigned long long>::iterator it;
    for ( it=memCpyTimes.begin() ; it < memCpyTimes.end(); it++ ) {
        cout << "memcpy[" << i << "] time = " << *it << "\n";
        out << "memcpy[" << i << "] time = " << *it << "\n";
        i++;
        totalMemCpyTime += *it;
    }
    cout << "total memcpy time = " << totalMemCpyTime << "\n";
    out << "total memcpy time = " << totalMemCpyTime << "\n";
}

void CEExitCallback::process()
{
    std::ostream *os = simout.find(stats_filename);
    if (!os) {
        os = simout.create(stats_filename);
    }
    ce_obj->cePrintStats(*os);
    *os << std::endl;
}
