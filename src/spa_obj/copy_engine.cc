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
        MemObject(p),  port(name() + ".cePort", this, 0), tickEvent(this),
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

void SPACopyEngine::CEPort::recvFunctional(PacketPtr pkt)
{
    panic("SPACopyEngine::CEPort::recvFunctional() not implemented!\n");
}

bool SPACopyEngine::CEPort::recvTimingResp(PacketPtr pkt)
{
    // packet done

    if (pkt->isRead()) {
        DPRINTF(SPACopyEngine, "done with a read addr: 0x%x, size: %d\n", pkt->req->getVaddr(), pkt->getSize());
        pkt->writeData(engine->curData + (pkt->req->getVaddr() - engine->beginAddr));

        // set the addresses we just got as done
        for (int i = pkt->req->getVaddr() - engine->beginAddr;
                i < pkt->req->getVaddr() - engine->beginAddr + pkt->getSize(); i++) {
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
        while (engine->readDone < engine->totalLength && engine->readsDone[engine->readDone]) {
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
            delete[] engine->curData;
            delete[] engine->readsDone;
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
    if (pkt->req) delete pkt->req;
    delete pkt;
    return true;
}

void SPACopyEngine::CEPort::recvRetry() {
    assert(outstandingPkt != NULL);

    DPRINTF(SPACopyEngine, "Got a retry...\n");
    if(sendTimingReq(outstandingPkt)) {
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
    size = readLeft > (size - 1) ? size : readLeft;
    req->setVirt(asid, currentReadAddr, size, flags, masterId, pc);

    DPRINTF(SPACopyEngine, "trying read addr: 0x%x, %d bytes\n", currentReadAddr, size);

    BaseTLB::Mode mode = BaseTLB::Read;

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
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

    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;
    req->setVirt(asid, currentWriteAddr, size, flags, masterId, pc);

    assert(	(totalLength-writeLeft +size) <= readDone);
    uint8_t *data = new uint8_t[size];
    std::memcpy(data, &curData[totalLength-writeLeft], size);
    req->setExtraData((uint64_t)data);

    DPRINTF(SPACopyEngine, "trying write addr: 0x%x, %d bytes, data %d\n", currentWriteAddr, size, *((int*)(&curData[totalLength-writeLeft])));

    BaseTLB::Mode mode = BaseTLB::Write;

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
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

    assert(length > 0);
    assert(!running);
    running = true;

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
    for (int i = 0; i < length; i++) {
        curData[i] = 0;
        readsDone[i] = false;
    }

    schedule(tickEvent, curTick()+driverDelay);

    return 0;
}


void SPACopyEngine::finishTranslation(WholeTranslationState *state)
{
    if (state->getFault() != NoFault) {
        panic("Translation encountered fault (%s) for address 0x%x", state->getFault()->name(), state->mainReq->getVaddr());
    }
    DPRINTF(SPACopyEngine, "Finished translation of Vaddr 0x%x -> Paddr 0x%x\n", state->mainReq->getVaddr(), state->mainReq->getPaddr());
    PacketPtr pkt;
    if (state->mode == BaseTLB::Read) {
        pkt = new Packet(state->mainReq, MemCmd::ReadReq);
        pkt->allocate();
    } else if (state->mode == BaseTLB::Write) {
        pkt = new Packet(state->mainReq, MemCmd::WriteReq);
        uint8_t *pkt_data = (uint8_t *)state->mainReq->getExtraData();
        pkt->dataDynamicArray(pkt_data);
    } else {
        panic("Finished translation of unknown mode: %d\n", state->mode);
    }
    if (!sendPkt(pkt)) {
        stallOnRetry = true;
    }
    delete state;
}

bool SPACopyEngine::sendPkt(PacketPtr pkt) {
    if (!port.sendTimingReq(pkt)) {
        DPRINTF(SPACopyEngine, "sendTiming failed in sendPkt (pkt->req->getVaddr()=0x%x)\n", (unsigned int)pkt->req->getVaddr());
        port.outstandingPkt = pkt;
        return false;
    }
    return true;
}

MasterPort&
SPACopyEngine::getMasterPort(const std::string &if_name, int idx)
{
    return port;
}


SPACopyEngine *SPACopyEngineParams::create() {
    return new SPACopyEngine(this);
}


void SPACopyEngine::cePrintStats(std::ostream& out) {
    int i = 0;
    unsigned long long total_memcpy_ticks = 0;
    unsigned long long max_memcpy_ticks = 0;
    unsigned long long min_memcpy_ticks = ULONG_LONG_MAX;
    vector<unsigned long long>::iterator it;
    out << "memcpy times in ticks:\n";
    for (it = memCpyTimes.begin(); it < memCpyTimes.end(); it++) {
        out << *it << ", ";
        i++;
        total_memcpy_ticks += *it;
        if (*it < min_memcpy_ticks) min_memcpy_ticks = *it;
        if (*it > max_memcpy_ticks) max_memcpy_ticks = *it;
    }
    out << "\n";
    out << "total memcpy ticks = " << total_memcpy_ticks << "\n";
    unsigned long long int average_memcpy_ticks = total_memcpy_ticks / i;
    out << "average ticks per memcpy = " << average_memcpy_ticks << "\n";
    out << "minimum ticks per memcpy = " << min_memcpy_ticks << "\n";
    out << "maximum ticks per memcpy = " << max_memcpy_ticks << "\n";
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
