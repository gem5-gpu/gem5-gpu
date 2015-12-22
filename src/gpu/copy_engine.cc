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

#include "arch/utility.hh"
#include "base/output.hh"
#include "debug/GPUCopyEngine.hh"
#include "gpu/copy_engine.hh"
#include "mem/page_table.hh"
#include "params/GPUCopyEngine.hh"
#include "sim/system.hh"

using namespace std;

GPUCopyEngine::GPUCopyEngine(const Params *p) :
    MemObject(p), ceExitCB(this, p->stats_filename),
    hostPort(name() + ".hostPort", this, 0),
    devicePort(name() + ".devicePort", this, 0), readPort(NULL),
    writePort(NULL), tickEvent(this), masterId(p->sys->getMasterId(name())),
    cudaGPU(p->gpu), cacheLineSize(p->cache_line_size),
    driverDelay(p->driver_delay), hostDTB(p->host_dtb),
    deviceDTB(p->device_dtb), readDTB(NULL), writeDTB(NULL)
{
    DPRINTF(GPUCopyEngine, "Created copy engine\n");

    needToRead = false;
    needToWrite = false;
    running = false;

    registerExitCallback(&ceExitCB);

    cudaGPU->registerCopyEngine(this);

    bufferDepth = p->buffering * cacheLineSize;
}

Tick GPUCopyEngine::CEPort::recvAtomic(PacketPtr pkt)
{
    panic("GPUCopyEngine::CEPort::recvAtomic() not implemented!\n");
    return 0;
}

void GPUCopyEngine::CEPort::recvFunctional(PacketPtr pkt)
{
    panic("GPUCopyEngine::CEPort::recvFunctional() not implemented!\n");
}

bool GPUCopyEngine::CEPort::recvTimingResp(PacketPtr pkt)
{
    engine->recvPacket(pkt);
    return true;
}

void GPUCopyEngine::CEPort::recvReqRetry() {
    assert(outstandingPkts.size());

    DPRINTF(GPUCopyEngine, "Got a retry...\n");
    while (outstandingPkts.size() && sendTimingReq(outstandingPkts.front())) {
        DPRINTF(GPUCopyEngine, "Unblocked, sent blocked packet.\n");
        outstandingPkts.pop();
        // TODO: This should just signal the engine that the packet completed
        // engine should schedule tick as necessary. Need a test case
        if (!engine->tickEvent.scheduled()) {
            engine->schedule(engine->tickEvent, engine->nextCycle());
        }
    }
}

void GPUCopyEngine::CEPort::sendPacket(PacketPtr pkt) {
    if (isStalled() || !sendTimingReq(pkt)) {
        DPRINTF(GPUCopyEngine, "sendTiming failed in sendPacket(pkt->req->getVaddr()=0x%x)\n", (unsigned int)pkt->req->getVaddr());
        setStalled(pkt);
    }
}

void GPUCopyEngine::finishMemcpy()
{
    running = false;
    readPort = writePort = NULL;
    readDTB = writeDTB = NULL;
    Tick total_time = curTick() - memCpyStartTime;
    numOperations++;
    operationTimeTicks += total_time;
    DPRINTF(GPUCopyEngine, "Total time was: %llu\n", total_time);
    memCpyStats.push_back(MemCpyStats(total_time, memCpyLength));
    cudaGPU->finishCopyOperation();
}

void GPUCopyEngine::recvPacket(PacketPtr pkt)
{
    if (pkt->isRead()) {
        DPRINTF(GPUCopyEngine, "done with a read addr: 0x%x, size: %d\n", pkt->req->getVaddr(), pkt->getSize());
        pkt->writeData(curData + (pkt->req->getVaddr() - beginAddr));
        bytesRead += pkt->getSize();

        // set the addresses we just got as done
        for (int i = pkt->req->getVaddr() - beginAddr;
                i < pkt->req->getVaddr() - beginAddr + pkt->getSize(); i++) {
            readsDone[i] = true;
        }

        DPRINTF(GPUCopyEngine, "Data is: %d\n", *((int*) (curData + (pkt->req->getVaddr() - beginAddr))));
        if (readDone < totalLength) {
            DPRINTF(GPUCopyEngine, "Trying to write\n");
            needToWrite = true;
            if (!tickEvent.scheduled()) {
                schedule(tickEvent, nextCycle());
            }
        }

        // mark readDone as only the contiguous region
        while (readDone < totalLength && readsDone[readDone]) {
            readDone++;
        }

        if (readDone >= totalLength) {
            DPRINTF(GPUCopyEngine, "done reading!!\n");
            needToRead = false;
        }
    } else {
        DPRINTF(GPUCopyEngine, "done with a write addr: 0x%x\n", pkt->req->getVaddr());
        writeDone += pkt->getSize();
        bytesWritten += pkt->getSize();
        if (!(writeDone < totalLength)) {
            // we are done!
            DPRINTF(GPUCopyEngine, "done writing, completely done!!!!\n");
            needToWrite = false;
            delete[] curData;
            delete[] readsDone;
            finishMemcpy();
        } else {
            if (!tickEvent.scheduled()) {
                schedule(tickEvent, nextCycle());
            }
        }
    }
    if (pkt->req) delete pkt->req;
    delete pkt;
}

void GPUCopyEngine::tryRead()
{
    RequestPtr req = new Request();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;
    //unsigned block_size = port.peerBlockSize();

    if (readLeft <= 0) {
        DPRINTF(GPUCopyEngine, "WHY ARE WE HERE?\n");
        return;
    }

    int size;
    if (currentReadAddr % cacheLineSize) {
        size = cacheLineSize - (currentReadAddr % cacheLineSize);
        DPRINTF(GPUCopyEngine, "Aligning\n");
    } else {
        size = cacheLineSize;
    }
    size = readLeft > (size - 1) ? size : readLeft;
    req->setVirt(asid, currentReadAddr, size, flags, masterId, pc);

    DPRINTF(GPUCopyEngine, "trying read addr: 0x%x, %d bytes\n", currentReadAddr, size);

    BaseTLB::Mode mode = BaseTLB::Read;

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
    DataTranslation<GPUCopyEngine*> *translation
            = new DataTranslation<GPUCopyEngine*>(this, state);

    readDTB->beginTranslateTiming(req, translation, mode);

    currentReadAddr += size;

    readLeft -= size;

    if (!(readLeft > 0)) {
        needToRead = false;
        if (!tickEvent.scheduled()) {
            schedule(tickEvent, nextCycle());
        }
    } else {
        if (!readPort->isStalled() && !tickEvent.scheduled()) {
            schedule(tickEvent, nextCycle());
        }
    }
}

void GPUCopyEngine::tryWrite()
{
    if (writeLeft <= 0) {
        DPRINTF(GPUCopyEngine, "WHY ARE WE HERE (write)?\n");
        return;
    }

    int size;
    if (currentWriteAddr % cacheLineSize) {
        size = cacheLineSize - (currentWriteAddr % cacheLineSize);
        DPRINTF(GPUCopyEngine, "Aligning\n");
    } else {
        size = cacheLineSize;
    }
    size = writeLeft > size-1 ? size : writeLeft;

    if (readDone < size+(totalLength-writeLeft)) {
        // haven't read enough yet
        DPRINTF(GPUCopyEngine, "Tried to write when we haven't read enough\n");
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

    DPRINTF(GPUCopyEngine, "trying write addr: 0x%x, %d bytes, data %d\n", currentWriteAddr, size, *((int*)(&curData[totalLength-writeLeft])));

    BaseTLB::Mode mode = BaseTLB::Write;

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
    DataTranslation<GPUCopyEngine*> *translation
            = new DataTranslation<GPUCopyEngine*>(this, state);

    writeDTB->beginTranslateTiming(req, translation, mode);

    currentWriteAddr += size;

    writeLeft -= size;

    if (!(writeLeft > 0) && !tickEvent.scheduled()) {
        schedule(tickEvent, nextCycle());
    }
}

bool GPUCopyEngine::buffersFull() {
    unsigned amount_buffered = readDone - (totalLength - writeLeft);
    return (bufferDepth > 0) && (amount_buffered > bufferDepth);
}

void GPUCopyEngine::tick()
{
    if (!running) return;
    if (readPort->isStalled() && writePort->isStalled()) {
        DPRINTF(GPUCopyEngine, "Stalled\n");
    } else {
        if (needToRead && !readPort->isStalled() && !buffersFull()) {
            DPRINTF(GPUCopyEngine, "trying read\n");
            tryRead();
        }
        if (needToWrite && !writePort->isStalled() && ((totalLength - writeLeft) < readDone)) {
            DPRINTF(GPUCopyEngine, "trying write\n");
            tryWrite();
        }
    }
}

int GPUCopyEngine::memcpy(Addr src, Addr dst, size_t length, stream_operation_type type)
{
    switch (type) {
    case stream_memcpy_host_to_device:
        readPort = &hostPort;
        readDTB = hostDTB;
        writePort = &devicePort;
        writeDTB = deviceDTB;
        break;
    case stream_memcpy_device_to_host:
        readPort = &devicePort;
        readDTB = deviceDTB;
        writePort = &hostPort;
        writeDTB = hostDTB;
        break;
    case stream_memcpy_device_to_device:
        readPort = &devicePort;
        readDTB = deviceDTB;
        writePort = &devicePort;
        writeDTB = deviceDTB;
        break;
    default:
        panic("Unknown stream memcpy type: %d!\n", type);
        break;
    }

    assert(length > 0);
    memCpyLength = length;
    assert(!running);
    running = true;

    DPRINTF(GPUCopyEngine, "Initiating copy of %d bytes from 0x%x to 0x%x\n", length, src, dst);
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

    if (!tickEvent.scheduled()) {
        schedule(tickEvent, nextCycle() + driverDelay);
    }

    return 0;
}

int GPUCopyEngine::memset(Addr dst, int value, size_t length)
{
    assert(!running && !readPort && !readDTB);
    readPort = &hostPort;
    readDTB = hostDTB;
    writePort = &devicePort;
    writeDTB = deviceDTB;

    assert(length > 0);
    running = true;

    DPRINTF(GPUCopyEngine, "Initiating memset of %d bytes at 0x%x to %d\n", length, dst, value);
    memCpyStartTime = curTick();
    memCpyLength = length;

    needToRead = false;
    needToWrite = true;

    currentWriteAddr = dst;

    readLeft = 0;
    writeLeft = length;

    totalLength = length;

    readDone = length;
    writeDone = 0;

    curData = new uint8_t[length];
    readsDone = new bool[length];
    for (int i = 0; i < length; i++) {
        curData[i] = value;
        readsDone[i] = true;
    }

    if (!tickEvent.scheduled()) {
        schedule(tickEvent, nextCycle() + driverDelay);
    }

    return 0;
}

void GPUCopyEngine::finishTranslation(WholeTranslationState *state)
{
    if (state->getFault() != NoFault) {
        panic("Translation encountered fault (%s) for address 0x%x", state->getFault()->name(), state->mainReq->getVaddr());
    }
    DPRINTF(GPUCopyEngine, "Finished translation of Vaddr 0x%x -> Paddr 0x%x\n", state->mainReq->getVaddr(), state->mainReq->getPaddr());
    PacketPtr pkt;
    if (state->mode == BaseTLB::Read) {
        pkt = new Packet(state->mainReq, MemCmd::ReadReq);
        pkt->allocate();
        readPort->sendPacket(pkt);
    } else if (state->mode == BaseTLB::Write) {
        pkt = new Packet(state->mainReq, MemCmd::WriteReq);
        uint8_t *pkt_data = (uint8_t *)state->mainReq->getExtraData();
        pkt->dataDynamic(pkt_data);
        writePort->sendPacket(pkt);
    } else {
        panic("Finished translation of unknown mode: %d\n", state->mode);
    }
    delete state;
}

BaseMasterPort&
GPUCopyEngine::getMasterPort(const std::string &if_name, PortID idx)
{
    if (if_name == "host_port")
        return hostPort;
    else if (if_name == "device_port")
        return devicePort;
    else
        return MemObject::getMasterPort(if_name, idx);
}

void GPUCopyEngine::regStats() {
    numOperations
        .name(name() + ".numOperations")
        .desc("Number of copy/memset operations")
        ;
    bytesRead
        .name(name() + ".opBytesRead")
        .desc("Number of copy bytes read")
        ;
    bytesWritten
        .name(name() + ".opBytesWritten")
        .desc("Number of copy/memset bytes written")
        ;
    operationTimeTicks
        .name(name() + ".opTimeTicks")
        .desc("Total time spent in copy/memset operations")
        ;
}

GPUCopyEngine *GPUCopyEngineParams::create() {
    return new GPUCopyEngine(this);
}

void GPUCopyEngine::cePrintStats(std::ostream& out) {
    int memcpy_cnt = 0;
    Tick total_memcpy_ticks = 0;
    Tick total_memcpy_bytes = 0;
    Tick max_memcpy_ticks = 0;
    Tick min_memcpy_ticks = ULONG_LONG_MAX;
    vector<MemCpyStats>::iterator it;

    out << "copy engine frequency: " << frequency()/(1000000000.0) << " GHz\n";
    out << "copy engine period: " << clockPeriod() << " ticks\n";

    out << "memcpy times in ticks:\n";
    for (it = memCpyStats.begin(); it < memCpyStats.end(); it++) {
        out << (*it).ticks << ", ";
        memcpy_cnt++;
        total_memcpy_ticks += (*it).ticks;
        if ((*it).ticks < min_memcpy_ticks) min_memcpy_ticks = (*it).ticks;
        if ((*it).ticks > max_memcpy_ticks) max_memcpy_ticks = (*it).ticks;
    }
    out << "\n";
    Tick average_memcpy_ticks;
    if (memcpy_cnt > 0) {
        average_memcpy_ticks = total_memcpy_ticks / memcpy_cnt;
    } else {
        average_memcpy_ticks = 0;
    }

    out << "memcpy sizes in bytes:\n";
    for (it = memCpyStats.begin(); it < memCpyStats.end(); it++) {
        size_t num_bytes = (*it).bytes;
        out << num_bytes << ", ";
        total_memcpy_bytes += num_bytes;
    }
    out << "\n";

    out << "average ticks per memcpy = " << average_memcpy_ticks << "\n";
    out << "minimum ticks per memcpy = " << min_memcpy_ticks << "\n";
    out << "maximum ticks per memcpy = " << max_memcpy_ticks << "\n";
    out << "total memcpy ticks = " << total_memcpy_ticks << "\n";
    out << "total memcpy bytes = " << total_memcpy_bytes << "\n";
    out << "\n";
}

void GPUCopyEngine::CEExitCallback::process()
{
    std::ostream *os = simout.find(statsFilename);
    if (!os) {
        os = simout.create(statsFilename);
    }
    engine->cePrintStats(*os);
    *os << std::endl;
}
