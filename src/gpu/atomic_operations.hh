#ifndef __ATOMIC_OPERATIONS_HH__
#define __ATOMIC_OPERATIONS_HH__

#include "base/misc.hh"
#include "base/types.hh"
#include "mem/simple_mem.hh"

/**
 * A class for representing GPU atomic operations requested by each GPU core
 * lane. Each instance contains the parameters of an atomic memory operation
 * sent to the gem5 memory hierarchy. This class implements a thin wrapper on
 * Ruby's physical memory access in the access hit callback. It performs atomic
 * operations on the data stored in the Ruby physical memory, updating data as
 * appropriate. Note: a CoalescedAccess may contain multiple AtomicRequests.
 *
 * This class implements the functional capabilities of atomic operations like
 * those found in GPU cache hierarchies (i.e. away from the GPU core). The
 * timing of these operations must be enforced by an appropriately designed
 * Ruby cache coherence protocol (e.g. VI_hammer).
 */
class AtomicOpRequest {

  public:
    // The type of the atomic operation
    enum Operation { ATOMIC_INVALID_OP,
                     ATOMIC_CAS_OP,
                     ATOMIC_ADD_OP,
                     ATOMIC_INC_OP,
                     ATOMIC_MAX_OP,
                     ATOMIC_MIN_OP };

    // The data type on which the atomic operates
    enum DataType { INVALID_TYPE,
                    S32_TYPE,
                    U32_TYPE,
                    F32_TYPE,
                    B32_TYPE };

    // An identifier for the requester (e.g. GPU lane ID)
    unsigned uniqueId;
    DataType dataType;
    Operation atomicOp;
    // The offset in the cache line specified by the parent packet
    unsigned lineOffset;
    // When stored as data in a packet, this variable signals if this
    // instance is the last access in the packet for iteration purposes
    bool lastAccess;

  private:
    uint8_t data[16];

  public:
    AtomicOpRequest() : atomicOp(ATOMIC_INVALID_OP) {}

    int dataSizeBytes() {
        switch (dataType) {
          case S32_TYPE:
          case U32_TYPE:
          case F32_TYPE:
          case B32_TYPE:
            return 4;
          default:
            panic("Unknown atomic type: %s\n", atomicOp);
            break;
        }
        return 0;
    }

    void setData(uint8_t *in_data) {
        assert(atomicOp != ATOMIC_INVALID_OP);
        memcpy(data, in_data, dataSizeBytes());
        if (atomicOp == ATOMIC_CAS_OP) {
            memcpy(&data[8], &in_data[8], dataSizeBytes());
        }
    }

    uint8_t *getData() {
        return data;
    }

    void writeData(uint8_t *out_data) {
        memcpy(out_data, data, dataSizeBytes());
    }

    // Called from the RubyPort hit callback to actually perform the atomic
    // operation requests in a CoalescedAccess (i.e. the passed PacketPtr)
    static void atomicMemoryAccess(PacketPtr pkt, SimpleMemory *phys_mem);

  private:
    // Perform the atomic's operation on the passed data
    // TODO: This will need to be expanded to support atomics with more operands
    void doAtomicOperation(uint8_t *read_data, uint8_t *write_data);

};

#endif // __ATOMIC_OPERATIONS_HH__
