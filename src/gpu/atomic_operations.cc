
#include "base/trace.hh"
#include "debug/AtomicOperations.hh"
#include "gpu/atomic_operations.hh"
#include "mem/simple_mem.hh"

void
AtomicOpRequest::atomicMemoryAccess(PacketPtr pkt, SimpleMemory *phys_mem)
{
    // The pkt's atomic operation commands
    AtomicOpRequest **atomic_ops =
                                (AtomicOpRequest**)pkt->getPtr<uint8_t*>();

    // Create a packet each for the reads and the writes to the physical
    // memory. These accesses occur with phys_mem.access(), which
    // turns the packet into a response
    int data_size_bytes = atomic_ops[0]->dataSizeBytes();
    assert(data_size_bytes == 4);
    Request atomic_req(pkt->getAddr(), data_size_bytes,
                       pkt->req->getFlags(), 0);

    // Do physical memory accesses for each of the pkt's atomic operations
    bool atomics_done = false;
    for (int i = 0; !atomics_done; i++) {

        uint8_t read_data[8];
        Packet atomic_read_pkt(&atomic_req, MemCmd::ReadReq, data_size_bytes);

        uint8_t write_data[8];
        Packet atomic_write_pkt(&atomic_req, MemCmd::WriteReq, data_size_bytes);

        assert(atomic_ops[i]->dataSizeBytes() == data_size_bytes);

        // Set up the read packet to get the correct data
        atomic_read_pkt.setAddr(pkt->getAddr() + atomic_ops[i]->lineOffset);
        atomic_read_pkt.dataStatic(read_data);

        // Read the current physical memory data
        phys_mem->access(&atomic_read_pkt);

        // Actually do the atomic operation with the data
        DPRINTF(AtomicOperations, "Performing operation for addr: %x\n",
                atomic_read_pkt.getAddr());
        atomic_ops[i]->doAtomicOperation(read_data, write_data);

        // Set up the write packet to put new data
        atomic_write_pkt.setAddr(pkt->getAddr() + atomic_ops[i]->lineOffset);
        atomic_write_pkt.dataStatic(write_data);

        // Write the atomic operation result back to the physical memory
        phys_mem->access(&atomic_write_pkt);

        atomics_done = atomic_ops[i]->lastAccess;
    }

    assert(pkt->needsResponse());
    pkt->makeResponse();
}

void
AtomicOpRequest::doAtomicOperation(uint8_t *read_data, uint8_t *write_data)
{
    switch (atomicOp) {
      //-----------------------------------------------------------------------
      // Perform compare and swap
      //-----------------------------------------------------------------------
      case ATOMIC_CAS_OP: {

        switch (dataType) {
        case B32_TYPE:
        case U32_TYPE: {
            unsigned int mem_data = *((unsigned int*)read_data);
            unsigned int reg_b_data = *((unsigned int*)&data[0]);
            unsigned int reg_c_data = *((unsigned int*)&data[8]);
            unsigned int new_mem_data =
                        (mem_data == reg_b_data) ? reg_c_data : mem_data;
            *((unsigned int*)&data[0]) = mem_data;
            *((unsigned int*)write_data) = new_mem_data;
            DPRINTF(AtomicOperations,
                    "Atomic compare and swap: (%u == %u) ? %u : %u = %u\n",
                    mem_data, reg_b_data, reg_c_data, mem_data, new_mem_data);
            break;
          }

          case INVALID_TYPE:
          default:
            panic("Unimplemented atomic compare and swap type: %s", dataType);
            break;
        }

        break;
      }

      //-----------------------------------------------------------------------
      // Perform addition
      //-----------------------------------------------------------------------
      case ATOMIC_ADD_OP: {

        switch (dataType) {
          case S32_TYPE: {
            int mem_data = *((int*)read_data);
            int reg_data = *((int*)&data[0]);
            int new_mem_data = reg_data + mem_data;
            *((int*)&data[0]) = mem_data;
            *((int*)write_data) = new_mem_data;
            DPRINTF(AtomicOperations,
                    "Atomic add: %u + %u = %u\n",
                    reg_data, mem_data, new_mem_data);
            break;
          }

          case U32_TYPE: {
            unsigned int mem_data = *((unsigned int*)read_data);
            unsigned int reg_data = *((unsigned int*)&data[0]);
            unsigned int new_mem_data = reg_data + mem_data;
            *((unsigned int*)&data[0]) = mem_data;
            *((unsigned int*)write_data) = new_mem_data;
            DPRINTF(AtomicOperations,
                    "Atomic add: %u + %u = %u\n",
                    reg_data, mem_data, new_mem_data);
            break;
          }

          case F32_TYPE: {
            float mem_data = *((float*)read_data);
            float reg_data = *((float*)&data[0]);
            float new_mem_data = reg_data + mem_data;
            *((float*)&data[0]) = mem_data;
            *((float*)write_data) = new_mem_data;
            DPRINTF(AtomicOperations,
                    "Atomic add: %f + %f = %f\n",
                    reg_data, mem_data, new_mem_data);
            break;
          }

          case INVALID_TYPE:
          default:
            panic("Unimplemented atomic add type: %s", dataType);
            break;
        }

        break;
      }

      //-----------------------------------------------------------------------
      // Perform increment
      //-----------------------------------------------------------------------
      case ATOMIC_INC_OP: {

        switch (dataType) {
          case U32_TYPE: {
            unsigned int mem_data = *((unsigned int*)read_data);
            unsigned int reg_data = *((unsigned int*)&data[0]);
            unsigned int new_mem_data =
                    (mem_data >= reg_data) ? 0 : mem_data + 1;
            *((unsigned int*)&data[0]) = mem_data;
            *((unsigned int*)write_data) = new_mem_data;
            DPRINTF(AtomicOperations,
                    "Atomic inc with overflow = %u: %u + 1 = %u\n",
                    reg_data, mem_data, new_mem_data);
            break;
          }

          case INVALID_TYPE:
          default:
            panic("Unimplemented atomic increment type: %s", dataType);
            break;
        }

        break;
      }

      //-----------------------------------------------------------------------
      // Perform maximum
      //-----------------------------------------------------------------------
      case ATOMIC_MAX_OP: {

        switch (dataType) {
          case U32_TYPE: {
            unsigned int mem_data = *((unsigned int*)read_data);
            unsigned int reg_data = *((unsigned int*)&data[0]);
            unsigned int new_mem_data =
                    (mem_data > reg_data) ? mem_data : reg_data;
            *((unsigned int*)&data[0]) = mem_data;
            *((unsigned int*)write_data) = new_mem_data;
            DPRINTF(AtomicOperations,
                    "Atomic max(operand: %u, memory: %u) = %u\n",
                    reg_data, mem_data, new_mem_data);
            break;
          }

          case S32_TYPE: {
            int mem_data = *((int*)read_data);
            int reg_data = *((int*)&data[0]);
            int new_mem_data =
                    (mem_data > reg_data) ? mem_data : reg_data;
            *((int*)&data[0]) = mem_data;
            *((int*)write_data) = new_mem_data;
            DPRINTF(AtomicOperations,
                    "Atomic max(operand: %u, memory: %u) = %u\n",
                    reg_data, mem_data, new_mem_data);
            break;
          }

          case INVALID_TYPE:
          default:
            panic("Unimplemented atomic max type: %s", dataType);
            break;
        }

        break;
      }

      //-----------------------------------------------------------------------
      // Perform minimum
      //-----------------------------------------------------------------------
      case ATOMIC_MIN_OP: {

        switch (dataType) {
          case S32_TYPE: {
            int mem_data = *((int*)read_data);
            int reg_data = *((int*)&data[0]);
            int new_mem_data =
                    (mem_data < reg_data) ? mem_data : reg_data;
            *((int*)&data[0]) = mem_data;
            *((int*)write_data) = new_mem_data;
            DPRINTF(AtomicOperations,
                    "Atomic min(operand: %u, memory: %u) = %u\n",
                    reg_data, mem_data, new_mem_data);
            break;
          }

          case U32_TYPE: {
            unsigned int mem_data = *((unsigned int*)read_data);
            unsigned int reg_data = *((unsigned int*)&data[0]);
            unsigned int new_mem_data =
                    (mem_data < reg_data) ? mem_data : reg_data;
            *((unsigned int*)&data[0]) = mem_data;
            *((unsigned int*)write_data) = new_mem_data;
            DPRINTF(AtomicOperations,
                    "Atomic min(operand: %u, memory: %u) = %u\n",
                    reg_data, mem_data, new_mem_data);
            break;
          }

          case INVALID_TYPE:
          default:
            panic("Unimplemented atomic min type: %s", dataType);
            break;
        }

        break;
      }

      default:
        panic("Unimplemented atomic operation: %s", atomicOp);
        break;

    }
}

// The rest of the code from GPGPU-Sim's atomics implementations
// TODO: These need to be implemented above with tests to verify correctness
//        switch ( m_atomic_spec ) {
//        // AND
//        case ATOMIC_AND:
//           {
//
//              switch ( to_type ) {
//              case B32_TYPE:
//              case U32_TYPE:
//                 op_result.u32 = data.u32 & src2_data.u32;
//                 break;
//              case S32_TYPE:
//                 op_result.s32 = data.s32 & src2_data.s32;
//                 break;
//              default:
//                 printf("Execution error: type mismatch (%x) with instruction\natom.AND only accepts b32\n", to_type);
//                 assert(0);
//                 break;
//              }
//
//              break;
//           }
//           // OR
//        case ATOMIC_OR:
//           {
//
//              switch ( to_type ) {
//              case B32_TYPE:
//              case U32_TYPE:
//                 op_result.u32 = data.u32 | src2_data.u32;
//                 break;
//              case S32_TYPE:
//                 op_result.s32 = data.s32 | src2_data.s32;
//                 break;
//              default:
//                 printf("Execution error: type mismatch (%x) with instruction\natom.OR only accepts b32\n", to_type);
//                 assert(0);
//                 break;
//              }
//
//              break;
//           }
//           // XOR
//        case ATOMIC_XOR:
//           {
//
//              switch ( to_type ) {
//              case B32_TYPE:
//              case U32_TYPE:
//                 op_result.u32 = data.u32 ^ src2_data.u32;
//                 break;
//              case S32_TYPE:
//                 op_result.s32 = data.s32 ^ src2_data.s32;
//                 break;
//              default:
//                 printf("Execution error: type mismatch (%x) with instruction\natom.XOR only accepts b32\n", to_type);
//                 assert(0);
//                 break;
//              }
//
//              break;
//           }
//           // EXCH
//        case ATOMIC_EXCH:
//           {
//              switch ( to_type ) {
//              case B32_TYPE:
//              case U32_TYPE:
//                 op_result.u32 = MY_EXCH(data.u32, src2_data.u32);
//                 break;
//              case S32_TYPE:
//                 op_result.s32 = MY_EXCH(data.s32, src2_data.s32);
//                 break;
//              default:
//                 printf("Execution error: type mismatch (%x) with instruction\natom.EXCH only accepts b32\n", to_type);
//                 assert(0);
//                 break;
//              }
//
//              break;
//           }
//           // INC
//        case ATOMIC_DEC:
//           {
//              switch ( to_type ) {
//              case U32_TYPE:
//                 op_result.u32 = MY_DEC_I(data.u32, src2_data.u32);
//                 break;
//              default:
//                 printf("Execution error: type mismatch with instruction\natom.DEC only accepts u32 and s32\n");
//                 assert(0);
//                 break;
//              }
//
//              break;
//           }
//           // DEFAULT
//        default:
//           {
//              assert(0);
//              break;
//           }
//        }
