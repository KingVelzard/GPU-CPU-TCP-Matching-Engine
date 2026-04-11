#pragma once 

#include "PlatformTraits.h"
#include <cstdint>
#include <atomic>
#include <cassert>


// MatchCommand: one entry in CPU -> GPU command 
// When CPU matcher executes a fill against CPUHalfBook, it needs to let GPU know 
// So GPU can apply same transaction to GPUBook so doesnt show false liquidity
//
// The ring buffer holding these MUST be allocated with cudaMallocHost, so visible to GPU 

struct alignas(16) MatchCommand {

  int64_t quantity; // ALWAYS negative (consumption)
  uint64_t push_ns; // timestamp CPU matched 
  uint32_t instrument; // which instrument (= GPU block)
  uint32_t price_idx; // addr_at(price) or (price - limit_down) / tick 
  uint8_t side; // 0 = bid, 1 = ask 
  uint8_t _pad[7] // pad to reach 32 bytes

};

static_assert(sizeof(MatchCommand) == 32, "MatchCommand is 32 bytes");

// MatchCommandRing: SPSC ring buffer in cudaMallocHost (pinned memory)
//
// Producer: CPU matcher thread 
// Consumer: GPU kernel (one thread per block reads at kernel start)
//
// GPU zerocopy, visible 

static constexpr uint32_t MATCH_RING_CAPACITY = 1024;
static constexpr uint32_t MATCH_RING_MASK = MATCH_RING_CAPACITY - 1;

struct alignas(64) MatchCommandRing {
  // GPU reads head, CPU writes tail
  // if they share, every CPU tail write messes up the cache line the GPU spins on = many cache misses 

  // both aligned so they dont align on cache line (no false sharing)
  alignas(64) std::atomic<uint32_t> tail{0}; // cpu writes, gpu reads | atomic so won't cache in register + avoids reording
  alignas(64) std::atomic<uint32_t> head{0}; // GPU writes | atomic so won't cache 

  std::array<MatchCommand, MATCH_RING_CAPACITY> slots;

  // CPU push -- CPU matcher pushes to ring buffer 
  // Returns false if ring is full (overflow)

  bool push(const MatchCommand& command) noexcept {
    uint32_t tail = this->tail.load(std::memory_order_relaxed); // relaxed but doesnt matter 
    uint32_t head = this->head.load(std::memory_order_relaxed); 

    if (tail - head >= MATCH_RING_CAPACITY) {
      return false; // overflow1!!!
    }

    // modulo in order to wrap around, 1024 & 1023 goes to 0
    // adds command to array
    slots[t & MATCH_RING_MASK] = cmd;

    // release: slot write needs to be visible before tail increment and command add 
    tail.store(tail + 1, std::memory_order_release);
    return true;
  }

  // GPU pop -- called inside the kernel by thread 0 of each block
  // NOTE THIS FUNCTION IS __device__ ONLY DO NOT CALL FROM CPU
  // Implemented in LOBKernel.cu 

  DEVICE_ATTR int pop();

}
