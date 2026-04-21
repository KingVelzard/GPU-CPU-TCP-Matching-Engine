#pragma once 
#include <cstdint>
#include <atomic>

// BookDelta: one entry in the GPU -> CPU log 
//
// After each GPU batch, the kernel writes the post-update quantity for every 
// book_[] slot it modified. The CPU applies these to its own book, keeping
// both copies in sync.
//
// Delta output buffer MUST be allocated with cudaMallocHost so the CUDA callback
// thread can read it from CPU without a cudaMemcpy

struct alignas(16) BookDelta {
  int64_t new_quantity; // ABSOLUTE post-update value (not relative change)
  uint32_t instrument;  // which instrument
  uint32_t price_idx; // index in book_[], same as addr_at(price)
  uint8_t side; // 0 = BID, 1 = ASK 
  uint8_t _pad[7];
};

static_assert(sizeof(BookDelta) == 24, "BookDelta must be 24 bytes");
static_assert(alignof(BookDelta) == 16, "BookDelta alignment must be 16 bytes");

// DeltaOutputBuffer: written by GPU kernel, read by CPU in on_batch_compelte()
//
// Worst case size is when one delta per order in a batch 
// TODO make constinit variables so we dont have to MANNUALLY do this 
//  but for now, MAX_BATCH_SIZE = 512 and 8 instruments: 4096 deltas maximum 
//  only 100KB per batch, nothing

static constexpr uint32_t MAX_DELTAS_PER_BATCH = 4096;

struct alignas(64) DeltaOutputBuffer {
  std::atomic<uint32_t> count{0}; // number of valid deltas written to 
  BookDelta deltas[MAX_DELTAS_PER_BATCH];

  void reset() noexcept {
    count.store(0, std::memory_order_relaxed); 
  }
};

static_assert(alignof(DeltaOutputBuffer) == 64, "DeltaOutputBuffer alignment must be 64 bytes");
