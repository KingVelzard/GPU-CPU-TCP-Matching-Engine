#pragma once
#include "../tools/utils.h"
#include <atomic>
#include <cstdint>

// MAX_BATCH_SIZE: total slots across all per-instrument sub-batches.
// GPU_N_INSTRUMENTS and BATCH_PER_INSTRUMENT come from utils.h.
static constexpr uint32_t MAX_BATCH_SIZE = BATCH_PER_INSTRUMENT * GPU_N_INSTRUMENTS;

// price_to_idx: flat book array index from a price in ticks.
// Thin wrapper around utils::addr_at so call sites stay readable.
inline uint32_t price_to_idx(int64_t price_ticks, uint32_t instrument) noexcept {
  return utils::addr_at(price_ticks, instrument);
}

// GPUBatch: SoA pinned arrays that reactors write into directly.
// GPU reads the same physical pages — zero copies.
//
// Per-instrument sub-batches: instrument i's orders occupy global slots
//   [i * BATCH_PER_INSTRUMENT, i * BATCH_PER_INSTRUMENT + count[i])
// The kernel addresses instrument blockIdx.x's region at a fixed offset —
// no prefix sum or sorting needed.
struct alignas(64) GPUBatch {

  // ── SoA order fields (cudaMallocHost, read by GPU) ─────────────────────
  GPUBatch(const GPUBatch &) = delete;
  GPUBatch(GPUBatch &&) = delete;
  GPUBatch &operator=(const GPUBatch &) = delete;
  GPUBatch &operator=(GPUBatch &&) = delete;
  uint32_t *price_idx; // (price - limit_down) / tick
  int64_t *quantity;   // positive = add, negative = cancel
  uint8_t *side;       // 0 = BID, 1 = ASK

  // ── Routing fields (malloc, not read by GPU) ────────────────────────────
  uint64_t *conn_token; // fd << 32 | generation for fill routing
  uint32_t *reactor_id_arr;

  // ── Per-instrument slot counters ────────────────────────────────────────
  // Reactor fetch_add on count[instrument] to claim a local slot.
  // One counter per instrument minimises cross-reactor contention:
  // two reactors writing different instruments never touch the same counter.
  alignas(64) std::atomic<uint32_t> count[GPU_N_INSTRUMENTS];

  // Arrival time of first order written; used by dispatch for timeout fire.
  std::atomic<uint64_t> first_order_ns{0};

  // Global slot index for instrument i, local slot j within its sub-batch.
  static constexpr uint32_t slot(uint32_t instrument, uint32_t local) noexcept {
    return instrument * BATCH_PER_INSTRUMENT + local;
  }

  // try_write: atomically claim a slot and write order fields.
  // Thread-safe for concurrent reactors writing to different instruments.
  // Returns false (backpressure) if the instrument's sub-batch is full.
  bool try_write(uint32_t instrument, uint32_t price_idx_val, int64_t quantity_val, uint8_t side_val,
                 uint64_t conn_token_val, uint32_t reactor_id_val, uint64_t arrive_ns) noexcept {

    uint32_t local = count[instrument].fetch_add(1, std::memory_order_relaxed);
    if (local >= BATCH_PER_INSTRUMENT) {
      count[instrument].fetch_sub(1, std::memory_order_relaxed);
      return false;
    }

    const uint32_t s = slot(instrument, local);
    price_idx[s] = price_idx_val;
    quantity[s] = quantity_val;
    side[s] = side_val;
    conn_token[s] = conn_token_val;
    reactor_id_arr[s] = reactor_id_val;

    uint64_t expected = 0;
    first_order_ns.compare_exchange_strong(expected, arrive_ns, std::memory_order_relaxed);
    return true;
  }

  // reset: zero all per-instrument counts and the first_order timestamp.
  // Called by dispatch before exposing the new buffer to reactors.
  void reset() noexcept {
    for (auto &c : this->count) {
      c.store(0, std::memory_order_relaxed);
    }
  }

  // Implemented in GPUBook.cuh (compiled by nvcc — needs cudaMallocHost).
  void init();
  void destroy();
};
