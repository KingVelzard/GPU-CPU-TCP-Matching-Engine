#pragma once
#include "Feed.h"
#include <atomic>
#include <cstdint>

struct alignas(64) InstrumentSnapshot {
  // --- Cache line 0: seqlock version (on its own line) ---
  alignas(64) std::atomic<uint64_t> version{0}; // odd=writing, even=done
  uint32_t _pad0[15];

  // --- Cache lines 1+: full top-N data (read by market data publisher) ---
  FeedSM best_bids[TOP_N]; // TOP_N × 32 bytes
  FeedSM best_asks[TOP_N];

  // --- Separate cache line: hot classifier fields ---
  // best_bid and best_ask share cache line for simultenous fetch
  alignas(64) std::atomic<int64_t> best_bid_price{0};
  std::atomic<int64_t> best_ask_price{0};
  std::atomic<uint32_t> batch_seq{0}; // monotonically increasing
  uint32_t _pad1[1];
};

// must whole number of cache lines to avoid false sharing between instruments
static_assert(sizeof(InstrumentSnapshot) % 64 == 0,
              "InstrumentSnapshot must be a whole number of cache lines to prevent "
              "false sharing between adjacent instruments in the pinned array");

// MatcherSnapshot: what the CPU Matcher reads — full top-N levels for book walking.
// Populated by read_hot_levels() via seqlock.
struct MatcherSnapshot {
  int64_t bid_price; // best bid (same as SpreadSnapshot)
  int64_t ask_price; // best ask
  uint32_t batch_seq;
  bool stale;
  int bid_sz;             // valid entries in hot_bids[]
  int ask_sz;             // valid entries in hot_asks[]
  FeedSM hot_bids[TOP_N]; // best bids, highest price first
  FeedSM hot_asks[TOP_N]; // best asks, lowest price first
};

// SpreadSnapshot: what the CPU classifier reads from InstrumentSnapshot.
// Guaranteed consistent by the seqlock protocol.
struct SpreadSnapshot {
  int64_t bid_price;
  int64_t ask_price;
  uint32_t batch_seq;
  bool stale; // set if the seqlock read retried too many times
};

//-------------------------------------------------------------------------------SEQ-LOCKS
//
// SeqLocks here work alongside the GPU, thats why the GPU uses zerocpy memory, (threadfence_system writes are apparent)
// We have a different seq lock for both tiers of reads, SpreadSnap, and MatchSnap, which read different

// read_hot_levels: seqlock reader for the CPU Matcher.
// Returns the full top-N bid/ask snapshot for book-walking during matching.
inline MatcherSnapshot read_hot_levels(const InstrumentSnapshot *snap) noexcept {
  MatcherSnapshot result{};
  uint64_t v1, v2;
  int retries = 0;

  do {
    v1 = snap->version.load(std::memory_order_acquire);
    if (v1 & 1) {
      __asm__ volatile("pause" ::: "memory"); // same thing as mm_pause
      if (++retries > 10000) {                // 10k retries
        result.stale = true;
        return result;
      }
      continue;
    }

    result.bid_price = snap->best_bid_price.load(std::memory_order_relaxed);
    result.ask_price = snap->best_ask_price.load(std::memory_order_relaxed);
    result.batch_seq = snap->batch_seq.load(std::memory_order_relaxed);
    result.bid_sz = TOP_N;
    result.ask_sz = TOP_N;
    for (int i = 0; i < TOP_N; ++i) {
      result.hot_bids[i] = snap->best_bids[i];
      result.hot_asks[i] = snap->best_asks[i];
    }

    std::atomic_thread_fence(
        std::memory_order_acquire); // thread fence for v2 version load since batching with many reads above
    v2 = snap->version.load(std::memory_order_relaxed);
  } while (v1 != v2);

  result.stale = false;
  return result;
}

// read_spread: CPU-side seqlock reader for classifier.
// Returns a guaranteed-consistent SpreadSnapshot.
// Inline so the compiler can see all the memory ordering and optimize
// the hot path — this is called on every single incoming order.
inline SpreadSnapshot read_spread(const InstrumentSnapshot *snap) noexcept {
  SpreadSnapshot result{};
  uint32_t v1, v2;
  int retries = 0;

  do {
    // acquire: __threadfence_system() + version store
    v1 = snap->version.load(std::memory_order_acquire);

    if (v1 & 1) {
      // GPU is mid-write. spin and retry.
      __asm__ volatile("pause" ::: "memory"); // pause

      // gpu mid-write retry
      if (++retries > 10000) {
        result.stale = true;

        // return last-known values rather than spinning forever
        result.bid_price = snap->best_bid_price.load(std::memory_order_relaxed);
        result.ask_price = snap->best_ask_price.load(std::memory_order_relaxed);
        result.batch_seq = snap->batch_seq.load(std::memory_order_relaxed);
        return result;
      }
      continue;
    }

    result.bid_price = snap->best_bid_price.load(std::memory_order_relaxed);
    result.ask_price = snap->best_ask_price.load(std::memory_order_relaxed);
    result.batch_seq = snap->batch_seq.load(std::memory_order_relaxed);

    // acquire fence: prevents the compiler/CPU from hoisting the v2 load
    // before the bid/ask loads, which would defeat the consistency check
    std::atomic_thread_fence(std::memory_order_acquire);

    v2 = snap->version.load(std::memory_order_relaxed);
    // version mismatch retry
    retries++;

    // ensure version hasn't changed since
  } while (v1 != v2);

  result.stale = false;
  return result;
}
