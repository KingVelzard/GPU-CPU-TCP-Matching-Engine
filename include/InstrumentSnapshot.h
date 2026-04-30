#pragma once
#include <cstdint>
#include <atomic>
#include "Feed.h"

struct alignas(64) InstrumentSnapshot {
    // --- Cache line 0: seqlock version (on its own line) ---
    alignas(64) std::atomic<uint32_t> version{0};  // odd=writing, even=done
    uint32_t _pad0[15];

    // --- Cache lines 1+: full top-N data (read by market data publisher) ---
    FeedSM best_bids[TOP_N];  // TOP_N × 32 bytes
    FeedSM best_asks[TOP_N];

    // --- Separate cache line: hot classifier fields ---
    alignas(64) std::atomic<int64_t> best_bid_price{0};
    std::atomic<int64_t>             best_ask_price{0};
    std::atomic<uint32_t>            batch_seq{0};    // monotonically increasing
    uint32_t _pad1[1];

    // Validate alignment at compile time.
    // If sizeof is not a multiple of 64, two snapshots in an array
    // would share cache lines — instruments would falsely share.
    // static_assert is below the struct definition.
};

static_assert(sizeof(InstrumentSnapshot) % 64 == 0,
    "InstrumentSnapshot must be a whole number of cache lines to prevent "
    "false sharing between adjacent instruments in the pinned array");

// SpreadSnapshot: what the CPU classifier reads from InstrumentSnapshot.
// Guaranteed consistent by the seqlock protocol.
struct SpreadSnapshot {
    int64_t  bid_price;
    int64_t  ask_price;
    uint32_t batch_seq;
    bool     stale; // set if the seqlock read retried too many times
};

// read_spread: CPU-side seqlock reader.
// Returns a guaranteed-consistent SpreadSnapshot.
// Inline so the compiler can see all the memory ordering and optimize
// the hot path — this is called on every single incoming order.
inline SpreadSnapshot read_spread(const InstrumentSnapshot* snap) noexcept {
    SpreadSnapshot result{};
    uint32_t v1, v2;
    int retries = 0;

    do {
        // acquire: establishes happens-before with the GPU's
        // __threadfence_system() + version store
        v1 = snap->version.load(std::memory_order_acquire);

        if (v1 & 1) {
            // GPU is mid-write. Spin and retry.
            __asm__ volatile("pause" ::: "memory"); // _mm_pause() equivalent
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

        // relaxed loads are safe here: the acquire on v1 already establishes
        // the memory ordering — we see everything written before the even
        // version store that produced v1
        result.bid_price = snap->best_bid_price.load(std::memory_order_relaxed);
        result.ask_price = snap->best_ask_price.load(std::memory_order_relaxed);
        result.batch_seq = snap->batch_seq.load(std::memory_order_relaxed);

        // acquire fence: prevents the compiler/CPU from hoisting the v2 load
        // before the bid/ask loads, which would defeat the consistency check
        std::atomic_thread_fence(std::memory_order_acquire);

        v2 = snap->version.load(std::memory_order_relaxed);
        retries++;
    } while (v1 != v2);

    result.stale = false;
    return result;
}
