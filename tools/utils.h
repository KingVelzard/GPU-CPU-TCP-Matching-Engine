#pragma once
#include <assert.h>
#include <cstdint>
#include <format>
#include <string>

// Price representation
static constexpr int64_t PRICE_SCALE = 100; // $1.00 = 100 internal units

// GPU layout
static constexpr uint32_t GPU_N_INSTRUMENTS = 8;  // number of instruments
static constexpr uint32_t GPU_MAX_LEVELS = 10000; // worst-case across all instruments
                                                  // must be >= max n_levels of any instrument

// Snapshot tiers
static constexpr int TOP_N = 50;       // TOP_N best prices to watch (tier 2)
static constexpr int EXTENDED_N = 500; // TOP_N best prices to watch (tier 3)

// Batch sizing
static constexpr uint32_t BATCH_PER_INSTRUMENT = 64;    // batch of orders per instrument on GPU
static constexpr uint64_t FAST_TIMEOUT_NS = 50'000ULL;  // timeout for fast stream
static constexpr uint64_t SLOW_TIMEOUT_NS = 500'000ULL; // timeout for slow stream

// Match command ring
static constexpr uint32_t MATCH_RING_CAPACITY = 512; // capacity for transfer from Matcher -> GPU

// Classifier
static constexpr int64_t NEAR_SPREAD_TICKS = 5; // ticks near spread

// Thread topology
static constexpr int N_REACTORS = 8;     // number of reactor threads
static constexpr int MATCHER_CORE = 17;  // core pinned to matcher
static constexpr int DISPATCH_CORE = 19; // core pinned to GPU dispatch

// Networking
static constexpr int PORT = 3490;    // port server runs on
static constexpr int BACKLOG = 8192; // backlog of orders per

struct InstrumentLimits {
  int64_t limit_down; // lowest valid price in ticks
  int64_t limit_up;   // highest valid price in ticks
  int64_t tick;       // minimum price increment in ticks
  uint32_t n_levels;  // (limit_up - limit_down) / ticks: total number of ticks
};

struct InstrumentRegistry {
  InstrumentLimits limits[GPU_N_INSTRUMENTS]; // Instument data for each instrument
  uint32_t count = 0;                         // number of instruments registered
  bool finalized = false;

  // returns GPU block index that corresponds to instrument
  uint32_t register_intruments(int64_t limit_down, int64_t limit_up, int64_t tick) {
    assert(!finalized && "Cannot register instruments after finalize");
    assert(count < GPU_N_INSTRUMENTS && "Too many instruments");

    uint32_t index = count++;
    // copy list init
    limits[index] = {limit_down, limit_up, tick, static_cast<uint32_t>((limit_up - limit_down) / tick + 1)};
    return index;
  }

  // Call before starting threads
  // After this, register is read-only
  void finalize() {
    assert(count > 0 && "No instruments registered");
    this->finalized = true;
  }

  // singleton get
  static InstrumentRegistry &get() {
    static InstrumentRegistry instance;
    return instance;
  }
};

namespace utils {

// converts a price to a fixed-point integer
inline int64_t price_to_int(double price) noexcept { return static_cast<int64_t>(price * PRICE_SCALE); }

// converts a fixed-point integer back to a string representation
inline std::string int_to_price(int64_t price) noexcept {
  return std::format("{:.2f}", static_cast<double>(price) / PRICE_SCALE);
}

// to translate price to index for fixed-width array
inline uint32_t addr_at(int64_t price, uint32_t instrument) noexcept {
  const auto &lim = InstrumentRegistry::get().limits[instrument];
  return static_cast<std::uint32_t>((price - lim.limit_down) / lim.tick);
}

// to find the price at index in fixed-width array
inline int64_t price_at(int index, uint32_t instrument) noexcept {
  const auto &lim = InstrumentRegistry::get().limits[instrument];
  return (static_cast<int64_t>(index - 1) * lim.tick) + lim.limit_down;
}

#ifdef __CUDACC__
__device__ inline uint32_t addr_at_device(int64_t price, int64_t limit_down, int64_t tick) noexcept {
  return static_cast<uint32_t>((price - limit_down) / tick);
}

__device__ inline int64_t price_at_device(uint32_t idx, int64_t limit_down, int64_t tick) noexcept {
  return limit_down + static_cast<int64_t>(idx) * tick;
}
#endif

} // namespace utils
