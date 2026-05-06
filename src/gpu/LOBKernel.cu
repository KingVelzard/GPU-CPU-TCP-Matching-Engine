#include "../../include/Feed.h"
#include "../../include/InstrumentSnapshot.h"
#include "GPUBook.cuh"      // GPU_INSTRUMENT_STRIDE_ELEMS, GPU_BID_OFFSET, GPU_ASK_OFFSET
#include "LOBKernelTypes.h" // LOBKernelParams, launch_lob_kernel declaration
#include <atomic>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// Shared memory layout for one block (one instrument per block).
struct alignas(64) BlockSharedMem {
  FeedSM bid_cache[TOP_N]; // top-N bid levels after recompute
  FeedSM ask_cache[TOP_N]; // top-N ask levels after recompute
  int bid_cache_sz;
  int ask_cache_sz;

  int32_t reduce_scratch[32];  // warp-lane 0 reduction intermediates
  uint32_t match_cmds_applied; // diagnostic; used by Phase 1
};

__device__ inline int64_t *bid_base(int64_t *book, uint32_t instrument) {
  return book + instrument * GPU_INSTRUMENT_STRIDE_ELEMS + GPU_BID_OFFSET;
}

__device__ inline int64_t *ask_base(int64_t *book, uint32_t instrument) {
  return book + instrument * GPU_INSTRUMENT_STRIDE_ELEMS + GPU_ASK_OFFSET;
}

// Per-instrument price at a flat array index.
__device__ inline int64_t price_at(const LOBKernelParams &params, uint32_t instrument, int32_t index) {
  return params.limit_down[instrument] + static_cast<int64_t>(index) * params.tick_sz[instrument];
}

// ── Phase 1: apply pending MatchCommands from CPU Matcher ─────────────────────
// Thread 0 only — single GPU consumer of the SPSC ring.

__device__ void apply_match_commands(LOBKernelParams &params, uint32_t instrument, BlockSharedMem &smem) {
  if (threadIdx.x != 0)
    return;

  // acquire load on tail: see all CPU writes that preceded the tail store.
  const uint32_t tail = params.match_ring->tail.load(std::memory_order_acquire);
  uint32_t head = params.match_ring->head.load(std::memory_order_relaxed);
  uint32_t applied = 0;

  while (head != tail) {
    const MatchCommand &cmd = params.match_ring->slots[head & MATCH_RING_MASK];

    if (cmd.instrument == instrument) {
      int64_t *book_side =
          (cmd.side == 0) ? bid_base(params.device_book, instrument) : ask_base(params.device_book, instrument);
      atomicAdd(reinterpret_cast<unsigned long long *>(&book_side[cmd.price_idx]),
                static_cast<unsigned long long>(cmd.quantity));
    }

    ++head;
    ++applied;
  }

  // release: CPU producer sees updated head before writing new commands.
  params.match_ring->head.store(head, std::memory_order_release);
  smem.match_cmds_applied = applied;
}

// ── Phase 2: load top-N cache from pinned snapshot into shared memory ─────────

__device__ void load_cache(LOBKernelParams &params, uint32_t instrument, BlockSharedMem &smem) {
  const InstrumentSnapshot &snap = params.snapshots[instrument];
  const int tid = static_cast<int>(threadIdx.x);

  if (tid < TOP_N)
    smem.bid_cache[tid] = snap.best_bids[tid];
  else if (tid < 2 * TOP_N)
    smem.ask_cache[tid - TOP_N] = snap.best_asks[tid - TOP_N];

  if (tid == 0) {
    smem.bid_cache_sz = TOP_N;
    smem.ask_cache_sz = TOP_N;
  }
  __syncthreads();
}

// ── Phase 3: apply add/cancel batch orders to the GPU book ────────────────────

__device__ void process_batch(LOBKernelParams &params, uint32_t instrument, BlockSharedMem &smem) {
  const uint32_t order_count = params.batch_count[instrument];
  const int tid = static_cast<int>(threadIdx.x);

  if (static_cast<uint32_t>(tid) < order_count) {
    // Sub-batch for instrument i occupies global slots
    // [i * BATCH_PER_INSTRUMENT, i * BATCH_PER_INSTRUMENT + count[i]).
    const uint32_t global_idx = instrument * BATCH_PER_INSTRUMENT + static_cast<uint32_t>(tid);
    const uint32_t price_idx = params.batch_price_idx[global_idx];
    const int64_t quantity = params.batch_quantity[global_idx];
    const uint8_t side = params.batch_side[global_idx];

    int64_t *book_side =
        (side == 0) ? bid_base(params.device_book, instrument) : ask_base(params.device_book, instrument);

    atomicAdd(reinterpret_cast<unsigned long long *>(&book_side[price_idx]), static_cast<unsigned long long>(quantity));
  }
  (void)smem;
}

// ── Phase 4: recompute top-N cache from updated book ─────────────────────────

__device__ void recompute_cache(LOBKernelParams &params, uint32_t instrument, BlockSharedMem &smem) {
  int64_t *bids = bid_base(params.device_book, instrument);
  int64_t *asks = ask_base(params.device_book, instrument);

  // Each thread scans its slice of GPU_MAX_LEVELS for local best bid/ask.
  const uint32_t levels_per_thread = (GPU_MAX_LEVELS + blockDim.x - 1) / blockDim.x;
  const uint32_t start = threadIdx.x * levels_per_thread;
  const uint32_t end = min(start + levels_per_thread, GPU_MAX_LEVELS);

  // Sentinel -1 for bid (any valid index > -1), INT32_MAX for ask (any valid index < INT32_MAX).
  int32_t local_best_bid = -1;
  int32_t local_best_ask = INT32_MAX;

  for (uint32_t i = start; i < end; ++i)
    if (bids[i] > 0)
      local_best_bid = static_cast<int32_t>(i);

  for (uint32_t i = start; i < end; ++i) {
    if (asks[i] > 0) {
      local_best_ask = static_cast<int32_t>(i);
      break;
    }
  }

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  // greater: picks the highest bid index across the warp (correct with -1 sentinel).
  // less:    picks the lowest  ask index across the warp (correct with INT32_MAX sentinel).
  const int32_t warp_best_bid = cg::reduce(warp, local_best_bid, cg::greater<int32_t>());
  const int32_t warp_best_ask = cg::reduce(warp, local_best_ask, cg::less<int32_t>());

  if (warp.thread_rank() == 0) {
    const int w = warp.meta_group_rank();
    smem.reduce_scratch[w * 2] = warp_best_bid;
    smem.reduce_scratch[w * 2 + 1] = warp_best_ask;
  }
  block.sync();

  if (threadIdx.x == 0) {
    const uint32_t n_warps = blockDim.x / 32;
    int32_t best_bid = -1;
    int32_t best_ask = INT32_MAX;

    for (uint32_t w = 0; w < n_warps; ++w) {
      const int32_t wb = smem.reduce_scratch[w * 2];
      const int32_t wa = smem.reduce_scratch[w * 2 + 1];
      if (wb > best_bid)
        best_bid = wb;
      if (wa < best_ask)
        best_ask = wa;
    }
    if (best_ask == INT32_MAX)
      best_ask = -1; // no asks found — make sentinel consistent

    // Walk TOP_N bids (highest price first, descending from best_bid).
    int slot_b = 0;
    for (int i = best_bid; i >= 0 && slot_b < TOP_N; --i)
      if (bids[i] > 0)
        smem.bid_cache[slot_b++] = {price_at(params, instrument, i), bids[i]};
    smem.bid_cache_sz = slot_b;

    // Walk TOP_N asks (lowest price first, ascending from best_ask).
    int slot_a = 0;
    if (best_ask >= 0) {
      for (int i = best_ask; i < static_cast<int>(GPU_MAX_LEVELS) && slot_a < TOP_N; ++i)
        if (asks[i] > 0)
          smem.ask_cache[slot_a++] = {price_at(params, instrument, i), asks[i]};
    }
    smem.ask_cache_sz = slot_a;
  }

  block.sync();
}

// ── Phase 5: seqlock writeback to pinned snapshot ─────────────────────────────
// All threads participate: thread 0 owns the seqlock; others write cache slots.

__device__ void seqlock_writeback(LOBKernelParams &params, uint32_t instrument, BlockSharedMem &smem) {
  InstrumentSnapshot &snap = params.snapshots[instrument];
  const int tid = static_cast<int>(threadIdx.x);

  if (tid == 0) {
    atomicAdd(reinterpret_cast<unsigned int *>(&snap.version), 1u); // version → odd
    __threadfence_system();
  }
  __syncthreads();

  // All TOP_N slots are always written — slots beyond cache_sz get zeroed so
  // stale entries from a previous batch never linger.
  const FeedSM zero{0LL, 0LL};
  if (tid < TOP_N)
    snap.best_bids[tid] = (tid < smem.bid_cache_sz) ? smem.bid_cache[tid] : zero;
  else if (tid < 2 * TOP_N) {
    const int a = tid - TOP_N;
    snap.best_asks[a] = (a < smem.ask_cache_sz) ? smem.ask_cache[a] : zero;
  }
  __syncthreads();

  if (tid == 0) {
    if (smem.bid_cache_sz > 0)
      atomicExch(reinterpret_cast<long long *>(&snap.best_bid_price), static_cast<long long>(smem.bid_cache[0].price));
    if (smem.ask_cache_sz > 0)
      atomicExch(reinterpret_cast<long long *>(&snap.best_ask_price), static_cast<long long>(smem.ask_cache[0].price));
    snap.batch_seq.store(params.batch_seq, std::memory_order_relaxed);
    __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned int *>(&snap.version), 1u); // version → even
  }
}

// ── Phase 6: delta output (stubbed) ───────────────────────────────────────────

__device__ void write_deltas(LOBKernelParams &, uint32_t, BlockSharedMem &) {
  // GPU→CPU sync now handled entirely through the Phase 5 seqlock snapshot.
}

// ── Main kernel ───────────────────────────────────────────────────────────────
// Launch: gridDim.x = GPU_N_INSTRUMENTS, blockDim.x = MAX_BATCH_SIZE,
//         smem = sizeof(BlockSharedMem)

__global__ void lob_kernel(LOBKernelParams params) {
  const uint32_t instrument = blockIdx.x;
  __shared__ BlockSharedMem smem;

  apply_match_commands(params, instrument, smem);
  __syncthreads();

  load_cache(params, instrument, smem);
  __syncthreads();

  process_batch(params, instrument, smem);
  __syncthreads();

  recompute_cache(params, instrument, smem);
  __syncthreads();

  seqlock_writeback(params, instrument, smem);
  __syncthreads();

  write_deltas(params, instrument, smem);
}

// ── Launch wrapper ────────────────────────────────────────────────────────────

void launch_lob_kernel(const LOBKernelParams &params, cudaStream_t stream) {
  const dim3 grid(GPU_N_INSTRUMENTS);
  const dim3 block(MAX_BATCH_SIZE);
  lob_kernel<<<grid, block, sizeof(BlockSharedMem), stream>>>(params);
}
