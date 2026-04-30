#include "GPUBook.cuh"
#include "../../include/InstrumentSnapshot.h"
#include "../../include/MatchCommand.h"
#include "../../include/BookDelta.h"
#include "../../include/Feed.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// =============================================================================
// Kernel parameters struct — passed as a single pointer to the kernel.
// Grouping parameters avoids the 4KB CUDA kernel parameter limit and makes
// the launch configuration explicit.
// =============================================================================

struct LOBKernelParams {
    // GPU book arrays
    int64_t*             device_book;           // GPUBook::device_ptr

    // Batch input (SoA, cudaMallocHost)
    const uint32_t*      batch_instrument_id;
    const uint32_t*      batch_price_idx;
    const int64_t*       batch_quantity;
    const uint8_t*       batch_side;
    const uint32_t*      batch_instrument_count; // [N_INSTRUMENTS] orders per instrument

    // Match commands (CPU→GPU, cudaMallocHost SPSC ring)
    MatchCommandRing*    match_ring;

    // Delta output (GPU→CPU, cudaMallocHost)
    DeltaOutputBuffer*   delta_out;

    // Pinned output buffer (seqlock-protected snapshots)
    InstrumentSnapshot*  snapshots;

    // Batch sequence number for batch_seq field in snapshot
    uint32_t             batch_seq;
};

// =============================================================================
// Shared memory layout for one block (one instrument).
// Must fit within sharedMemPerBlock (typically 48KB-96KB).
// =============================================================================

struct alignas(64) BlockSharedMem {
    FeedSM  bid_cache[TOP_N]; // current top-N bid levels (loaded from snapshot)
    FeedSM  ask_cache[TOP_N]; // current top-N ask levels
    int     bid_cache_sz;     // number of valid bid cache entries
    int     ask_cache_sz;     // number of valid ask cache entries

    // Working space for cache recompute (Option B: parallel reduction)
    // Sized for one warp's worth of candidates
    int64_t reduce_scratch[64]; // scratch for finding best bid/ask in block

    uint32_t delta_write_idx; // atomic counter for writing deltas (used by thread 0)

    // Match commands applied this batch (for debugging/stats)
    uint32_t match_cmds_applied;
};

// =============================================================================
// Device helper: get pointer to bid/ask array for this block's instrument
// =============================================================================

__device__ inline int64_t* bid_base(int64_t* book, uint32_t instrument) {
    return book + instrument * GPU_INSTRUMENT_STRIDE_ELEMS + GPU_BID_OFFSET;
}

__device__ inline int64_t* ask_base(int64_t* book, uint32_t instrument) {
    return book + instrument * GPU_INSTRUMENT_STRIDE_ELEMS + GPU_ASK_OFFSET;
}

// =============================================================================
// Phase 1: Apply pending MatchCommands from the CPU matcher.
// Run by thread 0 only (SPSC — one reader).
// Must complete before Phase 3 (batch processing).
// =============================================================================

__device__ void phase1_apply_match_commands(
    LOBKernelParams& p,
    uint32_t instrument,
    BlockSharedMem& smem)
{
    if (threadIdx.x != 0) return;

    // Read the ring tail with volatile to bypass L1 cache and see the CPU's
    // most recent store. Without volatile, the GPU may read a stale tail
    // from its own L1 and miss pending commands.
    uint32_t tail = *((volatile uint32_t*)&p.match_ring->tail);
    uint32_t head = p.match_ring->head;
    uint32_t applied = 0;

    while (head != tail) {
        const MatchCommand& cmd = p.match_ring->slots[head & MATCH_RING_MASK];

        // Only apply commands for this block's instrument
        if (cmd.instrument == instrument) {
            int64_t* book_side = (cmd.side == 0)
                ? bid_base(p.device_book, instrument)
                : ask_base(p.device_book, instrument);

            // TODO (HW4 Part 4, Task 2): apply atomicAdd with cmd.quantity (negative)
            // atomicAdd((unsigned long long*)&book_side[cmd.price_idx],
            //           (unsigned long long)cmd.quantity);
            (void)book_side; // suppress unused warning until TODO is done
        }

        head++;
        applied++;
    }

    // Advance the head to mark commands as consumed
    p.match_ring->head = head;
    smem.match_cmds_applied = applied;
}

// =============================================================================
// Phase 2: Load top-N cache from pinned snapshot into shared memory.
// All threads in the block participate (one thread per cache slot).
// =============================================================================

__device__ void phase2_load_cache(
    LOBKernelParams& p,
    uint32_t instrument,
    BlockSharedMem& smem)
{
    const InstrumentSnapshot& snap = p.snapshots[instrument];

    // Threads 0..TOP_N-1 load one bid slot each
    if (threadIdx.x < TOP_N) {
        smem.bid_cache[threadIdx.x] = snap.best_bids[threadIdx.x];
    }

    // Threads TOP_N..2*TOP_N-1 load one ask slot each
    if (threadIdx.x >= TOP_N && threadIdx.x < 2 * TOP_N) {
        smem.ask_cache[threadIdx.x - TOP_N] = snap.best_asks[threadIdx.x - TOP_N];
    }

    // Thread 0 reads cache sizes
    // TODO (HW4 Part 2, Task 1): store and retrieve actual cache sizes
    // For now, assume full cache
    if (threadIdx.x == 0) {
        smem.bid_cache_sz = TOP_N;
        smem.ask_cache_sz = TOP_N;
        smem.delta_write_idx = 0;
    }

    __syncthreads();
}

// =============================================================================
// Phase 3: Process add/cancel batch.
// Each thread handles one order from this instrument's batch.
// =============================================================================

__device__ void phase3_process_batch(
    LOBKernelParams& p,
    uint32_t instrument,
    BlockSharedMem& smem)
{
    uint32_t order_count = p.batch_instrument_count[instrument];

    // Thread index within this instrument's orders
    // (orders are stored contiguously per instrument in the batch after assembly)
    // TODO (HW4 Part 5): compute correct offset into batch arrays for this instrument
    uint32_t order_idx = threadIdx.x;

    if (order_idx < order_count) {
        // TODO (HW4 Part 1, Task 3): look up the global batch index for this
        // instrument's order_idx-th order. For now, use order_idx directly
        // (assumes orders are pre-sorted by instrument, one block per instrument).
        uint32_t global_idx = order_idx; // placeholder

        uint32_t price_idx = p.batch_price_idx[global_idx];
        int64_t  qty       = p.batch_quantity[global_idx];
        uint8_t  side      = p.batch_side[global_idx];

        // Select the correct side of the book
        int64_t* book_side = (side == 0)
            ? bid_base(p.device_book, instrument)
            : ask_base(p.device_book, instrument);

        // TODO (HW4 Part 1, Task 3): implement warp-aggregated atomics
        // for spread-level contention (orders clustering near top 3-5 levels).
        // For now, use a plain atomicAdd.
        atomicAdd((unsigned long long*)&book_side[price_idx],
                  (unsigned long long)qty);
    }

    __syncthreads(); // all atomicAdds must complete before cache recompute
}

// =============================================================================
// Phase 4: Recompute top-N cache from updated book_[].
// Option B (full parallel recompute) — correct by construction.
// TODO (HW4 Part 2): implement. Stub leaves smem cache unchanged.
// =============================================================================

__device__ void phase4_recompute_cache(
    LOBKernelParams& p,
    uint32_t instrument,
    BlockSharedMem& smem)
{
    // TODO (HW4 Part 2, Task 2): implement Option B (parallel scan over book_[])
    // or Option C (sort over active set from batch).
    //
    // The goal: find the top TOP_N occupied levels for bids (highest prices)
    // and asks (lowest prices) and write them into smem.bid_cache / smem.ask_cache.
    //
    // Starting point for Option B:
    //   Each thread scans book_[threadIdx.x .. threadIdx.x + step .. MAX_LEVELS]
    //   Use cub::BlockReduce to find the global maximum bid level.
    //   Extend to find top-N by repeating N times (or use partial sort).

    (void)p; (void)instrument; (void)smem; // suppress unused warnings
    __syncthreads();
}

// =============================================================================
// Phase 5: Write-back to pinned snapshot buffer (seqlock write protocol).
// Run by thread 0. Must be last — depends on Phase 4 completing.
// =============================================================================

__device__ void phase5_seqlock_writeback(
    LOBKernelParams& p,
    uint32_t instrument,
    BlockSharedMem& smem)
{
    if (threadIdx.x != 0) return;

    InstrumentSnapshot& snap = p.snapshots[instrument];

    // Step 1: signal write in progress (version goes odd)
    atomicAdd((unsigned int*)&snap.version, 1u);

    // Step 2: make the odd version visible to CPU before any data write
    __threadfence_system();

    // Step 3: write top-N cache arrays
    // TODO (HW4 Part 6, Task 2): write all TOP_N bid and ask slots from smem
    // For now, write only best_bid_price and best_ask_price as a minimal stub.
    // Full implementation copies smem.bid_cache[] and smem.ask_cache[] into
    // snap.best_bids[] and snap.best_asks[].

    if (smem.bid_cache_sz > 0) {
        atomicExch((long long*)&snap.best_bid_price, (long long)smem.bid_cache[0].price);
    }
    if (smem.ask_cache_sz > 0) {
        atomicExch((long long*)&snap.best_ask_price, (long long)smem.ask_cache[0].price);
    }
    snap.batch_seq.store(p.batch_seq, std::memory_order_relaxed);

    // Step 4: make all data writes visible before signaling done
    __threadfence_system();

    // Step 5: signal write done (version goes even)
    atomicAdd((unsigned int*)&snap.version, 1u);
}

// =============================================================================
// Phase 6: Write delta output buffer.
// Records (price_idx, new_quantity) for every level modified in this batch.
// CPU reads these in on_batch_complete() to call apply_delta().
// Run by thread 0. TODO (HW4 Part 4, Task 3): implement.
// =============================================================================

__device__ void phase6_write_deltas(
    LOBKernelParams& p,
    uint32_t instrument,
    BlockSharedMem& smem)
{
    if (threadIdx.x != 0) return;

    // TODO (HW4 Part 4, Task 3):
    // For each price_idx in the batch (plus MatchCommands applied),
    // write one BookDelta entry: { instrument, price_idx, new_quantity, side }
    // Use atomicAdd on p.delta_out->count to claim a slot.
    //
    // new_quantity = current value of book_[price_idx] after all atomics.
    // Read it with a plain load — all atomics in Phase 3 have completed
    // and no other kernel is running concurrently (ensured by the
    // cross-stream event guard in the dispatch thread).

    (void)p; (void)instrument; (void)smem; // suppress until TODO
}

// =============================================================================
// The main LOB kernel.
//
// Launch configuration:
//   gridDim.x  = N_INSTRUMENTS  (one block per instrument)
//   blockDim.x = MAX_BATCH_SIZE (must be >= max orders per instrument per batch)
//   shared mem = sizeof(BlockSharedMem)
//
// All six phases run sequentially within each block.
// Blocks for different instruments run concurrently on different SMs.
// =============================================================================

__global__ void lob_kernel(LOBKernelParams p) {
    // This block is responsible for instrument blockIdx.x
    uint32_t instrument = blockIdx.x;

    // Shared memory for this block (one block = one instrument)
    __shared__ BlockSharedMem smem;

    // Phase 1: Apply MatchCommands from CPU matcher (thread 0 only)
    phase1_apply_match_commands(p, instrument, smem);
    __syncthreads();

    // Phase 2: Load current cache from pinned snapshot
    phase2_load_cache(p, instrument, smem);
    // __syncthreads() is at the end of phase2

    // Phase 3: Process add/cancel batch (all threads)
    phase3_process_batch(p, instrument, smem);
    // __syncthreads() is at the end of phase3

    // Phase 4: Recompute top-N cache (all threads cooperate)
    phase4_recompute_cache(p, instrument, smem);
    // __syncthreads() is at the end of phase4

    // Phase 5: Seqlock write-back to pinned buffer (thread 0 only)
    phase5_seqlock_writeback(p, instrument, smem);
    __syncthreads();

    // Phase 6: Write delta output buffer (thread 0 only)
    phase6_write_deltas(p, instrument, smem);
}

// =============================================================================
// Host-side kernel launcher (called from GPUDispatch)
// =============================================================================

void launch_lob_kernel(
    const LOBKernelParams& params,
    cudaStream_t           stream)
{
    dim3 grid(GPU_N_INSTRUMENTS);
    dim3 block(MAX_BATCH_SIZE);
    size_t smem_bytes = sizeof(BlockSharedMem);

    lob_kernel<<<grid, block, smem_bytes, stream>>>(params);

    // TODO (HW4 Part 10, Task 4): wrap in CUDA Graph for fast stream
    // to eliminate per-launch CPU overhead (~5-10µs per launch).
}

