#pragma once
#include "../../include/GPUBatch.h"
#include "../../include/InstrumentSnapshot.h"
#include "../../include/MatchCommand.h"
#include "LOBKernelTypes.h"
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>

// Heap-allocated callback payload; freed inside on_batch_complete.
struct BatchCompletionData {
    GPUBatch* batch; // the batch that just finished — reset it next time it's used
};

// GPUDispatch: the GPU dispatch thread.
//
// One dedicated CPU thread, pinned to DISPATCH_CORE.
// Never blocks; busy-polls with _mm_pause when idle.
//
// Responsibilities:
//  1. Watch fast/slow GPUBatch fill levels and timeouts.
//  2. On threshold/timeout: flip the double-buffer, reset the fresh buffer,
//     expose it to reactors (release store), then launch lob_kernel on old buf.
//  3. Record slow_done_ event so fast stream can wait on it (ordering).
//  4. Register cudaStreamAddCallback to free the completed-batch payload.

class GPUDispatch {
public:
    GPUDispatch() = default;
    ~GPUDispatch() { destroy(); }

    // ── Reactor-facing ─────────────────────────────────────────────────────────
    // Reactors load these with acquire to discover which buffer is open for writes.
    // Dispatch stores with release during the buffer swap in fire_fast/slow.
    std::atomic<int> fast_current_buf_{0};
    std::atomic<int> slow_current_buf_{0};

    GPUBatch* fast_batches() noexcept { return fast_batch_; }
    GPUBatch* slow_batches() noexcept { return slow_batch_; }

    // ── Lifecycle ──────────────────────────────────────────────────────────────
    struct Config {
        int64_t*            device_book; // GPUBook::device_ptr — raw pointer avoids pulling in GPUBook.cuh
        MatchCommandRing*   match_ring;
        InstrumentSnapshot* snapshots;
    };

    void init(const Config& cfg);
    void destroy();
    void run();
    void stop() { running_.store(false, std::memory_order_relaxed); }

    uint64_t fast_batches_launched() const { return fast_launched_.load(std::memory_order_relaxed); }
    uint64_t slow_batches_launched() const { return slow_launched_.load(std::memory_order_relaxed); }

private:
    Config cfg_{};

    cudaStream_t fast_stream_{nullptr};
    cudaStream_t slow_stream_{nullptr};
    cudaEvent_t  slow_done_{nullptr};

    GPUBatch fast_batch_[2];
    GPUBatch slow_batch_[2];
    int fast_buf_{0};
    int slow_buf_{0};

    // Pinned per-instrument arrays passed as GPU zero-copy pointers in LOBKernelParams.
    uint32_t* fast_count_pinned_{nullptr};  // [GPU_N_INSTRUMENTS]
    uint32_t* slow_count_pinned_{nullptr};  // [GPU_N_INSTRUMENTS]
    int64_t*  limit_down_pinned_{nullptr};  // [GPU_N_INSTRUMENTS], filled at init
    int64_t*  tick_sz_pinned_{nullptr};     // [GPU_N_INSTRUMENTS], filled at init

    bool                  slow_event_valid_{false}; // true after first fire_slow()
    std::atomic<bool>     running_{false};
    std::atomic<uint64_t> fast_launched_{0};
    std::atomic<uint64_t> slow_launched_{0};
    uint32_t              batch_seq_{0};

    LOBKernelParams build_params(GPUBatch& batch, uint32_t* count_pinned) noexcept;
    void fire_fast();
    void fire_slow();

    static void CUDART_CB on_batch_complete(cudaStream_t stream,
                                            cudaError_t  status,
                                            void*        userdata);
};
