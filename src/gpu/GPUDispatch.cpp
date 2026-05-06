#include "GPUDispatch.h"
#include <cassert>
#include <ctime>
#include <immintrin.h>

static uint64_t now_ns() noexcept {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

void GPUDispatch::init(const Config& cfg) {
    cfg_ = cfg;

    // Fast stream gets higher priority (lower numeric value).
    int lo{}, hi{};
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStreamCreateWithPriority(&fast_stream_, cudaStreamNonBlocking, hi);
    cudaStreamCreateWithPriority(&slow_stream_, cudaStreamNonBlocking, lo);

    // Cross-stream event: fast waits on slow so they don't race on the book.
    cudaEventCreateWithFlags(&slow_done_, cudaEventDisableTiming);

    for (int i = 0; i < 2; ++i) {
        fast_batch_[i].init();
        slow_batch_[i].init();
    }

    // Pinned arrays for GPU zero-copy access via LOBKernelParams pointers.
    cudaMallocHost(&fast_count_pinned_, GPU_N_INSTRUMENTS * sizeof(uint32_t));
    cudaMallocHost(&slow_count_pinned_, GPU_N_INSTRUMENTS * sizeof(uint32_t));
    cudaMallocHost(&limit_down_pinned_, GPU_N_INSTRUMENTS * sizeof(int64_t));
    cudaMallocHost(&tick_sz_pinned_,    GPU_N_INSTRUMENTS * sizeof(int64_t));

    // Fill static per-instrument limits from InstrumentRegistry (read-only after finalize).
    const auto& reg = InstrumentRegistry::get();
    for (uint32_t i = 0; i < GPU_N_INSTRUMENTS; ++i) {
        limit_down_pinned_[i] = reg.limits[i].limit_down;
        tick_sz_pinned_[i]    = reg.limits[i].tick;
    }
}

void GPUDispatch::destroy() {
    if (fast_stream_) { cudaStreamDestroy(fast_stream_); fast_stream_ = nullptr; }
    if (slow_stream_) { cudaStreamDestroy(slow_stream_); slow_stream_ = nullptr; }
    if (slow_done_)   { cudaEventDestroy(slow_done_);    slow_done_   = nullptr; }

    for (int i = 0; i < 2; ++i) {
        fast_batch_[i].destroy();
        slow_batch_[i].destroy();
    }

    if (fast_count_pinned_) { cudaFreeHost(fast_count_pinned_); fast_count_pinned_ = nullptr; }
    if (slow_count_pinned_) { cudaFreeHost(slow_count_pinned_); slow_count_pinned_ = nullptr; }
    if (limit_down_pinned_) { cudaFreeHost(limit_down_pinned_); limit_down_pinned_ = nullptr; }
    if (tick_sz_pinned_)    { cudaFreeHost(tick_sz_pinned_);    tick_sz_pinned_    = nullptr; }
}

// ── Hot path ──────────────────────────────────────────────────────────────────

LOBKernelParams GPUDispatch::build_params(GPUBatch& batch, uint32_t* count_pinned) noexcept {
    // Snapshot per-instrument order counts into pinned buffer for GPU access.
    for (uint32_t i = 0; i < GPU_N_INSTRUMENTS; ++i)
        count_pinned[i] = batch.count[i].load(std::memory_order_relaxed);

    LOBKernelParams p{};
    p.device_book     = cfg_.device_book;
    p.batch_price_idx = batch.price_idx;
    p.batch_quantity  = batch.quantity;
    p.batch_side      = batch.side;
    p.batch_count     = count_pinned;
    p.limit_down      = limit_down_pinned_;
    p.tick_sz         = tick_sz_pinned_;
    p.match_ring      = cfg_.match_ring;
    p.snapshots       = cfg_.snapshots;
    p.batch_seq       = batch_seq_++;
    return p;
}

void GPUDispatch::fire_fast() {
    const int old_buf = fast_buf_;
    fast_buf_ ^= 1;

    // Reset new buffer before exposing it so reactors start with clean counters.
    fast_batch_[fast_buf_].reset();
    fast_current_buf_.store(fast_buf_, std::memory_order_release);

    LOBKernelParams params = build_params(fast_batch_[old_buf], fast_count_pinned_);

    // Wait for any in-flight slow kernel — both streams share the same book array.
    if (slow_event_valid_)
        cudaStreamWaitEvent(fast_stream_, slow_done_, 0);

    launch_lob_kernel(params, fast_stream_);

    auto* cbd = new BatchCompletionData{&fast_batch_[old_buf]};
    cudaStreamAddCallback(fast_stream_, on_batch_complete, cbd, 0);

    fast_launched_.fetch_add(1, std::memory_order_relaxed);
}

void GPUDispatch::fire_slow() {
    const int old_buf = slow_buf_;
    slow_buf_ ^= 1;

    slow_batch_[slow_buf_].reset();
    slow_current_buf_.store(slow_buf_, std::memory_order_release);

    LOBKernelParams params = build_params(slow_batch_[old_buf], slow_count_pinned_);

    launch_lob_kernel(params, slow_stream_);
    cudaEventRecord(slow_done_, slow_stream_);
    slow_event_valid_ = true;

    auto* cbd = new BatchCompletionData{&slow_batch_[old_buf]};
    cudaStreamAddCallback(slow_stream_, on_batch_complete, cbd, 0);

    slow_launched_.fetch_add(1, std::memory_order_relaxed);
}

void GPUDispatch::run() {
    running_.store(true, std::memory_order_relaxed);

    while (running_.load(std::memory_order_relaxed)) {
        const uint64_t now = now_ns();

        GPUBatch& fb = fast_batch_[fast_buf_];
        GPUBatch& sb = slow_batch_[slow_buf_];

        const uint64_t fast_first = fb.first_order_ns.load(std::memory_order_relaxed);
        const uint64_t slow_first = sb.first_order_ns.load(std::memory_order_relaxed);

        uint32_t fast_total = 0, slow_total = 0;
        for (uint32_t i = 0; i < GPU_N_INSTRUMENTS; ++i) {
            fast_total += fb.count[i].load(std::memory_order_relaxed);
            slow_total += sb.count[i].load(std::memory_order_relaxed);
        }

        const bool fast_rdy =
            fast_total >= MAX_BATCH_SIZE ||
            (fast_total > 0 && fast_first > 0 && now - fast_first >= FAST_TIMEOUT_NS);
        const bool slow_rdy =
            slow_total >= MAX_BATCH_SIZE ||
            (slow_total > 0 && slow_first > 0 && now - slow_first >= SLOW_TIMEOUT_NS);

        if (fast_rdy) fire_fast();
        if (slow_rdy) fire_slow();
        if (!fast_rdy && !slow_rdy) _mm_pause();
    }
}

void CUDART_CB GPUDispatch::on_batch_complete(cudaStream_t, cudaError_t, void* userdata) {
    // Cannot call CUDA API here (callback fires on an internal CUDA thread).
    // The completed batch will be reset at the start of the next fire_fast/slow
    // when dispatch flips back to it, so no action needed here.
    delete static_cast<BatchCompletionData*>(userdata);
}
