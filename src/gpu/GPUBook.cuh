#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cassert>

// =============================================================================
// GPU Book Layout
// =============================================================================
//
// The GPU maintains one flat int64_t[] array per side per instrument.
// Layout in GPU global memory:
//
//   [Instrument 0 Bids: MAX_LEVELS × int64_t]   ← 64-byte aligned
//   [Instrument 0 Asks: MAX_LEVELS × int64_t]
//   [Instrument 1 Bids: MAX_LEVELS × int64_t]
//   [Instrument 1 Asks: MAX_LEVELS × int64_t]
//   ...
//
// Each thread block handles exactly one instrument (blockIdx.x == instrument_id).
// The block computes its base pointer as: blockIdx.x × INSTRUMENT_STRIDE.
//
// Why bids and asks are interleaved per instrument (not all bids then all asks):
//   A block processing instrument 0 needs both bid and ask arrays within one
//   kernel. Interleaving keeps them close in memory → likely same L2 cache
//   segment → fewer capacity misses.

// =============================================================================
// TODO (HW4 Part 1, Task 1): Set these to match your CPUHalfBook parameters.
// For now they are set to placeholder values.
// Compute MAX_LEVELS = (limit_up - limit_down) / tick + 1
// =============================================================================
static constexpr uint32_t GPU_MAX_LEVELS       = 10000; // placeholder — set your value
static constexpr uint32_t GPU_N_INSTRUMENTS    = 8;     // must match CPUOrderBookManager count

// Stride in int64_t elements between instruments.
// 2 = one bid array + one ask array per instrument.
static constexpr uint32_t GPU_INSTRUMENT_STRIDE_ELEMS = 2 * GPU_MAX_LEVELS;

// Stride in bytes (for pointer arithmetic from void*)
static constexpr uint32_t GPU_INSTRUMENT_STRIDE_BYTES =
    GPU_INSTRUMENT_STRIDE_ELEMS * sizeof(int64_t);

// Side offsets within one instrument's stride
static constexpr uint32_t GPU_BID_OFFSET = 0;
static constexpr uint32_t GPU_ASK_OFFSET = GPU_MAX_LEVELS;

// =============================================================================
// GPUBook: the host-side handle for the GPU book allocation.
// =============================================================================

struct GPUBook {
    int64_t* device_ptr = nullptr; // cudaMalloc allocation, NULL until init()

    // Total size in bytes
    static constexpr size_t total_bytes() {
        return static_cast<size_t>(GPU_N_INSTRUMENTS)
             * GPU_INSTRUMENT_STRIDE_BYTES;
    }

    // init: allocate and zero-initialize the GPU book.
    // Must be called once at startup before any kernel launches.
    void init() {
        assert(device_ptr == nullptr && "GPUBook::init called twice");
        cudaError_t err = cudaMalloc(&device_ptr, total_bytes());
        assert(err == cudaSuccess && "cudaMalloc failed for GPUBook");
        err = cudaMemset(device_ptr, 0, total_bytes());
        assert(err == cudaSuccess && "cudaMemset failed for GPUBook");
    }

    // destroy: free the GPU allocation.
    void destroy() {
        if (device_ptr) {
            cudaFree(device_ptr);
            device_ptr = nullptr;
        }
    }

    // bid_ptr: returns device pointer to the bid array for instrument i.
    int64_t* bid_ptr(uint32_t instrument) const {
        return device_ptr
             + instrument * GPU_INSTRUMENT_STRIDE_ELEMS
             + GPU_BID_OFFSET;
    }

    // ask_ptr: returns device pointer to the ask array for instrument i.
    int64_t* ask_ptr(uint32_t instrument) const {
        return device_ptr
             + instrument * GPU_INSTRUMENT_STRIDE_ELEMS
             + GPU_ASK_OFFSET;
    }

    // copy_from_cpu: one-time sync at startup.
    // Copies CPU book_[] arrays into the corresponding GPU regions.
    // After this, all mutations go through the batch kernel.
    // cpu_book_arrays[i] must point to the book_[] vector data for instrument i,
    // side 0 = bid, side 1 = ask.
    void copy_from_cpu(uint32_t instrument,
                       const int64_t* cpu_bid_data,
                       const int64_t* cpu_ask_data,
                       uint32_t num_levels,
                       cudaStream_t stream = nullptr) {
        cudaMemcpyAsync(bid_ptr(instrument),
                        cpu_bid_data,
                        num_levels * sizeof(int64_t),
                        cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(ask_ptr(instrument),
                        cpu_ask_data,
                        num_levels * sizeof(int64_t),
                        cudaMemcpyHostToDevice,
                        stream);
    }

    // sync_to_cpu: for testing. Copies GPU book back to CPU buffer for diff.
    void sync_to_cpu(uint32_t instrument,
                     int64_t* out_bid,
                     int64_t* out_ask,
                     uint32_t num_levels) const {
        cudaMemcpy(out_bid, bid_ptr(instrument),
                   num_levels * sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(out_ask, ask_ptr(instrument),
                   num_levels * sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
    }
};

// =============================================================================
// GPUBatch: SoA layout for incoming orders.
// All arrays allocated in cudaMallocHost memory so both CPU (writes) and
// GPU (reads) can access without cudaMemcpy.
// =============================================================================

static constexpr uint32_t MAX_BATCH_SIZE = 512; // maximum orders per batch

struct alignas(64) GPUBatch {
    uint32_t* instrument_id; // [MAX_BATCH_SIZE] which instrument (= which block)
    uint32_t* price_idx;     // [MAX_BATCH_SIZE] pre-computed addr_at(price)
    int64_t*  quantity;      // [MAX_BATCH_SIZE] positive=add, negative=cancel
    uint8_t*  side;          // [MAX_BATCH_SIZE] 0=BID, 1=ASK
    uint32_t  count;         // number of valid entries

    // Per-instrument order count (for the kernel to know how many orders it has)
    uint32_t instrument_count[GPU_N_INSTRUMENTS];

    void init() {
        cudaMallocHost(&instrument_id, MAX_BATCH_SIZE * sizeof(uint32_t));
        cudaMallocHost(&price_idx,     MAX_BATCH_SIZE * sizeof(uint32_t));
        cudaMallocHost(&quantity,      MAX_BATCH_SIZE * sizeof(int64_t));
        cudaMallocHost(&side,          MAX_BATCH_SIZE * sizeof(uint8_t));
        count = 0;
        memset(instrument_count, 0, sizeof(instrument_count));
    }

    void destroy() {
        cudaFreeHost(instrument_id);
        cudaFreeHost(price_idx);
        cudaFreeHost(quantity);
        cudaFreeHost(side);
    }

    void reset() {
        count = 0;
        memset(instrument_count, 0, sizeof(instrument_count));
    }
};

