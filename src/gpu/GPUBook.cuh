#pragma once
#include "../../include/GPUBatch.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

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
// Each thread block handles exactly one instrument (blockIdx.x ==
// instrument_id). The block computes its base pointer as: blockIdx.x ×
// INSTRUMENT_STRIDE.
//
// Why bids and asks are interleaved per instrument (not all bids then all
// asks):
//   A block processing instrument 0 needs both bid and ask arrays within one
//   kernel. Interleaving keeps them close in memory → likely same L2 cache
//   segment → fewer capacity misses.

// =============================================================================
// TODO (HW4 Part 1, Task 1): Set these to match your CPUHalfBook parameters.
// For now they are set to placeholder values.
// Compute MAX_LEVELS = (limit_up - limit_down) / tick + 1
// =============================================================================

// GPU_MAX_LEVELS and GPU_N_INSTRUMENTS come from utils.h (via GPUBatch.h)

static constexpr uint32_t GPU_INSTRUMENT_STRIDE_ELEMS = 2 * GPU_MAX_LEVELS; // bids + asks
static constexpr uint32_t GPU_INSTRUMENT_STRIDE_BYTES =
    GPU_INSTRUMENT_STRIDE_ELEMS * sizeof(int64_t); // stride between instruments

static constexpr uint32_t GPU_BID_OFFSET = 0;              // beginning of struct
static constexpr uint32_t GPU_ASK_OFFSET = GPU_MAX_LEVELS; // middle of struct

// GPUBook: the host-side handle for GPU book allocation

struct GPUBook {
  int64_t *device_ptr{nullptr}; // cudaMalloc allocation

  // total size in bytes
  static constexpr size_t total_bytes() { return static_cast<size_t>(GPU_N_INSTRUMENTS) * GPU_INSTRUMENT_STRIDE_BYTES; }

  // init: allocate and zero-init the GPUBook
  // must be called once at startup before any kernel launches

  void init() {
    assert(device_ptr == nullptr && "GPU::init called twice");
    cudaError_t err = cudaMalloc(&device_ptr, total_bytes());
    assert(err == cudaSuccess && "cudaMalloc failed for GPUBook");
    // zero out the price levels
    err = cudaMemset(device_ptr, 0, total_bytes());
    assert(err == cudaSuccess && "cudaMemset failed for GPUBook");
  }

  // destroy: free the GPU allocation
  void destroy() {
    if (device_ptr) {
      cudaFree(device_ptr);
      device_ptr = nullptr;
    }
  }

  // bid_ptr: returns the device pointer to the bid array given an instrument
  int64_t *bid_ptr(uint32_t instrument) const {
    // pointer arithmetic to start of instrument bids
    return device_ptr + instrument * GPU_INSTRUMENT_STRIDE_ELEMS + GPU_BID_OFFSET;
  }

  // ask_ptr: returns the device_ptr to the ask array given an instrument
  int64_t *ask_ptr(uint32_t instrument) const {
    // pointer arithmetic to start of instrument asks
    return device_ptr + instrument * GPU_INSTRUMENT_STRIDE_ELEMS + GPU_ASK_OFFSET;
  }

  // copy_from_cpu : one time sync at startup
  // copies CPU book_[] arrays into GPU regions
  // after, all changes go through batch kernel
  // cpu_book_arrays[i] must point to the book[]_ data for instrument i
  // side 0 = bid 1 = ask
  void copy_from_cpu(uint32_t instrument, const int64_t *cpu_bid_data, const int64_t *cpu_ask_data, uint32_t num_levels,
                     cudaStream_t stream = nullptr) {
    cudaMemcpyAsync(bid_ptr(instrument), cpu_bid_data, num_levels * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ask_ptr(instrument), cpu_ask_data, num_levels * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  }

  // TESTING ONLY
  // sync_to_cpu: copies GPU back to CPU buffer for diff
  void sync_to_cpu(uint32_t instrument, int64_t cpu_bid_data, int64_t cpu_ask_data, uint32_t num_levels) const {
    cudaMemcpy(&cpu_bid_data, bid_ptr(instrument), num_levels * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cpu_ask_data, ask_ptr(instrument), num_levels * sizeof(int64_t), cudaMemcpyDeviceToHost);
  }
};

// GPUBatch::init / destroy — cudaMallocHost implementations.
// Defined here so nvcc sees the CUDA calls; struct body is in GPUBatch.h.
inline void GPUBatch::init() {
    cudaMallocHost(&price_idx,     MAX_BATCH_SIZE * sizeof(uint32_t));
    cudaMallocHost(&quantity,      MAX_BATCH_SIZE * sizeof(int64_t));
    cudaMallocHost(&side,          MAX_BATCH_SIZE * sizeof(uint8_t));
    conn_token     = static_cast<uint64_t*>(std::malloc(MAX_BATCH_SIZE * sizeof(uint64_t)));
    reactor_id_arr = static_cast<uint32_t*>(std::malloc(MAX_BATCH_SIZE * sizeof(uint32_t)));
    reset();
}

inline void GPUBatch::destroy() {
    cudaFreeHost(price_idx);
    cudaFreeHost(quantity);
    cudaFreeHost(side);
    std::free(conn_token);
    std::free(reactor_id_arr);
    price_idx      = nullptr;
    quantity       = nullptr;
    side           = nullptr;
    conn_token     = nullptr;
    reactor_id_arr = nullptr;
}
