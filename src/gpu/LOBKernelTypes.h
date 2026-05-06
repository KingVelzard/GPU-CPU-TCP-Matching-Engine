#pragma once
// LOBKernelTypes.h: plain C++ types shared between GPUDispatch.cpp (g++) and
// LOBKernel.cu (nvcc). No CUDA device qualifiers here.
#include "../../include/InstrumentSnapshot.h"
#include "../../include/MatchCommand.h"
#include <cstdint>
#include <cuda_runtime.h>

// Parameters passed into the LOB kernel on each launch.
// All pointer fields must point to cudaMallocHost (pinned) memory so the GPU
// can access them via zero-copy without an explicit cudaMemcpy.
struct LOBKernelParams {
    int64_t*            device_book;    // GPUBook::device_ptr (device allocation)
    const uint32_t*     batch_price_idx; // [MAX_BATCH_SIZE] SoA price indices
    const int64_t*      batch_quantity;  // [MAX_BATCH_SIZE] signed quantities
    const uint8_t*      batch_side;      // [MAX_BATCH_SIZE] 0=BID 1=ASK
    const uint32_t*     batch_count;     // [GPU_N_INSTRUMENTS] orders per instrument
    const int64_t*      limit_down;      // [GPU_N_INSTRUMENTS] per-instrument
    const int64_t*      tick_sz;         // [GPU_N_INSTRUMENTS] per-instrument
    MatchCommandRing*   match_ring;      // CPU→GPU correction channel (pinned SPSC)
    InstrumentSnapshot* snapshots;       // pinned seqlock snapshot array
    uint32_t            batch_seq;       // monotonically increasing batch counter
};

// launch_lob_kernel: wraps the <<<grid,block,smem,stream>>> syntax so that
// GPUDispatch.cpp (compiled by g++) can call it without seeing device code.
// Defined in LOBKernel.cu (compiled by nvcc).
void launch_lob_kernel(const LOBKernelParams& params, cudaStream_t stream);
