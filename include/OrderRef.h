#pragma once 
#include <cstdint>

// OrderRef: a lightweight descriptor pushed by Reactor threads into MPSC 
// queue destined for GPU dispatch thread 
//
// The actual order lives in the PinnedPool, so queue only moves lightweight reference
// 32 byte descriptor = transfer handles not data.
//
// conn_token: used to route results to correct TCP connection 
//    encode as (reactor_id << 20) | fd (or just raw fd if < 2^20) 

struct alignas(32) OrderRef {
  uint64_t arrive_ns; // kernel receive timestamp (CLOCK_MONOTONIC_RAW)
  uint64_t conn_token; // points to TCP fd that client connects from  
  uint32_t slot_id; // which pinned buffer slot holds order message 
  uint32_t byte_len; // how many bytes the order message uses 
  uint32_t reactor_id; // which reactor (for correct pinnedpool)
  uint32_t : 32; // padding 
};

static_assert(sizeof(OrderRef) == 32, "OrderRef must be 32 bytes");
