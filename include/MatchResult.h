#pragma once
#include <cstdint>

// MatchResult: sent back to client after a match 
// Written by CPU matcher after match executes
// Routed Back to original TCP connection via connection token

struct alignas(32) MatchResult {
  uint32_t order_id; // just like OrderMessage echoes
  uint32_t instrument; // which instrument 
  int64_t filled_qty; // how many shares were filled 
  int64_t avg_fill_price; // volume weighted fill price 
  uint8_t fully_filled; // 1 if fully filled 0 if partial fill 
  uint8_t side; // bid = 0, ask = 1
  uint8_t _pad[6];
  uint64_t fill_timestamp; // CPU timestamp at fill (CLOCK_MONOTONIC)
};


static_assert(sizeof(MatchResult) == 32, "match result should be 32 bytes");
static_assert(alignof(MatchResult) == 32, "match result alignment should be 32 bytes");

