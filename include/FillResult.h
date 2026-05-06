#pragma once
#include <cstdint>

// FillResult: internal fill notification sent from Matcher → Reactor via SPSCQueue.
// conn_token identifies the originating TCP connection so the Reactor can route
// the fill bytes back to the correct client.

struct alignas(64) FillResult {
    uint64_t conn_token;       // fd << 32 | generation — routing key for Reactor
    uint64_t fill_timestamp;   // CLOCK_MONOTONIC at fill (ns)
    int64_t  filled_qty;       // shares/contracts matched
    int64_t  avg_fill_price;   // VWAP in ticks
    uint32_t order_id;
    uint32_t instrument;
    uint8_t  fully_filled;     // 1 = fully filled, 0 = partial
    uint8_t  side;             // 0 = BID, 1 = ASK
    uint8_t  _pad[22];
};

static_assert(sizeof(FillResult) == 64,  "FillResult must be 64 bytes");
static_assert(alignof(FillResult) == 64, "FillResult alignment must be 64 bytes");
