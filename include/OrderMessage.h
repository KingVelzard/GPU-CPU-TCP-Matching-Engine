#pragma once
#include "../tools/utils.h" // PRICE_SCALE, Side constants
#include <cstdint>

// OrderMessage: a 32 byte Message that is sent by the client over the wire
// Read and then turned into ReactorOrderMessage in Reactor

struct alignas(32) OrderMessage {

  int64_t price_ticks; // price in fixed-point ticks (price * PRICE_SCALE) / tick_size
  int64_t quantity;    // quantity in contracts/shares
  uint32_t instrument; // which instrument into GPUBook, CPUOrderBookManager
  uint32_t order_id;   // represents which order for client
  uint8_t side;        // 0 = BID. 1 = ASK
  uint8_t type;        // 0 = ADD, 1 = CANCEL, 2 = MARKET (aggressive)
  uint8_t _pad[6];     // padding DON'T CHANGE UNLESS UPDATE GPU kernel
};

// ReactorOrderMessage: a 64 byte ordermessage struct that contains OrderMessage Data
// alongside Routing data for returning to client. Written into GPU Batch and for CPU Matcher
struct alignas(64) ReactorOrderMessage {

  uint8_t side;        // 0 = BID. 1 = ASK
  uint8_t type;        // 0 = ADD, 1 = CANCEL, 2 = MARKET (aggressive)
  uint8_t _pad0[2];    // padding DON'T CHANGE UNLESS UPDATE GPU kernel
  uint32_t order_id;   // represents which order for client
  int64_t price_ticks; // price in fixed-point ticks (price * PRICE_SCALE) / tick_size
  int64_t quantity;    // quantity in contracts/shares
  uint32_t instrument; // which instrument into GPUBook, CPUOrderBookManager
  uint8_t _pad1[4];    // padding

  // Routing Data

  uint64_t conn_token; // encodes fd + generation for fill conn_token = fd << 32 | generation
  uint64_t arrive_ns;  // CLOCK_MONOTONIC for testing
  uint32_t reactor_id; // which reactor owns connection
  uint8_t _pad2[4];    // padding
};

static_assert(sizeof(OrderMessage) == 32, "ordermessage should be 32 bytes");
static_assert(sizeof(ReactorOrderMessage) == 64, "ReactorOrderMessage should be 64 bytes");

// conveinence constants

namespace OrderType {
static constexpr uint8_t ADD = 0;
static constexpr uint8_t CANCEL = 1;
static constexpr uint8_t MARKET = 2; // goes to cpu matcher
} // namespace OrderType

namespace Side {
static constexpr uint8_t BID = 0;
static constexpr uint8_t ASk = 1;
} // namespace Side
