#pragma once
#include <cstdint>

// PRICE_SCALE: fixed-point multiplier
// Store all prices as int64_t to avoid floating point on GPU 
// also probably for data integrity 

static constexpr int64_t PRICE_SCALE = 100; // 1.23 -> 123

// OrderMessage: the 16-byte fixed-width message
// TCP Client -> Reactor (one of them lol) -> recv buffer -> MPSC queue
// -> GPU batch buffer. can be DMA'd into pinned memory with no translation stuff 
//
// IMPORTANT:::: DO NOT ADD VARIABLE LENGTH FIELDS
// GPU kernel casts raw bytes into OrderMessages directly!!!

struct alignas(16) OrderMessage {

  uint8_t side;   // 0 = BID. 1 = ASK 
  uint8_t type;   // 0 = ADD, 1 = CANCEL, 2 = MARKET (aggressive)
  uint16_t _pad;  // padding DON'T CHANGE UNLESS UPDATE GPU kernel
  uint32_t order_id; // unique, monotonically increases per client 
  int64_t price_ticks; // price in fixed-point ticks (price * PRICE_SCALE) / tick_size
  uint32_t quantity; // quantity in contracts/shares 
  uint32_t instrument; // which instrument into GPUBook, CPUOrderBookManager

};

static_assert(sizeof(OrderMessage) == 24, "ordermessage should be 24 bytes");

// conveinence constants

namespace OrderType {
  static constexpr uint8_t ADD = 0;
  static constexpr uint8_t CANCEL = 1;
  static constexpr uint8_t MARKET = 2; // goes to cpu matcher 
}

namespace Side {
  static constexpr uint8_t BID = 0;
  static constexpr uint8_t ASk = 1;
}
