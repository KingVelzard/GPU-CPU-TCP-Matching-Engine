#pragma once
#include "../tools/utils.h" // NEAR_SPREAD_TICKS
#include "InstrumentSnapshot.h"
#include "OrderMessage.h"

enum class Classification : uint8_t {
  AGGRESSIVE,   // crosses or touches the spread → CPU matcher
  PASSIVE_NEAR, // within NEAR_SPREAD_TICKS of spread → fast GPUBatch
  PASSIVE_FAR,  // beyond NEAR_SPREAD_TICKS → slow GPUBatch
};

// Classifier: stateless, branchless-friendly spread router.
// Called once per limit order after a seqlock read of the spread.
struct Classifier {
  Classification classify(const ReactorOrderMessage &msg, const SpreadSnapshot &spread) const noexcept {
    if (spread.stale)
      return Classification::PASSIVE_FAR;

    if (msg.side == Side::BID) {
      if (msg.price_ticks >= spread.ask_price)
        return Classification::AGGRESSIVE;
      if (spread.ask_price - msg.price_ticks <= NEAR_SPREAD_TICKS)
        return Classification::PASSIVE_NEAR;
    } else {
      if (msg.price_ticks <= spread.bid_price)
        return Classification::AGGRESSIVE;
      if (msg.price_ticks - spread.bid_price <= NEAR_SPREAD_TICKS)
        return Classification::PASSIVE_NEAR;
    }
    return Classification::PASSIVE_FAR;
  }
};
