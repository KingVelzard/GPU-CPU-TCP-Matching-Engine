#pragma once
#include <cstdint>
#include <array>

// TOP_N: the number of best price levels maintained in cache as of now

static constexpr int TOP_N = 25;

// Feed: entry in the TOP_N cache 
// represents a price level
struct Feed {
  int64_t price; // fixed point price
  int64_t quantity; // total quantity on price level 
};

// FeedSM: padded version of Feed for GPU shared memory
// because 32 threads in warp, pad 16 bytes so no banking conflicts

struct alignas(32) FeedSM {
  int64_t price; // fixed point price
  int64_t quantity; // total quantity on price level 
  int64_t _pad[2]; // pads to 32 bytes
};

// MarketData: the full top-N snapshot for the instrument
// there should be one in ask book and one in bid book

struct MarketData {
  std::array<Feed, TOP_N> bids;
  std::array<Feed, TOP_N> asks;
};
