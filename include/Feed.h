#pragma once
#include <cstdint>
#include <array>

// TOP_N: the number of best price levels maintained in cache as of now

static constexpr int TOP_N = 50;

// Feed: entry in the TOP_N cache 
// represents a price level
struct alignas(16) Feed {
  int64_t price; // fixed point price
  int64_t quantity; // total quantity on price level 
};

static_assert(sizeof(Feed) == 16, "Feed must be 16 bytes");
static_assert(alignof(Feed) == 16, "Feed alignment must be 16 bytes");

// FeedSM: padded version of Feed for GPU shared memory
// because 32 threads in warp, pad 16 bytes so no banking conflicts

struct alignas(32) FeedSM {
  int64_t price; // fixed point price
  int64_t quantity; // total quantity on price level 
  int64_t : 128; // pads to 32 bytes
};

static_assert(sizeof(FeedSM) == 32, "FeedSM must be 32 bytes");
static_assert(alignof(FeedSM) == 32, "FeedSM alignment must be 16 bytes");

// MarketData: the full top-N snapshot for the instrument
// there should be one in ask book and one in bid book

struct MarketData {
  std::array<Feed, TOP_N> bids;
  std::array<Feed, TOP_N> asks;
};

static_assert(sizeof(MarketData) == 2 * TOP_N * sizeof(Feed),
    "MarketData size unexpected — check Feed alignment and TOP_N");
