#ifndef CPUORDERBOOK_H
#define CPUORDERBOOK_H

#include "CPUHalfBook.h"

struct MarketData {
     std::array<Feed, TOP_N> bid_data;
     std::array<Feed, TOP_N> ask_data;
};

class CPUOrderBook {
    
    public:
        
        CPUOrderBook(int64_t limit_down, int64_t limit_up, int64_t tick);

        // wrapper functions
        void add(int64_t price, int64_t quantity, bool is_bid);
        int64_t cancel(int64_t price, int64_t quantity, bool is_bid);
        int64_t trade(int64_t quantity, bool is_bid_aggressor);
        MarketData publish();

        int64_t best_bid() const;
        int64_t best_ask() const;
        int64_t best_bid_qty() const;
        int64_t best_ask_qty() const;
        bool    ask_empty() const;
        bool    bid_empty() const;
        int64_t ask_at(int64_t p) const;
        int64_t bid_at(int64_t p) const;
        
    private:
        // both half books
        CPUHalfBook bid_;
        CPUHalfBook ask_;
};

#endif
