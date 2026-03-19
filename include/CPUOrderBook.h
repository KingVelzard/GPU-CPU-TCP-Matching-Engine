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
        void cancel(int64_t price, int64_t quantity, bool is_bid);
        void trade(int64_t quantity, bool is_bid_aggressor);
        MarketData publish();

        int64_t best_bid() const;
        int64_t best_ask() const;
        
    private:
        // both half books
        CPUHalfBook bid_;
        CPUHalfBook ask_;
};

#endif
