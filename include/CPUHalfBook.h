#ifndef CPUHALFBOOK_H
#define CPUHALFBOOK_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <format>

static constexpr int TOP_N = 5;

struct Feed {
    int64_t price    = 0;
    int64_t quantity = 0;
};


class CPUHalfBook {

    public:

        enum Side { BID, ASK };

        CPUHalfBook(int64_t limit_down, int64_t limit_up, int64_t tick, Side side);

        int64_t capacity();

        void add(int64_t price, int64_t quantity);

        int64_t cancel(int64_t price, int64_t quantity);
        int64_t match(int64_t quantity);
        std::array<Feed, TOP_N> publish();

        int64_t best_price() const;
        int64_t best_qty() const;
        int64_t book_at(int64_t price) const;

        std::vector<int> forward_walk() const;
        std::vector<Feed> cache_snapshot() const;
        int cache_size() const;        
        bool empty() const;

    private:

        Side side_;
        // price levels in book
        std::vector<int64_t> book_;
        // doubly linked list to find prev and next price level
        std::vector<int> nxt_;
        std::vector<int> prv_;
        // cache for TOP N best prices
        std::array<Feed, TOP_N> cache_{};
        int cache_sz_ = 0;
        int64_t limit_up;
        int64_t limit_down;
        int64_t tick;
        int64_t n_;

        // null checks for linked list
        static constexpr int kNull = -1;

        // cache methods
        //
        bool is_better(int64_t a, int64_t b) const;
        void sort_cache();
        int find_in_cache(int64_t price) const;
        bool belongs_in_cache(int64_t price) const;
        void cache_on_add(int64_t price, int64_t quantity);
        Feed next_from_book() const;
        void cache_evict(int64_t price);
        bool is_linked(int index) const;
        void poison(int index);

        // adding / removing from double linked list
        void ll_insert(int index); 
    
        void ll_remove(int index);
        // for easy addr/price translation
        int addr_at(int64_t price) const;
        int64_t price_at(int addr) const;
        // sentinel index helpers
        int head() const;
        int tail() const;
};

#endif
