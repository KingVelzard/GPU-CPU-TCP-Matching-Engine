#include "CPUHalfBook.h"

/*
UTILITY
*/
// converts a price to a fixed-point integer 
int64_t price_to_int(double price, int scale) {
    return static_cast<int64_t>(price * scale);
}

// converts a fixed-point integer back to a string representation
std::string int_to_price(int64_t price, int scale) {
    return std::format("{:.2f}", static_cast<double>(price) / scale);
}

CPUHalfBook::CPUHalfBook(int64_t limit_down, int64_t limit_up, int64_t tick, Side side){
            this->side_ = side;
            this->limit_up = limit_up;
            this->limit_down = limit_down;
            this->tick = tick;
            this->n_ = static_cast<int>((limit_up - limit_down) / tick) + 1;
            this->book_.resize(n_ + 2, 0);

            this->nxt_.assign(n_ + 2, kNull);
            this->prv_.assign(n_ + 2, kNull);
            this->nxt_[head()] = tail();
            this->prv_[tail()] = head();
        }

int64_t CPUHalfBook::capacity() {
    return this->n_;
}

void CPUHalfBook::add(int64_t price, int64_t quantity) {
    int addr = addr_at(price);

    if (book_[addr] == 0) {
        ll_insert(addr);
    }
    book_[addr] += quantity;
    cache_on_add(price, quantity);
}

void CPUHalfBook::cancel(int64_t price, int64_t quantity) {
    int addr = addr_at(price);
    int index = find_in_cache(price);
    if (index != -1) {
        cache_[index].quantity -= quantity;
    }

    book_[addr] -= quantity;
    if (book_[addr] <= 0) {
        book_[addr] = 0;
        ll_remove(addr);
        cache_evict(price);
    }
}

void CPUHalfBook::trade(int64_t quantity) {
    if (cache_sz_ == 0) return;

    int64_t top_price = cache_[0].price;
    int cacheAddr = addr_at(cache_[0].price);

    book_[cacheAddr] -= quantity;
    cache_[0].quantity -= quantity;

    if (book_[cacheAddr] <= 0) {
        book_[cacheAddr] = 0;
        ll_remove(cacheAddr);
        cache_evict(top_price);
    }
}

int64_t CPUHalfBook::best_price() const {
    return cache_[0].price;
}

std::array<Feed, TOP_N> CPUHalfBook::publish() {
    std::array<Feed, TOP_N> result;
    std::copy(cache_.begin(), cache_.begin() + cache_sz_, result.begin());
    return result;
}

int64_t CPUHalfBook::book_at(int64_t price) const {
    return book_[addr_at(price)];
}

std::vector<int> CPUHalfBook::forward_walk() const {
    std::vector<int> out;
    int cur = nxt_[head()];
    while (cur != tail()) { out.push_back(cur); cur = nxt_[cur]; }
    return out;
}

std::vector<Feed> CPUHalfBook::cache_snapshot() const {
    return std::vector<Feed>(cache_.begin(), cache_.begin() + cache_sz_);
}

int CPUHalfBook::cache_size() const { return cache_sz_; }
        

bool CPUHalfBook::is_better(int64_t a, int64_t b) const {
    switch(side_) {
        case ASK:
            return a < b;
        case BID:
            return a > b;
        default:
            return false;
    }
}

void CPUHalfBook::sort_cache() {

    for (int i = 1; i < cache_sz_; i++) {
        Feed key = cache_[i];
        int j = i - 1;

        while (j >= 0 && is_better(key.price, cache_[j].price)) {
            cache_[j + 1] = cache_[j];
            j--;
        }

        cache_[j + 1] = key;
    }
}

int CPUHalfBook::find_in_cache(int64_t price) const {

    for (int i = 0; i < cache_sz_; i++) {
        if (cache_[i].price == price) {
            return i;
        }
    }
    return -1;
}

bool CPUHalfBook::belongs_in_cache(int64_t price) const {
    if (cache_sz_ < TOP_N || is_better(price, cache_[cache_sz_ -1].price))
    {
        return true;
    } 

    return false;
}

void CPUHalfBook::cache_on_add(int64_t price, int64_t quantity) {
    if (int index = find_in_cache(price); index != -1) {
        cache_[index].quantity += quantity;
    }
    else if (belongs_in_cache(price)) {
        if (cache_sz_ < TOP_N) {
            cache_[cache_sz_] = Feed{price, quantity};
            cache_sz_++;
        } else {
            cache_[TOP_N - 1] = Feed{price, quantity};
        }
        sort_cache();
    }
    else {
        return;
    }
}

Feed CPUHalfBook::next_from_book() const {
    if (cache_sz_ == 0) {
        int slot = (side_ == ASK) ? nxt_[head()] : prv_[tail()];
        if (slot == tail() || slot == head()) return Feed{};
        return Feed{price_at(slot), book_[slot]};
    }

    int worst_slot = addr_at(cache_[cache_sz_ - 1].price);
    int next_slot = (side_ == ASK) ? nxt_[worst_slot] : prv_[worst_slot];

    if (next_slot == tail() || next_slot == head()) return Feed{};
    return Feed{price_at(next_slot), book_[next_slot]};
}

void CPUHalfBook::cache_evict(int64_t price) {
    int index = find_in_cache(price);
    if (index == -1) return;
            
    // shift over left
    for (int i = index; i < cache_sz_ - 1; i++) {
        cache_[i] = cache_[i + 1];
        }
    cache_[--cache_sz_] = Feed{};

    // reload next best 
    Feed next = next_from_book();
    if (next.quantity > 0) {
        cache_sz_++;
        cache_[cache_sz_ - 1] = next;
        sort_cache();
    }
}

bool CPUHalfBook::is_linked(int index) const { return nxt_[index] != kNull; }

void CPUHalfBook::poison(int index)          { nxt_[index] = kNull; prv_[index] = kNull; }
        
        // adding / removing from double linked list
void CPUHalfBook::ll_insert(int index) {
    assert(!is_linked(index));
    int prev = head();

    // walk
    while(nxt_[prev] < index) {
        prev = nxt_[prev];
    }
            
    int after = nxt_[prev];

    // stitch
    nxt_[prev] = index;
    prv_[index] = prev;
    nxt_[index] = after;
    prv_[after] = index;
            
}

void CPUHalfBook::ll_remove(int index) {
    assert(is_linked(index));
    int prev = prv_[index];
    int after = nxt_[index];

    nxt_[prev] = after;
    prv_[after] = prev;
    poison(index);    
}

        // for easy addr/price translation
int CPUHalfBook::addr_at(int64_t price) const {
    return ((price - this->limit_down) / this->tick) + 1;
}

int64_t CPUHalfBook::price_at(int addr) const {
    return (static_cast<int64_t>(addr - 1) * tick) + this->limit_down;
}

// sentinel index helpers
int CPUHalfBook::head() const {
    return 0;
}

int CPUHalfBook::tail() const {
    return this->n_ + 1;
}

