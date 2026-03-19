#include "CPUOrderBook.h" 

        
CPUOrderBook::CPUOrderBook(int64_t limit_down, int64_t limit_up, int64_t tick) :
    bid_(limit_down, limit_up, tick, CPUHalfBook::BID),
    ask_(limit_down, limit_up, tick, CPUHalfBook::ASK)
{}

void CPUOrderBook::add(int64_t price, int64_t quantity, bool is_bid) {
    if (is_bid) {
        this->bid_.add(price, quantity);
    } else {
        this->ask_.add(price, quantity);
    }
}

void CPUOrderBook::cancel(int64_t price, int64_t quantity, bool is_bid) {
    if (is_bid) {
        this->bid_.cancel(price, quantity);
    } else {
        this->ask_.cancel(price, quantity);
    }
}

void CPUOrderBook::trade(int64_t quantity, bool is_bid_aggressor) {
    if (is_bid_aggressor) {
        this->ask_.trade(quantity);
    } else {
        this->bid_.trade(quantity);
    }
}

MarketData CPUOrderBook::publish() {
    return { bid_.publish(), ask_.publish() };
}

int64_t CPUOrderBook::best_bid() const { return bid_.best_price(); }
int64_t CPUOrderBook::best_ask() const { return ask_.best_price(); }




