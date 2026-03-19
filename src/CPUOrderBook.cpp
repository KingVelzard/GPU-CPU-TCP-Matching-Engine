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

int64_t CPUOrderBook::cancel(int64_t price, int64_t quantity, bool is_bid) {
    return is_bid ? bid_.cancel(price, quantity) 
                  : ask_.cancel(price, quantity);
}

int64_t CPUOrderBook::trade(int64_t quantity, bool is_bid_aggressor) {
    return is_bid_aggressor ? ask_.match(quantity) 
                            : bid_.match(quantity);   
}

MarketData CPUOrderBook::publish() {
    return { bid_.publish(), ask_.publish() };
}

int64_t CPUOrderBook::best_bid() const { return bid_.best_price(); }
int64_t CPUOrderBook::best_ask() const { return ask_.best_price(); }
int64_t CPUOrderBook::best_bid_qty()  const { return bid_.best_qty(); }
int64_t CPUOrderBook::best_ask_qty()  const { return ask_.best_qty(); }
bool    CPUOrderBook::ask_empty()     const { return ask_.empty(); }
bool    CPUOrderBook::bid_empty()     const { return bid_.empty(); }
int64_t CPUOrderBook::ask_at(int64_t p) const { return ask_.book_at(p); }
int64_t CPUOrderBook::bid_at(int64_t p) const { return bid_.book_at(p); }




