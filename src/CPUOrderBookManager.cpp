#include "CPUOrderBookManager.h"


void CPUOrderBookManager::init(const std::string& id,
                  int64_t limit_down,
                  int64_t limit_up,
                  int64_t tick) 
{
        books_.emplace(std::piecewise_construct,
                        std::forward_as_tuple(id),
                        std::forward_as_tuple(limit_down, limit_up, tick));
}

void CPUOrderBookManager::add (const std::string& id, int64_t price, int64_t quantity, bool bid)
{ books_.at(id).add(price, quantity, bid); }

void CPUOrderBookManager::cancel (const std::string& id, int64_t price, int64_t quantity, bool bid)
{ books_.at(id).cancel(price, quantity, bid); }

void CPUOrderBookManager::trade (const std::string& id, int64_t quantity, bool bid_aggressor)
{ books_.at(id).trade(quantity, bid_aggressor); }

MarketData CPUOrderBookManager::publish (const std::string& id)
{ return books_.at(id).publish(); }

