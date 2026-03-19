#ifndef CPUORDERBOOKMANAGER_H
#define CPUORDERBOOKMANAGER_H
#include "CPUOrderBook.h"

class CPUOrderBookManager {
    public:

        void init(const std::string& id,
                  int64_t limit_down,
                  int64_t limit_up,
                  int64_t tick);

        void add (const std::string& id, int64_t price, int64_t quantity, bool bid);

        void cancel (const std::string& id, int64_t price, int64_t quantity, bool bid);

        void trade (const std::string& id, int64_t quantity, bool bid_aggressor);

        MarketData publish (const std::string& id);

    private:
        std::unordered_map<std::string, CPUOrderBook> books_;
};

#endif
