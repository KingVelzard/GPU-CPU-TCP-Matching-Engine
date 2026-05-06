#pragma once
#include "FillResult.h"
#include "InstrumentSnapshot.h"
#include "MatchCommand.h"
#include "OrderMessage.h"
#include "SPSCQueue.h"
#include "utils.h"
#include <cstdint>
#include <pthread.h>
#include <sched.h>

// Matcher: CPU matching engine thread.
//
// Drains per-reactor aggressive SPSCs in round-robin, walks the top-N snapshot
// to fill as much of each order as possible, pushes MatchCommands to the GPU
// correction ring, and sends FillResults back to the originating reactor.
//
// Pinning to MATCHER_CORE (utils.h) keeps it off the reactor/dispatch cores.

class Matcher {
public:
    struct Config {
        MatchCommandRing*                     match_ring;
        InstrumentSnapshot*                   snapshots;       // [GPU_N_INSTRUMENTS]
        SPSCQueue<ReactorOrderMessage, 256>** aggressive_queues; // [n_reactors]
        SPSCQueue<FillResult, 256>**          fill_queues;       // [n_reactors]
        int                                   n_reactors;
    };

    explicit Matcher(const Config& cfg) : cfg_(cfg) {}

    void pin_to_core(int core_id) noexcept;
    void run();

private:
    Config cfg_;

    void match(const ReactorOrderMessage& msg, int reactor_idx) noexcept;
    void send_fill(const ReactorOrderMessage& msg,
                   int64_t filled_qty,
                   int64_t avg_price_ticks,
                   bool    fully_filled,
                   int     reactor_idx) noexcept;
};
