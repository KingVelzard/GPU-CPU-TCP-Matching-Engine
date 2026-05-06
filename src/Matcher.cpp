#include "Matcher.h"
#include <algorithm>
#include <ctime>
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>

void Matcher::pin_to_core(int core_id) noexcept {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
        perror("Matcher::pin_to_core");
}

// ── Main loop ─────────────────────────────────────────────────────────────────

void Matcher::run() {
    int drain_idx = 0; // round-robin start

    while (true) {
        bool any = false;

        for (int i = 0; i < cfg_.n_reactors; ++i) {
            const int r = (drain_idx + i) % cfg_.n_reactors;
            ReactorOrderMessage msg;
            if (cfg_.aggressive_queues[r]->pop(msg)) {
                match(msg, r);
                any = true;
            }
        }

        drain_idx = (drain_idx + 1) % cfg_.n_reactors;
        if (!any) _mm_pause();
    }
}

// ── Matching logic ────────────────────────────────────────────────────────────

void Matcher::match(const ReactorOrderMessage& msg, int reactor_idx) noexcept {
    const MatcherSnapshot ms = read_hot_levels(&cfg_.snapshots[msg.instrument]);

    if (ms.stale) {
        send_fill(msg, 0, 0, false, reactor_idx);
        return;
    }

    int64_t remaining    = msg.quantity; // always positive
    int64_t total_value  = 0;            // sum of (fill_qty * price) for VWAP
    int64_t total_filled = 0;

    const bool is_market = (msg.type == OrderType::MARKET);

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const uint64_t fill_ns = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
                           + static_cast<uint64_t>(ts.tv_nsec);

    if (msg.side == Side::BID) {
        // Aggressive BUY: match against asks (lowest price first).
        for (int i = 0; i < ms.ask_sz && remaining > 0; ++i) {
            const FeedSM& level = ms.hot_asks[i];
            if (level.quantity <= 0) continue;
            if (!is_market && level.price > msg.price_ticks) break;

            const int64_t fill_qty = std::min(remaining, level.quantity);
            total_value  += fill_qty * level.price;
            total_filled += fill_qty;
            remaining    -= fill_qty;

            MatchCommand cmd{};
            cmd.quantity   = -fill_qty;
            cmd.push_ns    = fill_ns;
            cmd.instrument = msg.instrument;
            cmd.price_idx  = utils::addr_at(level.price, msg.instrument);
            cmd.side       = Side::ASk; // ask side consumed
            cfg_.match_ring->push(cmd); // ring overflow is silently dropped for now
        }
    } else {
        // Aggressive SELL: match against bids (highest price first).
        for (int i = 0; i < ms.bid_sz && remaining > 0; ++i) {
            const FeedSM& level = ms.hot_bids[i];
            if (level.quantity <= 0) continue;
            if (!is_market && level.price < msg.price_ticks) break;

            const int64_t fill_qty = std::min(remaining, level.quantity);
            total_value  += fill_qty * level.price;
            total_filled += fill_qty;
            remaining    -= fill_qty;

            MatchCommand cmd{};
            cmd.quantity   = -fill_qty;
            cmd.push_ns    = fill_ns;
            cmd.instrument = msg.instrument;
            cmd.price_idx  = utils::addr_at(level.price, msg.instrument);
            cmd.side       = Side::BID; // bid side consumed
            cfg_.match_ring->push(cmd);
        }
    }

    if (total_filled > 0) {
        const int64_t avg_price = total_value / total_filled;
        send_fill(msg, total_filled, avg_price, remaining == 0, reactor_idx);
    }
}

void Matcher::send_fill(const ReactorOrderMessage& msg,
                        int64_t filled_qty,
                        int64_t avg_price_ticks,
                        bool    fully_filled,
                        int     reactor_idx) noexcept {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    FillResult fill{};
    fill.conn_token     = msg.conn_token;
    fill.fill_timestamp = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
                        + static_cast<uint64_t>(ts.tv_nsec);
    fill.filled_qty     = filled_qty;
    fill.avg_fill_price = avg_price_ticks;
    fill.order_id       = msg.order_id;
    fill.instrument     = msg.instrument;
    fill.fully_filled   = fully_filled ? 1 : 0;
    fill.side           = msg.side;

    cfg_.fill_queues[reactor_idx]->push(fill);
}
