#ifndef REACTOR_H
#define REACTOR_H

#include "../tools/utils.h"
#include "Classifier.h"
#include "GPUBatch.h"
#include "InstrumentSnapshot.h"
#include "FillResult.h"
#include "OrderMessage.h"
#include "SPSCQueue.h"
#include <arpa/inet.h>
#include <cstdint>
#include <ctime>
#include <fcntl.h>
#include <immintrin.h>
#include <memory>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_FDS 65536

// Connection: one entry in the connection pool, indexed by fd.
// alignas(64) so adjacent connections never share a cache line.
struct alignas(64) Connection {
  uint32_t generation; // lower 32 bits of conn_token; bumped on reset
  int fd;              // -1 = slot is free
  uint8_t recv_bytes;  // bytes of a partial OrderMessage buffered (0–31)
  uint8_t _pad[3];
  // recv_buf holds up to BATCH_PER_INSTRUMENT complete OrderMessages
  // plus at most 31 partial bytes from a split TCP segment.
  uint8_t recv_buf[BATCH_PER_INSTRUMENT * 32]; // 2048 bytes

  Connection();
  void reset(int f);
};

static_assert(sizeof(Connection) % 64 == 0, "Connection must be a whole number of cache lines");

class Reactor {
public:
  Reactor(int worker_id, int port);
  void pin_to_core(int core_id);
  void run();

  // wire: called once after GPUDispatch and Matcher are initialised.
  // Sets all shared pointers before any reactor thread starts.
  struct WireConfig {
    InstrumentSnapshot *snapshots;
    std::atomic<int> *fast_current_buf;      // &GPUDispatch::fast_current_buf_
    std::atomic<int> *slow_current_buf;      // &GPUDispatch::slow_current_buf_
    GPUBatch *fast_batch;                    // GPUDispatch::fast_batches()
    GPUBatch *slow_batch;                    // GPUDispatch::slow_batches()
    SPSCQueue<FillResult, 256> *fill_queue; // Matcher pushes, reactor pops
  };
  void wire(const WireConfig &cfg) noexcept;

  // aggressive_queue: the Matcher registers a pointer to this queue
  // during startup to drain aggressive orders.
  SPSCQueue<ReactorOrderMessage, 256> aggressive_queue_;

private:
  int listener_;
  int epoll_;
  struct epoll_event events_[BACKLOG];
  int PORT_;
  uint32_t reactor_id_;
  std::unique_ptr<std::array<Connection, MAX_FDS>> connection_pool_;

  // Shared state wired in by wire() — read-only after startup
  InstrumentSnapshot *snapshots_ = nullptr;
  std::atomic<int> *fast_current_buf_ = nullptr;
  std::atomic<int> *slow_current_buf_ = nullptr;
  GPUBatch *fast_batch_ = nullptr;
  GPUBatch *slow_batch_ = nullptr;
  SPSCQueue<FillResult, 256> *fill_queue_ = nullptr;

  Classifier classifier_;

  [[gnu::hot]] void handle_client_data(Connection *conn);
  [[gnu::cold]] void handle_new_connection();
  [[gnu::cold]] void rearm(Connection *conn, uint32_t extra_flags);
  [[gnu::cold]] void *get_in_addr(struct sockaddr *sa);
  [[gnu::cold]] int get_listener();
  [[gnu::cold]] void init_epoll();
  [[gnu::cold]] Connection *conn_from_event(struct epoll_event &ev);

  // route_order: build a ReactorOrderMessage and send it to the right path.
  // Called for every complete OrderMessage pulled from recv_buf.
  [[gnu::hot]] void route_order(Connection *conn, const OrderMessage *om) noexcept;
};

#endif
