#include "Reactor.h"

#include <cstring>
#include <ctime>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <sys/prctl.h>

// ── Connection ───────────────────────────────────────────────────────────────

Connection::Connection() : generation(0), fd(-1), recv_bytes(0) {}

void Connection::reset(int f) {
  fd = f;
  recv_bytes = 0;
  ++generation;
  // recv_buf does not need zeroing: recv_bytes tracks valid bytes
}

// ── Reactor ──────────────────────────────────────────────────────────────────

Reactor::Reactor(int worker_id, int port) : reactor_id_(static_cast<uint32_t>(worker_id)) {
  PORT_ = port;
  connection_pool_ = std::make_unique<std::array<Connection, MAX_FDS>>();

  char name[16];
  snprintf(name, sizeof(name), "Reactor-%d", worker_id);
  prctl(PR_SET_NAME, name, 0, 0, 0);

  listener_ = get_listener();
  (*connection_pool_)[listener_].fd = listener_;
  init_epoll();
}

void Reactor::wire(const WireConfig &cfg) noexcept {
  snapshots_ = cfg.snapshots;
  fast_current_buf_ = cfg.fast_current_buf;
  slow_current_buf_ = cfg.slow_current_buf;
  fast_batch_ = cfg.fast_batch;
  slow_batch_ = cfg.slow_batch;
  fill_queue_ = cfg.fill_queue;
}

void Reactor::pin_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (result != 0)
    perror("pthread_setaffinity_np");
  else
    std::cout << "Reactor-" << reactor_id_ << " pinned to core " << core_id << '\n';
}

void Reactor::run() {
  int nfds = -1;
  while (true) {
    nfds = epoll_wait(epoll_, events_, BACKLOG, 0);

    if (__builtin_expect(nfds > 0, 1)) {
      for (int i = 0; i < nfds; ++i) {
        Connection *conn = conn_from_event(events_[i]);

        if (conn->fd == listener_)
          handle_new_connection();
        else
          handle_client_data(conn);
      }
    } else if (__builtin_expect(nfds == 0, 0)) {
      _mm_pause();
    } else {
      if (errno == EINTR)
        continue;
      perror("epoll_wait");
      break;
    }
  }
}

// ── Hot path ─────────────────────────────────────────────────────────────────

// route_order: build ReactorOrderMessage, classify, and write to the correct
// destination (aggressive SPSC or passive GPUBatch sub-batch).
void Reactor::route_order(Connection *conn, const OrderMessage *om) noexcept {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);

  ReactorOrderMessage msg{};
  msg.side = om->side;
  msg.type = om->type;
  msg.order_id = om->order_id;
  msg.price_ticks = om->price_ticks;
  msg.quantity = om->quantity;
  msg.instrument = om->instrument;
  msg.conn_token = (static_cast<uint64_t>(conn->fd) << 32) | conn->generation;
  msg.arrive_ns = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<uint64_t>(ts.tv_nsec);
  msg.reactor_id = reactor_id_;

  // MARKET orders are always aggressive — bypass the spread check
  if (om->type == OrderType::MARKET) {
    aggressive_queue_.push(msg);
    return;
  }

  // Limit orders: classify by proximity to spread
  const SpreadSnapshot spread = read_spread(&snapshots_[om->instrument]);
  const Classification cls = classifier_.classify(msg, spread);

  if (cls == Classification::AGGRESSIVE) {
    aggressive_queue_.push(msg);
    return;
  }

  // Passive: write directly into the current GPUBatch (zero copy)
  GPUBatch &batch = (cls == Classification::PASSIVE_NEAR)
                        ? fast_batch_[fast_current_buf_->load(std::memory_order_acquire)]
                        : slow_batch_[slow_current_buf_->load(std::memory_order_acquire)];

  // CANCEL sends negative quantity so the GPU subtracts from the book level
  const int64_t qty =
      (om->type == OrderType::CANCEL) ? -static_cast<int64_t>(om->quantity) : static_cast<int64_t>(om->quantity);

  const bool ok = batch.try_write(om->instrument, price_to_idx(om->price_ticks, om->instrument), qty, om->side,
                                  msg.conn_token, reactor_id_, msg.arrive_ns);

  if (!ok) {
    // Sub-batch full — order dropped for this batch interval.
    // TODO Part 8 Task 3: proper backpressure — skip rearm, re-arm
    // this connection when dispatch fires the next buffer.
  }
}

// handle_client_data: drain the kernel TCP buffer in batches, process all
// complete OrderMessages, carry partial bytes across calls.
void Reactor::handle_client_data(Connection *conn) {
  // TODO Part 8 Task 4: drain fill_queue_ and send MatchResult bytes back
  // to the client connection identified by conn_token.

  while (true) {
    ssize_t n =
        recv(conn->fd, conn->recv_buf + conn->recv_bytes, sizeof(conn->recv_buf) - conn->recv_bytes, MSG_DONTWAIT);

    if (n > 0) {
      const int total = conn->recv_bytes + static_cast<int>(n);
      const int n_msgs = total / static_cast<int>(sizeof(OrderMessage));

      // TODO(velzard) CHANGE TO STD::BYTE INSTEAD OF UINT8_T ================================
      for (int i = 0; i < n_msgs; ++i) {
        route_order(conn, reinterpret_cast<const OrderMessage *>(conn->recv_buf + i * sizeof(OrderMessage)));
      }

      // Preserve any partial trailing bytes at the front of recv_buf
      const int leftover = total % static_cast<int>(sizeof(OrderMessage));
      if (leftover && n_msgs)
        std::memmove(conn->recv_buf, conn->recv_buf + n_msgs * sizeof(OrderMessage), leftover);
      conn->recv_bytes = static_cast<uint8_t>(leftover);

    } else if (n == 0) {
      goto close_conn;
    } else {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        break;
      goto close_conn;
    }
  }

  rearm(conn, EPOLLIN);
  return;

close_conn:
  epoll_ctl(epoll_, EPOLL_CTL_DEL, conn->fd, nullptr);
  close(conn->fd);
  conn->fd = -1;
}

// ── Cold paths ────────────────────────────────────────────────────────────────

void Reactor::rearm(Connection *conn, uint32_t extra_flags) {
  struct epoll_event ev;
  ev.events = EPOLLET | EPOLLONESHOT | extra_flags;
  ev.data.ptr = conn;
  epoll_ctl(epoll_, EPOLL_CTL_MOD, conn->fd, &ev);
}

void Reactor::handle_new_connection() {
  struct sockaddr_storage remoteaddr;
  socklen_t addrlen = sizeof(remoteaddr);

  while (true) {
    int newfd =
        accept4(listener_, reinterpret_cast<struct sockaddr *>(&remoteaddr), &addrlen, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (newfd == -1) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        break;
      perror("accept4");
      break;
    }

    const int enable = 1;
    const int micro_seconds = 50;
    const int cpu = sched_getcpu();
    setsockopt(newfd, IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(enable));
    setsockopt(newfd, SOL_SOCKET, SO_BUSY_POLL, &micro_seconds, sizeof(int));
    setsockopt(newfd, IPPROTO_TCP, TCP_QUICKACK, &enable, sizeof(enable));
    setsockopt(newfd, SOL_SOCKET, SO_INCOMING_CPU, &cpu, sizeof(cpu));

    Connection *conn = &(*connection_pool_)[newfd];
    conn->reset(newfd);

    struct epoll_event ev;
    ev.events = EPOLLIN | EPOLLET | EPOLLONESHOT;
    ev.data.ptr = conn;

    if (epoll_ctl(epoll_, EPOLL_CTL_ADD, newfd, &ev) == -1) {
      perror("epoll_ctl: newfd");
      close(newfd);
    }
  }
}

void *Reactor::get_in_addr(struct sockaddr *sa) {
  if (sa->sa_family == AF_INET)
    return &reinterpret_cast<struct sockaddr_in *>(sa)->sin_addr;
  return &reinterpret_cast<struct sockaddr_in6 *>(sa)->sin6_addr;
}

int Reactor::get_listener() {
  struct addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  char port_str[6];
  snprintf(port_str, sizeof(port_str), "%d", PORT_);

  struct addrinfo *servinfo = nullptr;
  if (int rv = getaddrinfo(nullptr, port_str, &hints, &servinfo); rv != 0) {
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
    exit(1);
  }

  int listen_fd = -1;
  for (struct addrinfo *p = servinfo; p; p = p->ai_next) {
    listen_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (listen_fd == -1) {
      perror("socket");
      continue;
    }

    const int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));

    const int micro_seconds = 50;
    setsockopt(listen_fd, SOL_SOCKET, SO_BUSY_POLL, &micro_seconds, sizeof(int));

    const int cpu = sched_getcpu();
    setsockopt(listen_fd, SOL_SOCKET, SO_INCOMING_CPU, &cpu, sizeof(cpu));

    if (bind(listen_fd, p->ai_addr, p->ai_addrlen) == -1) {
      close(listen_fd);
      perror("bind");
      continue;
    }
    break;
  }
  freeaddrinfo(servinfo);

  if (listen_fd == -1) {
    fprintf(stderr, "server: failed to bind\n");
    exit(1);
  }

  if (listen(listen_fd, BACKLOG) == -1) {
    perror("listen");
    exit(1);
  }

  const int flags = fcntl(listen_fd, F_GETFL, 0);
  fcntl(listen_fd, F_SETFL, flags | O_NONBLOCK);

  return listen_fd;
}

void Reactor::init_epoll() {
  epoll_ = epoll_create1(EPOLL_CLOEXEC);
  if (epoll_ == -1) {
    perror("epoll_create1");
    exit(EXIT_FAILURE);
  }

  struct epoll_event ev;
  ev.events = EPOLLIN | EPOLLET;
  ev.data.ptr = &(*connection_pool_)[listener_];

  if (epoll_ctl(epoll_, EPOLL_CTL_ADD, listener_, &ev) == -1) {
    perror("epoll_ctl: listener");
    exit(EXIT_FAILURE);
  }
}

Connection *Reactor::conn_from_event(struct epoll_event &ev) { return static_cast<Connection *>(ev.data.ptr); }
