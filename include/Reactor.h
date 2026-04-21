#ifndef REACTOR_H
#define REACTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>
#include "ThreadPool.h"
#include <poll.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <unordered_map>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <pthread.h>
#include <sched.h>
#include <iostream>
#include <immintrin.h>
#include <sys/prctl.h>

#define MAX_FDS 65536
#define RECV_BUF_SIZE 65536
#define BACKLOG 8192

static const char RESPONSE[] =
    "HTTP/1.1 200 OK\r\n"
    "Content-Length: 13\r\n"
    "Connection: keep-alive\r\n"
    "\r\n"
    "Hello, world!";
static const size_t RESPONSE_LEN = sizeof(RESPONSE) - 1;


struct alignas(64) Connection {
    public:
        int fd;
        size_t write_offset;
       
        // TODO (VALIDATION): add partial message framing here.
        // TCP is a stream — recv() may return fewer than sizeof(OrderMessage) bytes.
        // Need: uint8_t recv_buf[sizeof(OrderMessage)] and uint32_t recv_bytes
        // to accumulate a complete message before processing.
        // See handle_client_data in Reactor.cpp for where this is consumed.

        Connection();
        Connection(int f, size_t wo);

        void reset(int f);
};


class Reactor {

    private:
        int listener;
        int epoll;
        struct epoll_event events[BACKLOG];
        int PORT;
        std::unique_ptr<std::array<Connection, MAX_FDS>> connection_pool;

    public:

        Reactor(int worker_id, int port);
        void pin_to_core(int core_id);
        void run();

    private:

        [[ gnu::hot, gnu::optimize("O3") ]] void handle_client_data(Connection* conn);
        [[ gnu::cold ]] void handle_new_connection();
        [[ gnu::cold ]] void rearm(Connection* conn, uint32_t extra_flags);
        [[ gnu::cold ]] void* get_in_addr(struct sockaddr *sa);
        [[ gnu::cold ]] int get_listener();
        [[ gnu::cold ]] inline void init_epoll();
        [[ gnu::cold ]] inline Connection* conn_from_event(struct epoll_event& ev);

};

#endif
