#include "Reactor.h"

Connection::Connection() : fd(-1), write_offset(0) {}

Connection::Connection(int f, size_t wo) : fd(f), write_offset(wo) {}

void Connection::reset(int f) {
    fd = f;
    write_offset = 0;
}


Reactor::Reactor(int worker_id, int port) {
    
    this->PORT = port;
    this->connection_pool = std::make_unique<std::array<Connection, MAX_FDS>>();

    char name[16];
    snprintf(name, sizeof(name), "Reactor-%d", worker_id);
    prctl(PR_SET_NAME, name, 0,0,0);

    this->listener = get_listener();
    (*connection_pool)[this->listener].fd = this->listener;
    init_epoll();

}

void Reactor::pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();

    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

    if (result != 0) {
        perror("pthread_setaffinity_np");
    } else {
        std::cout << "Thread pinned to Core: " << core_id << std::endl;
    }
}

void Reactor::run() 
{
    int nfds = -1;
    while (true) 
    { // main accept loop
        nfds = epoll_wait(this->epoll, this->events, BACKLOG, 0);

        if (__builtin_expect(nfds > 0, 1)) {
            for (int nfd_i{0}; nfd_i < nfds; ++nfd_i) 
            {
                Connection* conn = conn_from_event(events[nfd_i]);

                if (conn->fd == listener) {
                    // if its the listener, new connection
                    handle_new_connection();
                } else {
                    // otherwise it's a regular client request
                    handle_client_data(conn);
                }
            }
        }

        else if (__builtin_expect(nfds == 0, 0)) {
            _mm_pause(); // syscall to pause instruction without giving up to CPU
        }
        else {
            if (errno == EINTR) continue;
            perror("epoll_wait");
            break;
        }

    }
}

__attribute__((optimize("O3")))
void Reactor::handle_client_data(Connection* conn)
{

  // -------------------------------------------------------------------------
    // TODO (VALIDATION): This implementation has three known issues to fix:
    //
    // 1. PARTIAL READS: recv() may return fewer than sizeof(OrderMessage) bytes.
    //    Need to accumulate bytes in Connection::recv_buf until a full message
    //    arrives before processing. Currently a partial read produces a corrupt
    //    OrderMessage that gets passed to the GPU.
    //
    // 2. MULTIPLE MESSAGES: recv() may return more than sizeof(OrderMessage)
    //    bytes if the client sent multiple orders in one TCP segment. Currently
    //    only the first message is processed and the rest are silently discarded.
    //
    // 3. NO VALIDATION: The bytes are blindly cast to OrderMessage* with no
    //    checks on side, type, instrument index, price bounds, or qty.
    //    A malformed or malicious message can cause an out-of-bounds write
    //    in GPU global memory via an unchecked instrument index or price_idx.
    //    Add a validate(OrderMessage*) call before pushing to the queue.
    //    See validate() stub in Classifier.h.
    // -------------------------------------------------------------------------

    char buf[RECV_BUF_SIZE];	// buffer for client data
    int sender_fd = conn->fd;

    while (true) 
    {
        ssize_t nbytes = recv(sender_fd, buf, sizeof(buf), MSG_DONTWAIT);
        
        if (nbytes > 0) { // got data
            
            size_t total_sent = conn->write_offset;

            while (total_sent < RESPONSE_LEN) {
                ssize_t send_bytes = send(
                    sender_fd, 
                    RESPONSE + total_sent, 
                    RESPONSE_LEN - total_sent, 
                    0);
            
                if (__builtin_expect(send_bytes == -1,0)) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        conn->write_offset = total_sent;
                        this->rearm(conn, EPOLLIN | EPOLLOUT);
                        return;
                    } 
                    goto error_cleanup;
                }

                total_sent += send_bytes;

            }

            conn->write_offset = 0; // reset after complete send
            this->rearm(conn, EPOLLIN);
                
        } else if (nbytes == 0) { // Connection closed
            //printf("pollserver: socket %d hung up \n", sender_fd);
            goto error_cleanup;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // no more data right now
                break;
            }
            goto error_cleanup;
        }
    }
    return;

error_cleanup:
        epoll_ctl(this->epoll, EPOLL_CTL_DEL, sender_fd, nullptr);
        close(sender_fd);
        (*connection_pool)[sender_fd].fd = -1;
}
    
inline void Reactor::rearm(Connection* conn, uint32_t extra_flags)
{
    struct epoll_event ev;
    ev.events = EPOLLET | EPOLLONESHOT | extra_flags;
    ev.data.ptr = conn;
    epoll_ctl(this->epoll, EPOLL_CTL_MOD, conn->fd, &ev);
}


void Reactor::handle_new_connection()
{
    struct sockaddr_storage remoteaddr; // Client address
    socklen_t addrlen = sizeof remoteaddr;
    int newfd; // newly accepted sockfd
    char remoteIP[INET6_ADDRSTRLEN];

    struct epoll_event event_settings;
    // set both so we don't have to modify 
    event_settings.events = EPOLLIN | EPOLLET | EPOLLONESHOT;
    event_settings.data.ptr = conn;

    while (true) {
        newfd = accept4(this->listener, (struct sockaddr*)&remoteaddr,
                &addrlen, SOCK_NONBLOCK | SOCK_CLOEXEC);


        if (newfd == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) { 
                break; 
            }   
            perror("accept");
            break;
        } 

        // performance tweaks
        
        int enable = 1;
        int cpu = sched_getcpu();
        int micro_seconds = 50; // Poll for 50us before sleeping
        setsockopt(newfd, IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(enable)); // no delay
        setsockopt(newfd, SOL_SOCKET, SO_BUSY_POLL, &micro_seconds, sizeof(int)); // make sure to keep
        setsockopt(newfd, IPPROTO_TCP, TCP_QUICKACK, &enable, sizeof(enable)); // quick ack

        Connection* conn = &(*connection_pool)[newfd];
        conn->reset(newfd);

        

        if (__builtin_expect(epoll_ctl(epoll, EPOLL_CTL_ADD, newfd, &event_settings) == -1, 0)) {
            perror("epoll_ct1: newfd");
            close(newfd); // pool slot remains for next user
            exit(EXIT_FAILURE);
        }

        //printf("pollserver: new connection from %s on socket %d\n",
                //inet_ntop(remoteaddr.ss_family, 
                    //get_in_addr((struct sockaddr*)&remoteaddr), 
                    //remoteIP, sizeof remoteIP),
                //newfd);
        
    }
}


void* Reactor::get_in_addr(struct sockaddr *sa)
{
    if (sa->sa_family == AF_INET) {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }

    return &(((struct sockaddr_in6*)sa)->sin6_addr);
}


int Reactor::get_listener()
{
    int listen_sockfd;
    struct addrinfo hints, *servinfo, *p;
    int yes=1;
    int rv;

    char port_str[6];
    snprintf(port_str, sizeof(port_str), "%d", this->PORT);

    // get binding socket address information

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE; // use my ip
        
    if ((rv = getaddrinfo(NULL, port_str, &hints, &servinfo)) != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
        exit(1);
    }
    
    // loop through all the results and bind to the first we can
    
    for (p = servinfo; p != NULL; p = p->ai_next) {
        if ((listen_sockfd = socket(p->ai_family, p->ai_socktype,
                        p->ai_protocol)) == -1) {
            perror("server : socket");
            continue;
        }

        if (setsockopt(listen_sockfd, SOL_SOCKET, SO_REUSEADDR, &yes,
                    sizeof(int)) == -1) {
            perror("setsockopt: SO_REUSEADDR");
            exit(1);
        }

        if (setsockopt(listen_sockfd, SOL_SOCKET, SO_REUSEPORT, &yes,
                    sizeof(int)) == -1) {
            perror("setsockopt: SO_REUSEPORT");
            exit(1);
        }
        

        // POSSIBLY DELETE SINCE CUDA OVERHEAD ABOVE, WASTING CPU RESOURCE 
        int micro_seconds = 50;
        if (setsockopt(listen_sockfd, SOL_SOCKET, SO_BUSY_POLL, &micro_seconds,
                    sizeof(int)) == -1) {
            perror("setsockopt: SO_BUSY_POLL");
            exit(1);
        }
        
        int cpu = sched_getcpu();
        if (setsockopt(listen_sockfd, SOL_SOCKET, SO_INCOMING_CPU, &cpu, sizeof(cpu)) == -1) {
            perror("setsockopt: SO_INCOMING_CPU");
            exit(1);
        }

        if (bind(listen_sockfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(listen_sockfd);
            perror("server: bind");
            continue;
        }

        break;
    }

    freeaddrinfo(servinfo); // all done with serv info
    
    if (p == NULL) {
        fprintf(stderr, "server: failed to bind\n");
        exit(1);
    }

    // listen

    if (listen(listen_sockfd, BACKLOG) == -1) {
        perror("listen");
        exit(1);
    }

    // Set flags

    int flags = fcntl(listen_sockfd, F_GETFL, 0);
    fcntl(listen_sockfd, F_SETFL, flags | O_NONBLOCK);


    return listen_sockfd;
}

inline void Reactor::init_epoll()
{
    this->epoll = epoll_create1(EPOLL_CLOEXEC);
    if (epoll == -1) {
        perror("epoll_create1");
        exit(EXIT_FAILURE);
    }

    struct epoll_event event_settings;
    event_settings.events = EPOLLIN | EPOLLET;
    event_settings.data.ptr = &(*connection_pool)[listener];
    if (epoll_ctl(this->epoll, EPOLL_CTL_ADD, this->listener, &event_settings) == -1) {
        perror("epoll_ctl: listen_sockfd");
        exit(EXIT_FAILURE);
    }
}

inline Connection* Reactor::conn_from_event(struct epoll_event& ev) {
    return static_cast<Connection*>(ev.data.ptr);
}
    





