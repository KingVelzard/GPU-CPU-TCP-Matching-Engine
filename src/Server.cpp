/*
 * server.cpp -- a stream socket server demo
 */
#include "Server.h"

void Server::run() {
  for (int i{0}; i < N_REACTORS; ++i) {
    worker_threads.emplace_back([this, i]() {
      Reactor reactor(i, PORT);
      reactor.pin_to_core(this->reactor_cores[i]);
      reactor.run();
    });
  }
}
