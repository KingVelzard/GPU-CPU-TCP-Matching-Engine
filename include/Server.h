#ifndef SERVER_H
#define SERVER_H

#include "JoinedThread.h"
#include "Reactor.h"

class Server {
private:
  static constexpr int reactor_cores[] = {1, 3, 5, 7, 9, 11, 13, 15};
  std::vector<joined_thread> worker_threads;

public:
  void run();
};

#endif
