/** Galois Network Backend for MPI -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Hoang-Vu Dang <hdang8@illinois.edu>
 */

#ifdef GALOIS_USE_LWCI

#include "Galois/Runtime/NetworkIO.h"
#include "Galois/Runtime/Tracer.h"
#include "Galois/Substrate/SimpleLock.h"

//#include "hash/crc32.h"

#include "lc.h"
#include "lc/helper.h"

#include <iostream>

#include <cassert>
#include <cstring>
#include <mpi.h>
#include <deque>
#include <string>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <list>

#ifdef GALOIS_USE_VTUNE
#include "ittnotify.h"
#endif

class NetworkIOLWCI;
NetworkIOLWCI* __ctx;
lch* mv;

struct mpiMessage {
  lc_ctx ctx;
  uint32_t rank;
  uint32_t tag;
  std::vector<uint8_t> buf;

  mpiMessage() { }

  mpiMessage(uint32_t r, uint32_t t, std::vector<uint8_t> &b) :
    rank(r), tag(t), buf(std::move(b))  {};
};

class NetworkIOLWCI : public Galois::Runtime::NetworkIO {

  static void handleError(int rc) {
    if (rc != MPI_SUCCESS) {
      //GALOIS_ERROR(false, "MPI ERROR");
      MPI_Abort(MPI_COMM_WORLD, rc);
    }
  }

  static int getID() {
    int taskRank;
    handleError(MPI_Comm_rank(MPI_COMM_WORLD, &taskRank));
    return taskRank;
  }

  static int getNum() {
    int numTasks;
    handleError(MPI_Comm_size(MPI_COMM_WORLD, &numTasks));
    return numTasks;
  }

  std::mutex m;

  std::pair<int, int> initMPI() {
    m.lock();
    lc_open((size_t) 1024 * 1024 * 1024, &mv);
    m.unlock();
    __ctx = this;
    return std::make_pair(getID(), getNum());
  }

  std::deque<mpiMessage> inflight;
  std::list<mpiMessage> recv;

public:
int save;

  NetworkIOLWCI(uint32_t& ID, uint32_t& NUM) {
    auto p = initMPI();
    ID = p.first;
    NUM = p.second;
    save = ID;
  }

  ~NetworkIOLWCI() {
    lc_close(mv);
  }

  void fence() {
    while (!inflight.empty())
      ;
  }

  void complete_send() {
    while (!inflight.empty()) {
      auto& f = inflight.front();
      if (lc_test(&f.ctx)) {
        inflight.pop_front();
      } else {
        break;
      }
    }
  }

  virtual void enqueue(message m) {
    inflight.emplace_back(m.host, m.tag, m.data);
    auto& f = inflight.back();
    while (!lc_send_queue(mv, f.buf.data(), f.buf.size(), m.host, m.tag, &f.ctx)) {
      progress();
    }
  }

  void probe() {
    recv.emplace_back();
    auto& m = recv.back();
    int size;
    if (lc_recv_queue(mv, &size, (int*) &m.rank, (int*) &m.tag, &m.ctx)) {
      m.buf.resize(size);
      lc_recv_queue_post(mv, m.buf.data(), &m.ctx);
    } else {
      recv.pop_back();
    }
  }

  virtual message dequeue() {
    if (!recv.empty()) {
      for (auto it = recv.begin(); it != recv.end(); it ++) {
        auto &m = *it;
        if (lc_test(&m.ctx)) {
          message msg{m.rank, m.tag, std::move(m.buf)};
          recv.erase(it);
          return std::move(msg);
        }
        lc_progress(mv);
      }
    }
    return message{~0U, 0, std::vector<uint8_t>()};
  }

  virtual void progress() {
    complete_send();
    probe();
    lc_progress(mv);
  }

};

void __fence__()
{
  __ctx->fence();
}

std::tuple<std::unique_ptr<Galois::Runtime::NetworkIO>,uint32_t,uint32_t> Galois::Runtime::makeNetworkIOLWCI() {
  uint32_t ID, NUM;
  std::unique_ptr<Galois::Runtime::NetworkIO> n{new NetworkIOLWCI(ID, NUM)};
  return std::make_tuple(std::move(n), ID, NUM);
}

void main_task(intptr_t) {}
#endif
