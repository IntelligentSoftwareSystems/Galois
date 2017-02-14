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

#include "Galois/Runtime/NetworkIO.h"
#include "Galois/Runtime/Tracer.h"
#include "Galois/Substrate/SimpleLock.h"

//#include "hash/crc32.h"

#include "mv.h"
#include "mv/helper.h"
#include "mv_allocator.h"

#include <iostream>

#include <cassert>
#include <cstring>
#include <mpi.h>
#include <deque>
#include <string>
#include <fstream>
#include <unistd.h>
#include <vector>

class NetworkIOMV;
NetworkIOMV* __ctx;
mvh* mv;

struct mpiMessage {
  mv_ctx ctx;
  uint32_t rank;
  uint32_t tag;
  mv_vector<uint8_t> buf;

  mpiMessage() {};

  mpiMessage(uint32_t r, uint32_t t, mv_vector<uint8_t> &&b) :
    rank(r), tag(t), buf(std::move(b)) {}
};

class NetworkIOMV : public Galois::Runtime::NetworkIO {

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

  std::pair<int, int> initMPI() {
    mv_open(NULL, NULL, (size_t) 1024 * 1024 * 1024, &mv);
    __ctx = this;
    return std::make_pair(getID(), getNum());
  }

  std::deque<mpiMessage> inflight;
  std::deque<mpiMessage> recv;

public:
int save;

  NetworkIOMV(uint32_t& ID, uint32_t& NUM) {
    auto p = initMPI();
    ID = p.first;
    NUM = p.second;
    save = ID;
  }

  ~NetworkIOMV() {
    mv_close(mv);
  }

  void complete_send() {
    while (!inflight.empty()) {
      auto& f = inflight.front();
      if (mv_test(&f.ctx)) {
        inflight.pop_front();
      } else if (mv_send_enqueue_post(mv, &f.ctx, 0)) {
        inflight.pop_front();
      } else {
        break;
      }
    }
  }

  virtual void enqueue(message m) {
    inflight.emplace_back(m.host, m.tag, std::move(m.data));
    auto& f = inflight.back();
    int count = 0;
    while (!mv_send_enqueue_init(mv, f.buf.data(), f.buf.size(), m.host, m.tag, &f.ctx)) {
      progress();
    }
  }

  virtual message dequeue() {
    while (!recv.empty()) {
      auto& m = recv.front();
      if (mv_test(&m.ctx)) {
        message msg{m.rank, m.tag, std::move(m.buf)};
        recv.pop_front();
        return std::move(msg);
      } else {
        break;
      }
    }
    recv.emplace_back();
    auto& m = recv.back();
    int size;
    if (mv_recv_dequeue_init(mv, &size, (int*) &m.rank, (int*) &m.tag, &m.ctx)) {
      m.buf.resize(size);
      if (mv_recv_dequeue_post(mv, m.buf.data(), &m.ctx)) {
        message msg{m.rank, m.tag, std::move(m.buf)};
        recv.pop_back();
        return std::move(msg);
      }
    } else {
      recv.pop_back();
    }
    return message{~0U, 0, mv_vector<uint8_t>()};
  }

  virtual void progress() {
    complete_send();
    mv_progress(mv);
  }

};

std::tuple<std::unique_ptr<Galois::Runtime::NetworkIO>,uint32_t,uint32_t> Galois::Runtime::makeNetworkIOMV() {
  uint32_t ID, NUM;
  std::unique_ptr<Galois::Runtime::NetworkIO> n{new NetworkIOMV(ID, NUM)};
  return std::make_tuple(std::move(n), ID, NUM);
}

void main_task(intptr_t) {}
