/** Galois Network Backend for MPI -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
#include "galois/runtime/NetworkIO.h"
#include "galois/runtime/Tracer.h"
#include "galois/substrate/SimpleLock.h"

#include "galois/runtime/LWCI.h"

#include <iostream>
#include <list>
#include <limits>

#ifdef GALOIS_USE_VTUNE
#include "ittnotify.h"
#endif

class NetworkIOLWCI;
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

void* alloc_cb(void* ctx, size_t size)
{
  mpiMessage* msg = (mpiMessage*) ctx;
  msg->buf.resize(size);
  return msg->buf.data();
}

#undef  LC_SYNC_SIGNAL
#define LC_SYNC_SIGNAL thread_signal

static void thread_signal(void* sync) {
  mpiMessage* msg = (mpiMessage*) sync;
  assert(msg && "Should not happen\n");
  delete msg;
}

/**
 * LWCI implementation of network IO.
 */
class NetworkIOLWCI : public galois::runtime::NetworkIO {
  static int getID() {
    return lc_id(mv);
  }

  static int getNum() {
    return lc_size(mv);
  }

  /**
   * Initializes LWCI's communication layer.
   */
  std::pair<int, int> initMPI() {
    lc_open(&mv, 1);
    lc_sync_init(NULL, NULL, thread_signal, NULL);
    return std::make_pair(getID(), getNum());
  }

  std::list<mpiMessage> recv;

public:
  NetworkIOLWCI(galois::runtime::MemUsageTracker& tracker, uint32_t& ID, uint32_t& NUM) 
    : NetworkIO(tracker) {
    auto p = initMPI();
    ID = p.first;
    NUM = p.second;
  }

  ~NetworkIOLWCI() {
    lc_close(mv);
  }

  virtual void enqueue(message m) {
    memUsageTracker.incrementMemUsage(m.data.size());
    mpiMessage *f = new mpiMessage(m.host, m.tag, m.data);
    f->ctx.type = LC_REQ_PEND;
    while (!lc_send_queue(mv, f->buf.data(), f->buf.size(), m.host, m.tag, 0, &f->ctx)) {
      progress();
    }
    if (lc_post(&f->ctx, (void*) f)) {
      assert(f->ctx.type == LC_REQ_DONE && "Something wrong\n");
      delete f;
    }
  }

  void probe() {
    recv.emplace_back();
    auto& m = recv.back();
    size_t size; lc_qtag tag;
    if (lc_recv_queue(mv, &size, (int*) &m.rank, (lc_qtag*) &tag, 0, alloc_cb, &m, &m.ctx)) {
      m.tag = tag;
      memUsageTracker.incrementMemUsage(size);
    } else {
      recv.pop_back();
    }
  }

  virtual message dequeue() {
    if (!recv.empty()) {
      uint32_t min_tag = std::numeric_limits<uint32_t>::max();
      for (auto it = recv.begin(); it != recv.end(); it ++) {
        auto &m = *it;
        if ((m.tag < min_tag) && (m.tag > 0)) {
          min_tag = m.tag;
        }
        if (m.tag > min_tag) { // do not return messages from the next phase
          break;
        }
        if (lc_test(&m.ctx)) {
          message msg{m.rank, m.tag, std::move(m.buf)};
          recv.erase(it);
          return std::move(msg);
        }
      }
    }
    return message{~0U, 0, std::vector<uint8_t>()};
  }

  virtual void progress() {
    probe();
    lc_progress(mv);
  }
};

/**
 * Creates the LWCI network IO for use in an network interface.
 *
 * @returns Tuple with a pointer to the LWCI network IO as well as the id of
 * the caller in the network layer + total number of hosts in the system.
 */
std::tuple<std::unique_ptr<galois::runtime::NetworkIO>,
                           uint32_t,
                           uint32_t> galois::runtime::makeNetworkIOLWCI(galois::runtime::MemUsageTracker& tracker) {
  uint32_t ID, NUM;
  std::unique_ptr<galois::runtime::NetworkIO> n{new NetworkIOLWCI(tracker, ID, NUM)};
  return std::make_tuple(std::move(n), ID, NUM);
}

void main_task(intptr_t) {}
#endif
