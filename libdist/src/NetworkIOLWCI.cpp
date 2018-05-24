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
  lc_mem_fence();
  mpiMessage* msg = (mpiMessage*) ctx;
  msg->buf.resize(size);
  return msg->buf.data();
}

void thread_signal(void* signal) 
{
  mpiMessage* msg = (mpiMessage*) signal;
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
    lc_sync_init(NULL, thread_signal);
    return std::make_pair(getID(), getNum());
  }

  std::list<mpiMessage> recv;
  int save;

public:
  NetworkIOLWCI(galois::runtime::MemUsageTracker& tracker, uint32_t& ID, uint32_t& NUM) 
    : NetworkIO(tracker) {
    auto p = initMPI();
    ID = p.first;
    NUM = p.second;
    save = ID;
  }

  ~NetworkIOLWCI() {
    lc_close(mv);
  }

  virtual void enqueue(message m) {
    memUsageTracker.incrementMemUsage(m.data.size());
    mpiMessage* f = new mpiMessage(m.host, m.tag, m.data);
    lc_info info = {LC_SYNC_WAKE, LC_SYNC_NULL, {0, (int16_t) m.tag}};
    while (!lc_send_queue(mv, f->buf.data(), f->buf.size(), m.host, &info, &f->ctx)) {
      progress();
    }
    if (lc_post(&f->ctx, (void*) f)) {
      delete f;
    }
  }

  void probe() {
    recv.emplace_back();
    auto& m = recv.back();
    size_t size; lc_qtag tag;
    if (lc_recv_queue(mv, &size, (int*) &m.rank, (lc_qtag*) &tag, 0, alloc_cb, &m, LC_SYNC_NULL, &m.ctx)) {
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
        lc_progress(mv);
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
