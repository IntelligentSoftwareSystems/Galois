#include "galois/substrate/Barrier.h"
#include "galois/substrate/ThreadPool.h"

#include <mutex>
#include <condition_variable>

namespace {

class OneWayBarrier: public galois::substrate::Barrier {
  std::mutex lock;
  std::condition_variable cond;
  unsigned count; 
  unsigned total;

public:
  OneWayBarrier(unsigned p) {
    reinit(p);
  }
  
  virtual ~OneWayBarrier() {
  }

  virtual void reinit(unsigned val) {
    count = 0;
    total = val;
  }

  virtual void wait() {
    std::unique_lock<std::mutex> tmp(lock);
    count += 1;
    cond.wait(tmp, [this] () { return count >= total; });
    cond.notify_all();
  }

  virtual const char* name() const { return "OneWayBarrier"; }
};

class SimpleBarrier: public galois::substrate::Barrier {
  OneWayBarrier barrier1;
  OneWayBarrier barrier2;
  unsigned total;
public:
  SimpleBarrier(unsigned p): barrier1(p), barrier2(p), total(p) { }

  virtual ~SimpleBarrier() { }

  virtual void reinit(unsigned val) {
    total = val;
    barrier1.reinit(val);
    barrier2.reinit(val);
  }

  virtual void wait() {
    barrier1.wait();
    if (galois::substrate::ThreadPool::getTID() == 0)
      barrier1.reinit(total);
    barrier2.wait();
    if (galois::substrate::ThreadPool::getTID() == 0)
      barrier2.reinit(total);
  }

  virtual const char* name() const { return "SimpleBarrier"; }

};

} // end anonymous namespace

std::unique_ptr<galois::substrate::Barrier> galois::substrate::createSimpleBarrier(unsigned int v) {
  return std::unique_ptr<Barrier>(new SimpleBarrier(v));
}

