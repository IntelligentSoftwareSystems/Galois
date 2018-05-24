#include "galois/substrate/ThreadPool.h"
#include "galois/substrate/Barrier.h"
#include "galois/substrate/CompilerSpecific.h"

namespace {

class CountingBarrier: public galois::substrate::Barrier {
  std::atomic<unsigned> count;
  std::atomic<bool> sense;
  unsigned num;
  std::vector<galois::substrate::CacheLineStorage<bool> > local_sense;

  void _reinit(unsigned val) {
    count = num = val;
    sense = false;
    local_sense.resize(val);
    for (unsigned i = 0; i < val; ++i)
      local_sense.at(i).get() = false;
  }

public:
  CountingBarrier(unsigned int activeT) {
    _reinit(activeT);
  }

  virtual ~CountingBarrier() {}

  virtual void reinit(unsigned val) { _reinit(val); }

  virtual void wait() {
    bool& lsense = local_sense.at(galois::substrate::ThreadPool::getTID()).get();
    lsense = !lsense;
    if (--count == 0) {
      count = num;
      sense = lsense;
    } else {
      while (sense != lsense) { galois::substrate::asmPause(); }
    }
  }

  virtual const char* name() const { return "CountingBarrier"; }
};

}

std::unique_ptr<galois::substrate::Barrier> galois::substrate::createCountingBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(new CountingBarrier(activeThreads));
}

