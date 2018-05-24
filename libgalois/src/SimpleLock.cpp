#include "galois/substrate/SimpleLock.h"

void galois::substrate::SimpleLock::slow_lock() const {
  int oldval = 0;
  do {
    while (_lock.load(std::memory_order_acquire) != 0) {
      asmPause();
    }
    oldval = 0;
  } while (!_lock.compare_exchange_weak(oldval, 1, std::memory_order_acq_rel, std::memory_order_relaxed));
  assert(is_locked());
}
