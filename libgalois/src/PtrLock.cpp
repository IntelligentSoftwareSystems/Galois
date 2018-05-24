#include "galois/substrate/PtrLock.h"

void galois::substrate::internal::ptr_slow_lock(std::atomic<uintptr_t>& _l) {
  uintptr_t oldval;
  do {
    while ((_l.load(std::memory_order_acquire) & 1) != 0) {
      asmPause();
    }
    oldval = _l.fetch_or(1, std::memory_order_acq_rel);
  } while (oldval & 1);
  assert(_l);
}

