#ifndef GALOIS_SUBSTRATE_STATICINSTANCE_H
#define GALOIS_SUBSTRATE_STATICINSTANCE_H

#include "galois/substrate/CompilerSpecific.h"

namespace galois {
namespace substrate {

//This should be much simpler in c++03 mode, but be general for now
//This exists because ptrlock is not a pod, but this is.
template<typename T>
struct StaticInstance {
  volatile T* V;
  volatile int _lock;

  inline void lock() {
    int oldval;
    do {
      while (_lock != 0) {
        substrate::asmPause();
      }
      oldval = __sync_fetch_and_or(&_lock, 1);
    } while (oldval & 1);
  }

  inline void unlock() {
    compilerBarrier();
    _lock = 0;
  }

  T* get() {
    volatile T* val = V;
    if (val)
      return (T*)val;
    lock();
    val = V;
    if (!val)
      V = val = new T();
    unlock();
    return (T*)val;
  }
};

} // end namespace substrate
} // end namespace galois

#endif
