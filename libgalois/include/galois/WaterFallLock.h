#ifndef _wf_h_
#define _wf_h_

#include <cstdint>
#include <stdlib.h>
#include <galois/substrate/PerThreadStorage.h>
#include <galois/substrate/CacheLineStorage.h>

using namespace galois::substrate;
namespace galois {
/** This is an array that is cacheline padded
 * TODO(AdityaAtulTewari) Should be moved to substrate
 * @author AdityaAtulTewari
 */
template <typename T>
class CacheLinePaddedArr {
  galois::substrate::CacheLineStorage<T>* arr;
  uint64_t sz;

  void initialize_values(T def) {
    for (uint64_t i = 0; i < sz; i++) {
      arr[i] = def;
    }
  }

public:
  CacheLinePaddedArr(T def)
      : arr(new CacheLineStorage<T>[getThreadPool().getMaxThreads()]),
        sz(getThreadPool().getMaxThreads()) {
    initialize_values(def);
  }

  CacheLinePaddedArr(uint64_t sz, T def)
      : arr(new CacheLineStorage<T>[sz]), sz(sz) {
    initialize_values(def);
  }

  T* get(uint64_t i) { return &arr[i].data; }

  uint64_t size() { return sz; }

  template <typename n_type>
  T& operator[](n_type i) {
    return arr[i].data;
  }
};

/** This is a Barrier style lock used for fine grained release control
 * TODO(AdityaAtulTewari) Should be moved to a substrate
 * @author AdityaAtulTewari
 */
template <typename T>
class WaterFallLock {
  T wfc{0};

public:
  WaterFallLock() : wfc(0) {}

  void reset() {
    for (unsigned i = 0; i < wfc.size(); i++)
      *(wfc.get(i)) = 0;
  }

  const char* name() { return typeid(WaterFallLock<T>).name(); }

  template <char val>
  void wait(uint64_t num) {
    while (__atomic_load_1((char*)wfc.get(num), __ATOMIC_ACQUIRE) != val)
      ;
  }
  template <char val>
  void done(uint64_t num) {
    __atomic_store_1((char*)wfc.get(num), val, __ATOMIC_RELEASE);
  }
};
} // namespace galois
#endif
