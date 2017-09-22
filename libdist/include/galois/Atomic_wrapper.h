#ifndef _ATOMIC_WRAPPER_H_
#define _ATOMIC_WRAPPER_H_

#include <atomic>
#include <type_traits>
#include "Galois/Runtime/Extra_dist_traits.h"

namespace galois {

  template<class T>
    class CopyableAtomic : public std::atomic<T>
  {
    public:

      CopyableAtomic() : 
        std::atomic<T>(T{})
        {}

      constexpr CopyableAtomic(T desired) : 
        std::atomic<T>(desired) 
    {}

      constexpr CopyableAtomic(const CopyableAtomic<T>& other) :
        CopyableAtomic(other.load(std::memory_order_relaxed))
    {}

      CopyableAtomic& operator=(const CopyableAtomic<T>& other) {
        this->store(other.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
      }

      // only typedef if T is trivially copyable to use mem copy in serialize/deserialize.
      typedef typename std::enable_if<std::is_trivially_copyable<T>::value, int>::type tt_is_copyable;
  };
}
#endif
