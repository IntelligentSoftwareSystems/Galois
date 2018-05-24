#ifndef _ARRAY_WRAPPER_H_
#define _ARRAY_WRAPPER_H_

#include <array>
#include "galois/runtime/Extra_dist_traits.h"

namespace galois {
  template<class T, size_t N>
  class CopyableArray : public std::array<T, N> {
  public:
    // only typedef if T is trivially copyable to use mem copy in serialize/deserialize.
    using tt_is_copyable = 
      typename std::enable_if<galois::runtime::is_memory_copyable<T>::value, int>::type;
  };
}
#endif
