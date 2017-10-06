#ifndef _ARRAY_WRAPPER_H_
#define _ARRAY_WRAPPER_H_

#include <array>
#include <type_traits>
#include "Galois/Runtime/Extra_dist_traits.h"

namespace galois {

  template<class T, size_t N>
    class CopyableArray : public std::array<T, N>
  {
    public:

      // only typedef if T is trivially copyable to use mem copy in serialize/deserialize.
      typedef typename std::enable_if<Galois::Runtime::is_memory_copyable<T>::value, int>::type tt_is_copyable;
  };
}
#endif
