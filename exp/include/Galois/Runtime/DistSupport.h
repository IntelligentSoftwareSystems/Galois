#include "Galois/Runtime/Context.h"

#include <vector>
#include <ostream>

#include <cstddef>
#include <cstring>
#include <cstdlib>

#include <boost/mpl/has_xxx.hpp>

namespace Galois {
namespace Runtime {
namespace Distributed {

template<typename T>
class gptr;

template<typename T>
class gptr {
  T* ptr;

public:
  typedef T element_type;
  
  constexpr gptr() :ptr(nullptr) {}
  constexpr gptr(std::nullptr_t) :ptr(nullptr) {}

  gptr( const gptr& r ) :ptr(r.ptr) {}
  explicit gptr(T* p) :ptr(p) {}

  ~gptr() {}

  gptr& operator=(const gptr& sp) {
    ptr = sp.ptr;
    return *this;
  }

  T& operator*() const {
    GaloisRuntime::acquire(ptr,Galois::MethodFlag::ALL);
    return *ptr;
  }
  T *operator->() const {
    GaloisRuntime::acquire(ptr,Galois::MethodFlag::ALL);
    return ptr;
  }
  operator bool() const { return ptr != nullptr; }

  typedef int tt_has_serialize;
  void serialize(std::ostream& os) const {
    os << ptr;
  }
};

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_has_serialize)
template<typename T>
struct has_serialize : public has_tt_has_serialize<T> {};

template<typename T>
void serialize(std::ostream& os, const T& data, typename std::enable_if<std::is_pod<T>::value>::type* = 0) {
  os.write(&data, sizeof(T));
}

template<typename T>
void serialize(std::ostream& os, const T& data, typename std::enable_if<has_serialize<T>::value>::type* = 0) {
  data.serialize(os);
}

// template<typename T>
// T* deSerialize(memBuffer& buf) {
//   if (std::is_pod<T>::value) {
//     T* n = new T;
//     buf.memcpyOut(n);
//     abort();
//   } else if (has_static_serialize<T>::value) {
//     return new T(buf);
//   } else {
//     abort();
//   }
//   return 0;
// }

} //namespace Distributed
} //namespace Runtime
} //namespace Galois
