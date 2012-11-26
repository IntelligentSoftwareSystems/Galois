#include "Galois/Runtime/Context.h"

#include <vector>

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
};

/**
 * Indicates the operator may request the parallel loop to be suspended and a
 * given function run in serial
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_static_serialize)
template<typename T>
struct has_static_serialize : public has_tt_static_serialize<T> {};


class memBuffer {
  std::vector<unsigned char> data;
  
public:
  template<typename T>
  void memcpy(T* obj) {
    size_t old = data.size();
    size_t n = sizeof(T);
    data.resize(old + n);
    std::memcpy(&obj[old], obj, n);
  }
};

template<typename T>
void Serialize(const T& data, memBuffer& buf) {
  if (std::is_pod<T>::value) {
    buf.memcpy(&data);
  } else if (has_static_serialize<T>::value) {
    data.serialize(buf);
  } else {
    abort();
  }
}

template<typename T>
T* deSerialize(memBuffer& buf) {
  if (std::is_pod<T>::value) {
    abort();
  } else if (has_static_serialize<T>::value) {
    return new T(buf);
  } else {
    abort();
  }
  return 0;
}

} //namespace Distributed
} //namespace Runtime
} //namespace Galois
