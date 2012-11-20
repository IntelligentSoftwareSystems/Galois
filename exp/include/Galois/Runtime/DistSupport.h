#include "Galois/Runtime/Context.h"

#include <vector>

#include <cstddef>
#include <cstring>
#include <cstdlib>

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

template<typename BaseTy>
class DistBase {
};

class memBuffer {
  std::vector<unsigned char> data;
  
public:
  template<typename T>
  void memcpy(T* obj, size_t n) {
    size_t old = data.size();
    data.resize(old + n);
    std::memcpy(&obj[old], obj, n);
  }

};

template<typename T>
void Serialize(const T& data, memBuffer& buf) {
  if (std::is_pod<T>::value) {
    buf.memcpy(&data, sizeof(T));
  } else {
    abort();
  }
}


} //namespace Distributed
} //namespace Runtime
} //namespace Galois
