/** Galois Distributed Pointer and Object Types -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_DISTSUPPORT_H
#define GALOIS_RUNTIME_DISTSUPPORT_H

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Serialize.h"

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

  //serialize
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    s.serialize(ptr);
  }
  void deserialize(Galois::Runtime::Distributed::SerializeBuffer& s) {
    s.deserialize(ptr);
  }

  void dump(std::ostream& os) {
    os << ptr;
  }
};

} //namespace Distributed
} //namespace Runtime
} //namespace Galois

#endif //DISTSUPPORT
