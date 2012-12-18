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
#include "Galois/Runtime/Directory.h"

namespace Galois {
namespace Runtime {
namespace Distributed {

template<typename T>
class gptr {
  uintptr_t ptr;
  uint32_t owner;

public:
  typedef T element_type;
  
  constexpr gptr() :ptr(0), owner(0) {}
  constexpr gptr(std::nullptr_t) :ptr(0), owner(0) {}

  gptr( const gptr& r ) :ptr(r.ptr), owner(networkHostID) {}
  explicit gptr(T* p) :ptr(reinterpret_cast<uintptr_t>(p)), owner(networkHostID) {}

  ~gptr()=default;
  gptr& operator=(const gptr& sp) =default;

  T& operator*() const {
    T* rptr = resolve();
    Galois::Runtime::acquire(rptr,Galois::MethodFlag::ALL);
    return *rptr;
  }
  T *operator->() const {
    T* rptr = resolve();
    Galois::Runtime::acquire(rptr,Galois::MethodFlag::ALL);
    return rptr;
  }
  operator bool() const { return ptr != 0; }

  //Resolve
  T* resolve() const {
    if (owner == networkHostID)
      return reinterpret_cast<T*>(ptr);
    return getSystemRemoteDirectory().resolve<T>(ptr, owner);
  }

  //serialize
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    s.serialize(ptr);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
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
