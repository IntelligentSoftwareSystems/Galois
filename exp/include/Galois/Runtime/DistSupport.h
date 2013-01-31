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

//Objects with this tag have to be stored in a persistent cache
//Objects with this tag use the Persistent Directory
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_is_persistent)
template<typename T>
struct is_persistent : public has_tt_is_persistent<T> {};

template<typename T>
class gptr {
  uintptr_t ptr;
  uint32_t owner;

  T* resolve(typename std::enable_if<!is_persistent<T>::value>::type* = 0) const {
    T* rptr = nullptr;
    assert(ptr);
    if (owner == networkHostID)
      rptr = getSystemLocalDirectory().resolve<T>(ptr);
    else 
      rptr = getSystemRemoteDirectory().resolve<T>(ptr, owner);
    assert(rptr);
    if (inGaloisForEach)
      acquire(rptr, MethodFlag::ALL);
    return rptr;
  }

  // resolve for persistent objects!
/*
  T* resolve(typename std::enable_if<is_persistent<T>::value>::type* = 0) const {
    T* rptr = nullptr;
    assert(ptr);
    if (owner == networkHostID)
      rptr = reinterpret_cast<T>(ptr);
    else
      rptr = getSystemPersistentDirectory().resolve<T>(ptr, owner);
    assert(rptr);
    return rptr;
  }
*/

public:
  typedef T element_type;
  
  constexpr gptr() :ptr(0), owner(0) {}

  explicit gptr(T* p) :ptr(reinterpret_cast<uintptr_t>(p)), owner(networkHostID) {}

  T& operator*() const {
    return *resolve();
  }
  T *operator->() const {
    return resolve();
  }
  operator bool() const { return ptr != 0; }
  gptr& operator=(T* p) {
    ptr = reinterpret_cast<uintptr_t>(p);
    owner = networkHostID;
    return *this;
  }
  gptr& operator=(const gptr& p) {
    ptr = p.ptr;
    owner = p.owner;
    return *this;
  }

  //serialize
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    s.serialize(ptr);
    s.serialize(owner);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    s.deserialize(ptr);
    s.deserialize(owner);
  }

  void dump(std::ostream& os) {
    os << "[" << owner << ", " << ptr << "]";
  }
};

} //namespace Distributed
} //namespace Runtime
} //namespace Galois

#endif //DISTSUPPORT
