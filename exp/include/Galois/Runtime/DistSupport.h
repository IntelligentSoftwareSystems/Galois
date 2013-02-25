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

namespace {

template<typename T, bool> struct resolve_dispatch;

template<typename T>
struct resolve_dispatch<T, false> {
    static T* go(uint32_t owner, T* ptr) {
    T* rptr = nullptr;
    assert(ptr);
    if (owner == networkHostID) {
      // have to enter the directory when outside the for each to
      // check for remote objects! can't be found otherwise as
      // acquire isn't called outside the for each.
      if (inGaloisForEach) {
        rptr = ptr;
        try {
          // acquire the lock if inside the for each
          acquire (rptr, Galois::MethodFlag::ALL);
        }
        catch (const conflict_ex& ex) {
          rptr = getSystemLocalDirectory().resolve<T>((uintptr_t)ptr, getThreadContext());
        }
      }
      else {
        rptr = getSystemLocalDirectory().resolve<T>((uintptr_t)ptr, getThreadContext());
      }
    } else
      rptr = getSystemRemoteDirectory().resolve<T>((uintptr_t)ptr, owner, getThreadContext());
    assert(rptr);
    return rptr;
  }
};

// resolve for persistent objects!
template<typename T>
struct resolve_dispatch<T,true> {
  static T* go(uint32_t owner, T*  ptr) {
    T* rptr = nullptr;
    assert(ptr);
    if (owner == networkHostID)
      rptr = ptr;
    else
      rptr = getSystemPersistentDirectory().resolve<T>((uintptr_t)ptr, owner);
    assert(rptr);
    return rptr;
  }
};
}

template<typename T>
T* resolve(const gptr<T>& p) {
  return resolve_dispatch<T,is_persistent<T>::value>::go(p.owner, p.ptr);
}

template<typename T>
class gptr {
  T* ptr;
  uint32_t owner;

  friend T* resolve<>(const gptr<T>&);

public:
  typedef T element_type;
  
  constexpr gptr() :ptr(0), owner(0) {}

  explicit gptr(T* p) :ptr(p), owner(networkHostID) {}

  // calling resolve acquires the lock, used after a prefetch
  // IMP: have to be changed when local objects aren't passed to the directory
  void acquire() const {
    (void) *resolve(*this);
  }

  // check if the object is available, else just make a call to fetch
  T* transientAcquire() {
    if (owner == networkHostID)
      return getSystemLocalDirectory().transientAcquire<T>(reinterpret_cast<uintptr_t>(ptr));
    else
      return getSystemRemoteDirectory().transientAcquire<T>(reinterpret_cast<uintptr_t>(ptr), owner);
  }

  // check if the object is available, else just make a call to fetch
  void transientRelease() {
    if (owner == networkHostID)
      getSystemLocalDirectory().transientRelease(reinterpret_cast<uintptr_t>(ptr));
    else
      getSystemRemoteDirectory().transientRelease(reinterpret_cast<uintptr_t>(ptr), owner);
  }

  // check if the object is available, else just make a call to fetch
  void prefetch() {
    if (owner == networkHostID)
      getSystemLocalDirectory().prefetch<T>(reinterpret_cast<uintptr_t>(ptr));
    else
      getSystemRemoteDirectory().prefetch<T>(reinterpret_cast<uintptr_t>(ptr), owner);
  }

  T& operator*() const {
    return *resolve(*this);
  }
  T* operator->() const {
    return resolve(*this);
  }

  bool operator<(const gptr& rhs) const {
    if (owner == rhs.owner)
      return ptr < rhs.ptr;
    return owner < rhs.owner;
  }
  bool operator>(const gptr& rhs) const {
    if (owner == rhs.owner)
      return ptr > rhs.ptr;
    return owner > rhs.owner;
  }
  bool operator==(const gptr& rhs) const {
    return rhs.ptr == ptr && rhs.owner == owner;
  }
  bool operator!=(const gptr& rhs) const {
    return rhs.ptr != ptr || rhs.owner != owner;
  }
  explicit operator bool() const { return ptr != 0; }

  bool isLocal() const {
    return owner == Galois::Runtime::Distributed::networkHostID;
  }

  bool sameHost(const gptr& rhs) const {
    return owner == rhs.owner;
  }

  void initialize(T* p) {
    ptr = p;
    owner = ptr ? networkHostID : 0;
  }

  //serialize
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,ptr, owner);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,ptr, owner);
  }

  void dump() const {
    printf("[%u,%lx]", owner, (size_t)ptr);
  }
};

template<typename T>
remote_ex make_remote_ex(const Distributed::gptr<T>& p) {
  return remote_ex{&Distributed::LocalDirectory::localReqLandingPad<T>, p.ptr, p.owner};
}
template<typename T>
remote_ex make_remote_ex(uintptr_t ptr, uint32_t owner) {
  return remote_ex{&Distributed::LocalDirectory::localReqLandingPad<T>, ptr, owner};
}

} //namespace Distributed
} //namespace Runtime
} //namespace Galois

#endif //DISTSUPPORT
