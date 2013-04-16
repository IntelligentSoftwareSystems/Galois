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

class PerBackend_v2;
class PerBackend_v3;

namespace Distributed {

//Objects with this tag have to be stored in a persistent cache
//Objects with this tag use the Persistent Directory
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_is_persistent)
template<typename T>
struct is_persistent : public has_tt_is_persistent<T> {};

SimpleRuntimeContext& getTransCnx();

namespace {

template<typename T, bool> struct resolve_dispatch;

//Normal objects resolve here
template<typename T>
struct resolve_dispatch<T, false> {
  static T* go(uint32_t owner, T* ptr) {
    if (owner == networkHostID) {
      // have to enter the directory when outside the for each to
      // check for remote objects! can't be found otherwise as
      // acquire isn't called outside the for each.
      if (inGaloisForEach)
	acquire(ptr, Galois::MethodFlag::ALL);
      else if (isAcquired(ptr))
	getSystemLocalDirectory().recall(ptr, true);
      return ptr;
    } else {
      T* rptr = getSystemRemoteDirectory().resolve<T>(owner,ptr);
      if (inGaloisForEach) {
	try {
	  acquire(rptr, Galois::MethodFlag::ALL);
	} catch (...) {
	  //	  std::cerr << "HHHHHAAHAHAHA\n";
	  throw (int)5;
	  abort();
	}
      } else {
	while (isAcquired(rptr)) {
	  if (!LL::getTID())
	    getSystemNetworkInterface().handleReceives();
	}
      }
      return rptr;
    }
  }
};

// resolve for persistent objects!
template<typename T>
struct resolve_dispatch<T,true> {
  static T* go(uint32_t owner, T*  ptr) {
    T* rptr = nullptr;
    if (owner == networkHostID)
      rptr = ptr;
    else
      rptr = getSystemPersistentDirectory().resolve<T>(ptr, owner);
    assert(rptr);
    return rptr;
  }
};
}

template<typename T>
T* resolve(const gptr<T>& p) {
  if (!p.ptr) {
  //   //    std::cerr << "aborting in resolve for " << typeid(T).name() << "\n";
    assert(p.ptr);
  }
  return resolve_dispatch<T,is_persistent<T>::value>::go(p.owner, p.ptr);
}

template <typename T>
T* transientAcquire(const gptr<T>& p) {
  if (p.owner == networkHostID) {
    while (!getTransCnx().try_acquire(p.ptr)) {
      getSystemLocalDirectory().recall(p.ptr, false);
      if (!LL::getTID())
	getSystemNetworkInterface().handleReceives();
    }
    return p.ptr;
  } else { // REMOTE
    do {
      T* rptr = getSystemRemoteDirectory().resolve<T>(p.owner, p.ptr);
      //DATA RACE with delete
      if (getTransCnx().try_acquire(rptr))
	return rptr;
      if (!LL::getTID())
	getSystemNetworkInterface().handleReceives();
    } while (true);
  }
}

template<typename T>
void transientRelease(const gptr<T>& p) {
  T* ptr = p.ptr;
  if (p.owner != networkHostID)
    ptr = getSystemRemoteDirectory().resolve<T>(p.owner, p.ptr);
  getTransCnx().release(ptr);
}

template <typename T>
T* getSharedObj(const gptr<T>& p) {
  T* ptr = p.ptr;
  if (p.owner != networkHostID) {
    ptr = getSystemRemoteDirectory().sresolve<T>(p.owner, p.ptr);
    // locked by Remote Directory if not local
    // abort the iteration
    if (isAcquired(ptr))
      throw conflict_ex{ptr};
  }
  // should never lock an object that is sharable
//assert(!isAcquired(ptr));
  return ptr;
}

void clearSharedCache();

template<typename T>
class gptr {
  T* ptr;
  uint32_t owner;

  friend T* resolve<>(const gptr<T>&);
  friend T* transientAcquire<>(const gptr<T>& p);
  friend void transientRelease<>(const gptr<T>& p);
  friend T* getSharedObj<>(const gptr<T>& p);
  friend PerBackend_v2;
  friend PerBackend_v3;

  gptr(uint32_t o, T* p) :ptr(p), owner(o) {}

public:
  typedef T element_type;
  
  constexpr gptr() noexcept :ptr(0), owner(0) {}
  explicit gptr(T* p) noexcept :ptr(p), owner(networkHostID) {}
  
  // calling resolve acquires the lock, used after a prefetch
  // IMP: have to be changed when local objects aren't passed to the directory
  // void acquire() const {
  //   (void) *resolve(*this);
  // }

  // // check if the object is available, else just make a call to fetch
  void prefetch() {
    if (0) {
      if (owner == networkHostID)
	getSystemLocalDirectory().recall(ptr, false); 
      else
	getSystemRemoteDirectory().resolve<T>(owner, ptr);
    }
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

  void dump(std::ostream& os) const {
    os << "[" << owner << "," << ptr << "]";
  }
};

template<typename T>
remote_ex make_remote_ex(const Distributed::gptr<T>& p) {
  return remote_ex{&Distributed::LocalDirectory::reqLandingPad<T>, p.ptr, p.owner};
}
template<typename T>
remote_ex make_remote_ex(Lockable* ptr, uint32_t owner) {
  return remote_ex{&Distributed::LocalDirectory::reqLandingPad<T>, ptr, owner};
}

} //namespace Distributed
} //namespace Runtime
} //namespace Galois

#endif //DISTSUPPORT
