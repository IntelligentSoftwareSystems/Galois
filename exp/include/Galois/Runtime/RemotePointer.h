/** Galois Distributed Pointer -*- C++ -*-
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
#ifndef GALOIS_RUNTIME_REMOTEPOINTER_H
#define GALOIS_RUNTIME_REMOTEPOINTER_H

//#include "Galois/MethodFlags.h"
#include "Galois/Runtime/Directory.h"
#include "Galois/Runtime/CacheManager.h"
#include "Galois/Runtime/FatPointer.h"

namespace Galois {
namespace Runtime {

template<typename T>
class gptr {
  fatPointer ptr;

  T* inner_resolve(bool write) const;

  friend std::ostream& operator<<(std::ostream& os, const gptr<T>& v) {
    return os << v.ptr;
  }

public:
  typedef T element_type;
  
  constexpr gptr() noexcept :gptr(0, nullptr) {}
  explicit constexpr  gptr(T* p) noexcept :gptr(NetworkInterface::ID, p) {}
  gptr(uint32_t o, T* p) noexcept :ptr(o, static_cast<void*>(p)) {}
  
  //operator fatPointer() const { return ptr; }

  T& operator*()  { return *inner_resolve(true); }
  T* operator->() { return  inner_resolve(true); }
  const T& operator*()  const { return *inner_resolve(false); }
  const T* operator->() const { return  inner_resolve(false); }

  //  T* resolve(MethodFlag m) const { return  inner_resolve(m); }

  bool operator< (const gptr& rhs) const { return ptr < rhs.ptr;  }
  bool operator> (const gptr& rhs) const { return ptr > rhs.ptr;  }
  bool operator==(const gptr& rhs) const { return ptr == rhs.ptr; }
  bool operator!=(const gptr& rhs) const { return ptr != rhs.ptr; }
  bool operator<=(const gptr& rhs) const { return ptr <= rhs.ptr; }
  bool operator>=(const gptr& rhs) const { return ptr >= rhs.ptr; }

  explicit operator bool()         const { return bool(ptr); }
  explicit operator fatPointer()   const { return ptr; }

  bool isLocal() const { return ptr.isLocal(); }

  // bool sameHost(const gptr& rhs) const {
  //   return ptr.getHost() == rhs.ptr.getHost();
  // }

  // void initialize(T* p) {
  //   ptr.setObj(p);
  //   ptr.setHost(p ? NetworkInterface::ID : 0);
  // }

};

template<typename T>
T* gptr<T>::inner_resolve(bool write) const {
  T* retval = nullptr;
  if (isLocal())
    retval = static_cast<T*>(ptr.getObj());
  else
    retval = static_cast<T*>(getCacheManager().resolve(ptr, write));
  if (inGaloisForEach)
    return retval;

  if (isLocal()) {
    while(!getLocalDirectory().fetch(static_cast<Lockable*>(ptr.getObj()))) {}
    return retval;
  }
  
  do {
    getRemoteDirectory().fetch<T>(ptr, write ? ResolveFlag::RW : ResolveFlag::RO);
    retval = static_cast<T*>(getCacheManager().resolve(ptr, write));
  } while (!retval);
  return retval;
}


} //namespace Runtime
} //namespace Galois

#endif //DISTSUPPORT
