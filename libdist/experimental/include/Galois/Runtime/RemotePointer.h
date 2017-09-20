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
#include "Galois/Runtime/Context.h"

namespace galois {
namespace Runtime {

template<typename T>
class gptr {
  fatPointer ptr;

  friend std::ostream& operator<<(std::ostream& os, const gptr<T>& v) {
    return os << v.ptr;
  }

  T* resolve_safe() const {
    if (inGaloisForEach) {
      //parallel code ensures acquire happens before use
    } else {
      //Serial code needs hand holding
      serial_acquire(*this);
    }
    return resolve();
  }

  explicit gptr(fatPointer p) :ptr(p) {}

public:
  typedef T element_type;
  
  constexpr gptr() noexcept :ptr(0, 0) {}
  explicit constexpr  gptr(T* p) noexcept :gptr(NetworkInterface::ID, p) {}
  gptr(uint32_t o, T* p) noexcept :ptr(o, reinterpret_cast<uintptr_t>(p)) {}
  
  T& operator*()  { return *resolve_safe(); }
  T* operator->() { return  resolve_safe(); }
  T& operator*()  const { return *resolve_safe(); }
  T* operator->() const { return  resolve_safe(); }

  bool operator< (const gptr& rhs) const { return ptr < rhs.ptr;  }
  bool operator> (const gptr& rhs) const { return ptr > rhs.ptr;  }
  bool operator==(const gptr& rhs) const { return ptr == rhs.ptr; }
  bool operator!=(const gptr& rhs) const { return ptr != rhs.ptr; }
  bool operator<=(const gptr& rhs) const { return ptr <= rhs.ptr; }
  bool operator>=(const gptr& rhs) const { return ptr >= rhs.ptr; }

  explicit operator bool()         const { return bool(ptr); }
  explicit operator fatPointer()   const { return ptr; }

  bool isLocal() const { return ptr.isLocal(); }

  //experimental
  gptr operator+(int n) {
    return gptr(ptr.arith(n*sizeof(T)));
  }

  T* resolve() const {
    void* obj = nullptr;
    if (ptr.isLocal()) {
      obj = ptr.getPtr<void>();
    } else {
      ResolveCache* tr = getThreadResolve();
      if (tr) // equivalent to in a parallel loop
        obj = tr->resolve(ptr);
      else
        obj = getCacheManager().resolve(ptr);
    }
    return obj ? static_cast<T*>(obj) : nullptr;
  }

  //Trivially_copyable
  typedef int tt_is_copyable;

  // bool sameHost(const gptr& rhs) const {
  //   return ptr.getHost() == rhs.ptr.getHost();
  // }

  // void initialize(T* p) {
  //   ptr.setObj(p);
  //   ptr.setHost(p ? NetworkInterface::ID : 0);
  // }

};

// template<typename T>
// T* gptr<T>::inner_resolve(bool write) const {
//   T* retval = nullptr;
//   if (isLocal())
//     retval = static_cast<T*>(ptr.getObj());
//   else
//     retval = static_cast<T*>(getCacheManager().resolve(ptr, write));
//   if (inGaloisForEach)
//     return retval;

//   if (isLocal()) {
//     while(!getLocalDirectory().fetch(static_cast<Lockable*>(ptr.getObj()))) {}
//     return retval;
//   }
  
//   do {
//     getRemoteDirectory().fetch<T>(ptr, write ? ResolveFlag::RW : ResolveFlag::RO);
//     retval = static_cast<T*>(getCacheManager().resolve(ptr, write));
//   } while (!retval);
//   return retval;
// }


} //namespace Runtime
} //namespace galois

#endif //DISTSUPPORT
