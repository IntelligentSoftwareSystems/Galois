/** Galois Distributed Pointer Manipulation -*- C++ -*-
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

#include "galois/runtime/Context.h"
#include "galois/runtime/Serialize.h"
#include "galois/runtime/Directory.h"
#include "galois/runtime/RemotePointer.h"

namespace galois {
namespace runtime {


SimpleRuntimeContext& getTransCnx();

#if 0

template<typename T>
T* resolve(const gptr<T>& p) {
  fatPointer ptr = p;
  // if (!ptr.second)
  //   return nullptr;

  remoteObjImpl<T>* robj = nullptr;
  Lockable* obj = ptr.getObj();
  if (ptr.getHost() != NetworkInterface::ID) {
    robj = getCacheManager().resolve<T>(ptr);
    obj = robj->getObj();
  }

  if (inGaloisForEach) {
    try {
      acquire(obj, galois::MethodFlag::ALL);
      return static_cast<T*>(obj);
    } catch (const conflict_ex& ex) {
      if (isAcquiredBy(obj, &getSystemDirectory())) {
        getSystemDirectory().fetch(ptr, static_cast<T*>(obj));
        throw remote_ex{ptr};
      } else {
        throw ex;
      }
    }
  } else { //serial code
    while (isAcquiredBy(obj, &getSystemDirectory())) {
      getSystemDirectory().fetch(ptr, static_cast<T*>(obj));
      doNetworkWork();
    }
    return static_cast<T*>(obj);
  }
}

template <typename T>
T* transientAcquire(const gptr<T>& p) {
  fatPointer ptr = p;
  if (!ptr.getObj())
    return nullptr;
  
  remoteObjImpl<T>* robj = nullptr;
  Lockable* obj = ptr.getObj();
  if (ptr.getHost() != NetworkInterface::ID) {
    robj = getCacheManager().resolve<T>(ptr);
    obj = robj->getObj();
  }

  while (!getTransCnx().tryAcquire(obj)) {
    if (isAcquiredBy(obj, &getSystemDirectory()))
      getSystemDirectory().fetch(ptr, static_cast<T*>(obj));
    doNetworkWork();
  }
  return static_cast<T*>(obj);
}

#if 0
template <typename T>
T* transientAcquireNonBlocking(const gptr<T>& p) {
  fatPointer ptr = p;
  if (ptr.getHost() == NetworkInterface::ID) {
    if (!getTransCnx().tryAcquire(ptr.getObj())) {
      getSystemLocalDirectory().recall<T>(ptr);
      return NULL;
    }
    return ptr.getObj();
  } else { // REMOTE
    T* rptr = getSystemRemoteDirectory().resolve<T>(ptr);
    //DATA RACE with delete
    if (getTransCnx().tryAcquire(rptr))
      return rptr;
    return NULL;
  }
}
#endif

template<typename T>
void transientRelease(const gptr<T>& p) {
  fatPointer ptr = p;
  remoteObjImpl<T>* robj = nullptr;
  Lockable* rptr = ptr.getObj();
  if (ptr.getHost() != NetworkInterface::ID) {
    robj = getCacheManager().resolve<T>(ptr);
    rptr = robj->getObj();
  }
  getTransCnx().release(rptr);
}

#endif

} //namespace runtime
} //namespace galois

#endif //DISTSUPPORT
