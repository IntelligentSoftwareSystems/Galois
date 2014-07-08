/** simple galois context and contention manager -*- C++ -*-
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
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_CONTEXT_H
#define GALOIS_RUNTIME_CONTEXT_H

//#include "Galois/config.h"
#include "Galois/MethodFlags.h"
#include "Galois/Runtime/Lockable.h"
#include "Galois/Runtime/RemotePointer.h"
#include "Galois/Runtime/CacheManager.h"
#include "Galois/Runtime/Directory.h"
//#include "Galois/Runtime/ll/PtrLock.h"
//#include "Galois/Runtime/ll/gio.h"

#include <boost/utility.hpp>

#include <cassert>
#include <cstdlib>

namespace Galois {
namespace Runtime {

//Things we can throw:
struct conflict_ex { Lockable* ptr; };
struct failsafe_ex {};
//struct remote_ex { fatPointer ptr; };

void signalConflict(Lockable*);

void forceAbort();


class SimpleRuntimeContext { //: public LockManagerBase {
public: //FIXME
  //! The locks we hold
  LockManagerBase locks;
  bool customAcquire;

  //protected:
  friend void doAcquire(Lockable*);

  // static SimpleRuntimeContext* getOwner(Lockable* lockable) {
  //   LockManagerBase* owner = LockManagerBase::getOwner (lockable);
  //   return static_cast<SimpleRuntimeContext*>(owner);
  // }

  virtual bool subAcquire(Lockable* lockable);
  virtual unsigned subCommit();
  virtual unsigned subCancel();

public:

  bool acquire(Lockable* lockable);
  void release(Lockable* lockable);

  SimpleRuntimeContext(bool child = false): customAcquire(child) { }
  virtual ~SimpleRuntimeContext() { }

  void startIteration() {
    assert(locks.empty());
  }
  
  unsigned cancelIteration();
  unsigned commitIteration();
};

//! get the current conflict detection class, may be null if not in parallel region
SimpleRuntimeContext* getThreadContext();

//! used by the parallel code to set up conflict detection per thread
void setThreadContext(SimpleRuntimeContext* n);

//! Helper function to decide if the conflict detection lock should be taken
inline bool shouldLock(const Galois::MethodFlag g) {
  // Mask out additional "optional" flags
  switch (g & ALL) {
  case NONE:
  case SAVE_UNDO:
    return false;
  case ALL:
  case CHECK_CONFLICT:
    return true;
  default:
    // XXX(ddn): Adding error checking code here either upsets the inlining
    // heuristics or icache behavior. Avoid complex code if possible.
    //GALOIS_DIE("shouldn't get here");
    assert(false);
  }
  return false;
}

// //! actual locking function.  Will always lock.
// inline void doAcquire(Lockable* lockable) {
//   SimpleRuntimeContext* ctx = getThreadContext();
//   if (ctx)
//     ctx->acquire(lockable);
// }

//! Master function which handles conflict detection
//! used to acquire a lockable thing
inline void acquire(Lockable* lockable, Galois::MethodFlag m) {
  if (shouldLock(m)) {
    SimpleRuntimeContext* ctx = getThreadContext();
    if (ctx)
      if (!ctx->acquire(lockable))
        signalConflict(lockable);
    //doAcquire (lockable);
  }
}

template<typename T>
class gptr;

template<typename T>
inline void acquire(gptr<T> ptr, Galois::MethodFlag m) {
  if (inGaloisForEach) {
    T* obj = ptr.resolve();
    if (!obj) {
      //FIXME Better resolve flag
      getRemoteDirectory().fetch<T>(static_cast<fatPointer>(ptr), ResolveFlag::RW);
      throw remote_ex{static_cast<fatPointer>(ptr), m, &RemoteDirectory::fetch<T>};
    }
    if (shouldLock(m)) {
      SimpleRuntimeContext* ctx = getThreadContext();
      Lockable* lockable = static_cast<Lockable*>(obj);
      if (ctx)
        if (!ctx->acquire(lockable))
          throw remote_ex{static_cast<fatPointer>(ptr), m, &RemoteDirectory::fetch<T>};
    }
  } else {
    serial_acquire(ptr);
  }
}

//! ensures conherent serial access
template<typename T>
inline void serial_acquire(gptr<T> ptr) {
  do {
    T* obj = ptr.resolve();
    if (!obj) {
      //FIXME Better resolve flag
      getRemoteDirectory().fetch<T>(static_cast<fatPointer>(ptr), ResolveFlag::RW);
    } else if (LockManagerBase::isAcquiredAny(obj)) {
      getLocalDirectory().fetch<T>(static_cast<fatPointer>(ptr), ResolveFlag::RW);
    } else {
      return;
    }
    doNetworkWork();
  } while(true);
}

template<typename T>
inline void prefetch(gptr<T> ptr, Galois::MethodFlag m = MethodFlag::ALL) {
  T* obj = ptr.resolve();
  if (!obj) {
    //FIXME Better resolve flag
    getRemoteDirectory().fetch<T>(static_cast<fatPointer>(ptr), ResolveFlag::RW);
  } else if (LockManagerBase::isAcquiredAny(obj)) {
    getLocalDirectory().fetch<T>(static_cast<fatPointer>(ptr), ResolveFlag::RW);
  }
}

// inline bool isAcquired(Lockable* C) {
//   return C->owner.is_locked();
// }

// inline bool isAcquiredBy(Lockable* C, SimpleRuntimeContext* cnx) {
//   return C->owner.getValue() == cnx;
// }

// struct AlwaysLockObj {
//   void operator()(Lockable* lockable) const {
//     doAcquire(lockable);
//   }
// };

// struct CheckedLockObj {
//   Galois::MethodFlag m;
//   CheckedLockObj(Galois::MethodFlag _m) :m(_m) {}
//   void operator()(Lockable* lockable) const {
//     acquire(lockable, m);
//   }
// };


}
} // end namespace Galois

#endif
