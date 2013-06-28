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

#include "Galois/MethodFlags.h"
#include "Galois/Runtime/ll/PtrLock.h"
#include "Galois/Runtime/ll/gio.h"
#include <cassert>
#include <cstdlib>

//! Throwing exceptions can be a scalability bottleneck.

namespace Galois {
namespace Runtime {

//forward declaration for throw list
class Lockable;

//Things we can throw:
struct conflict_ex { Lockable* obj; };
struct failsafe_ex{};
//struct remote_ex { fatPointer ptr; };

enum PendingFlag {
  NON_DET,
  PENDING,
  COMMITTING
};

//! Used by deterministic and ordered executor
void setPending(PendingFlag value);
PendingFlag getPending ();

#define CHK_LOCK ((Galois::Runtime::SimpleRuntimeContext*)0x422)
#define USE_LOCK ((Galois::Runtime::SimpleRuntimeContext*)0x423)

class SimpleRuntimeContext;
namespace Distributed {
class LocalDirectory;
class RemoteDirectory;
}

namespace DeterministicImpl {
template <typename, typename>
struct DeterministicContext;
}

//! All objects that may be locked (nodes primarily) must inherit from Lockable.
//! Use an intrusive list to track objects in a context without allocation overhead
class Lockable {
  LL::PtrLock<SimpleRuntimeContext, true> Owner;
  Lockable* next;

  //Lots of friends!
  friend class SimpleRuntimeContext;
  friend class Distributed::LocalDirectory;
  friend class Distributed::RemoteDirectory;
  template <typename, typename>
  friend struct Galois::Runtime::DeterministicImpl::DeterministicContext;
  friend bool isAcquired(Lockable*);
  friend bool isAcquiredBy(Lockable*, SimpleRuntimeContext*);
public:
  uintptr_t auxData;
  Lockable() :next(0) {}
};

class SimpleRuntimeContext {
protected:
  //! The locks we hold
  Lockable* locks;
  bool customAcquire;

  virtual void sub_acquire(Lockable* L);

public:
  //0: fail, 1: new owner, 2: already owner
  int try_acquire(Lockable* L);
  void release(Lockable* L);
  unsigned count_locks();

  public:
  SimpleRuntimeContext(bool child = false): locks(0), customAcquire(child) {
    //    LL::gDebug("SRC: ", this);
  }
  virtual ~SimpleRuntimeContext();

  void start_iteration() {
    assert(!locks);
  }
  
  unsigned cancel_iteration();
  unsigned commit_iteration();
  void acquire(Lockable* L);
  bool swap_lock(Lockable* L, SimpleRuntimeContext* nptr);
  bool swap_acquire(Lockable* L, SimpleRuntimeContext* nptr);
};

//! get the current conflict detection class, may be null if not in parallel region
SimpleRuntimeContext* getThreadContext();

//! used by the parallel code to set up conflict detection per thread
void setThreadContext(SimpleRuntimeContext* n);


//! Helper function to decide if the conflict detection lock should be taken
inline bool shouldLock(const Galois::MethodFlag g) {
  // Mask out additional "optional" flags
  switch (g & MethodFlag::ALL) {
  case MethodFlag::NONE:
  case MethodFlag::SAVE_UNDO:
    return false;
  case MethodFlag::ALL:
  case MethodFlag::CHECK_CONFLICT:
    return true;
  default:
    GALOIS_DIE("shouldn't get here");
    return false;
  }
}

//! actual locking function.  Will always lock.
void doAcquire(Lockable* C);

inline void acquire(Lockable* C, SimpleRuntimeContext* cnx, const Galois::MethodFlag m) {
  if (shouldLock(m) && cnx)
    cnx->acquire(C);
}

//! Master function which handles conflict detection
//! used to acquire a lockable thing
inline void acquire(Lockable* C, Galois::MethodFlag m) {
  if (shouldLock(m)) {
    SimpleRuntimeContext* cnx = getThreadContext();
    acquire(C, cnx, m);
  }
}

inline bool isAcquired(Lockable* C) {
  return C->Owner.is_locked();
}

inline bool isAcquiredBy(Lockable* C, SimpleRuntimeContext* cnx) {
  return C->Owner.getValue() == cnx;
}

struct AlwaysLockObj {
  void operator()(Lockable* C) const {
    doAcquire(C);
  }
};

struct CheckedLockObj {
  Galois::MethodFlag m;
  CheckedLockObj(Galois::MethodFlag _m) :m(_m) {}
  void operator()(Lockable* C) const {
    acquire(C, m);
  }
};

void signalConflict(Lockable*);

}
} // end namespace Galois

#endif
