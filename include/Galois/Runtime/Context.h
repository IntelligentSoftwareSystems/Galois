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

#include "Galois/config.h"
#include "Galois/MethodFlags.h"
#include "Galois/Runtime/ll/PtrLock.h"
#include "Galois/Runtime/ll/gio.h"

#include <boost/utility.hpp>

#include <cassert>
#include <cstdlib>

#ifdef GALOIS_USE_LONGJMP
#include <setjmp.h>
#endif

namespace Galois {
namespace Runtime {

enum ConflictFlag {
  CONFLICT = -1,
  NO_CONFLICT = 0,
  REACHED_FAILSAFE = 1,
  BREAK = 2
};

enum PendingFlag {
  NON_DET,
  PENDING,
  COMMITTING
};

//! Used by deterministic and ordered executor
void setPending(PendingFlag value);
PendingFlag getPending ();

//! used to release lock over exception path
static inline void clearConflictLock() { }

class SimpleRuntimeContext;

namespace DeterministicImpl {
template <typename, typename>
class DeterministicContext;
}

#ifdef GALOIS_USE_LONGJMP
extern __thread jmp_buf hackjmp;

/**
 * Stack-allocated classes that require their deconstructors to be called,
 * after an abort for instance, should inherit from this class. 
 */
class Releasable {
  Releasable* next;
public:
  Releasable();
  virtual ~Releasable() { }
  virtual void release() = 0;
  void releaseAll();
};
void clearReleasable();
#else
class Releasable {
public:
  virtual ~Releasable() { }
  virtual void release() = 0;
};
static inline void clearReleasable() { }
#endif


/**
 * All objects that may be locked (nodes primarily) must inherit from
 * Lockable. 
 */
class Lockable {
  LL::PtrLock<SimpleRuntimeContext, true> Owner;
  //! Use an intrusive list to track objects in a context without allocation overhead.
  Lockable* next;
  friend class SimpleRuntimeContext;
  template <typename, typename>
    friend class Galois::Runtime::DeterministicImpl::DeterministicContext;
  friend bool isAcquired(Lockable*);
  friend bool isAcquiredBy(Lockable*, SimpleRuntimeContext*);
public:
  LL::PtrLock<void, true> auxPtr;
  Lockable() :next(0) {}
};

class SimpleRuntimeContext: private boost::noncopyable {
protected:
  //! The locks we hold
  Lockable* locks;
  bool customAcquire;

  virtual void sub_acquire(Lockable* L);

  //0: fail, 1: new owner, 2: already owner
  int try_acquire(Lockable* L);
  void release(Lockable* L);
  
public:
  SimpleRuntimeContext(bool child = false): locks(0), customAcquire(child) { }
  virtual ~SimpleRuntimeContext() { }

  void start_iteration() {
    assert(!locks);
  }
  
  unsigned cancel_iteration();
  unsigned commit_iteration();
  void acquire(Lockable* L);
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
    GALOIS_DIE("shouldn't get here");
    return false;
  }
}

//! actual locking function.  Will always lock.
void doAcquire(Lockable* L);

inline void acquire(Lockable* L, SimpleRuntimeContext* cnx, const Galois::MethodFlag m) {
  if (shouldLock(m) && cnx)
    cnx->acquire(L);
}

//! Master function which handles conflict detection
//! used to acquire a lockable thing
inline void acquire(Lockable* L, Galois::MethodFlag m) {
  if (shouldLock(m)) {
    SimpleRuntimeContext* cnx = getThreadContext();
    acquire(L, cnx, m);
  }
}

inline bool isAcquired(Lockable* L) {
  return L->Owner.is_locked();
}

inline bool isAcquiredBy(Lockable* L, SimpleRuntimeContext* cnx) {
  return L->Owner.getValue() == cnx;
}

struct AlwaysLockObj {
  void operator()(Lockable* L) const {
    doAcquire(L);
  }
};

struct CheckedLockObj {
  Galois::MethodFlag m;
  CheckedLockObj(Galois::MethodFlag _m) :m(_m) {}
  void operator()(Lockable* L) const {
    acquire(L, m);
  }
};

//! Actually break for_each loop
void breakLoop();

void signalConflict(Lockable*);

void forceAbort();

}
} // end namespace Galois

#endif
