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
#include <setjmp.h>

//! Throwing exceptions can be a scalability bottleneck.

namespace Galois {
namespace Runtime {

//forward declaration for throw list
class Lockable;

//Things we can throw:
struct conflict_ex { Lockable* obj; };
struct break_ex {};
struct failsafe_ex{};
struct remote_ex; //tried to access a remote object

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
  friend class SimpleRuntimeContext;
  friend class Distributed::LocalDirectory;
  friend class Distributed::RemoteDirectory;
  template <typename, typename>
  friend struct Galois::Runtime::DeterministicImpl::DeterministicContext;
public:
  LL::PtrLock<void, true> auxPtr;
  Lockable() :next(0) {}
};

class SimpleRuntimeContext {
protected:
  //! The locks we hold
  Lockable* locks;
  bool customAcquire;

  virtual void sub_acquire(Lockable* L);

public:
  SimpleRuntimeContext(bool child = false): locks(0), customAcquire(child) { }
  virtual ~SimpleRuntimeContext();

  void start_iteration() {
    assert(!locks);
  }
  
  unsigned cancel_iteration();
  unsigned commit_iteration();
  void acquire(Lockable* L);
  bool do_trylock(Lockable* L);
  void do_unlock(Lockable* L);
  void *do_getValue(Lockable* L);
  bool do_isMagicLock(Lockable* L);
  void do_setMagicLock(Lockable* L);
};

//! get the current conflict detection class, may be null if not in parallel region
SimpleRuntimeContext* getThreadContext();

//! used by the parallel code to set up conflict detection per thread
void setThreadContext(SimpleRuntimeContext* n);


//! Helper function to decide if the conflict detection lock should be taken
static inline bool shouldLock(Galois::MethodFlag g) {
  // Mask out additional "optional" flags
  switch (g & MethodFlag::ALL) {
  case MethodFlag::NONE:
  case MethodFlag::SAVE_UNDO:
    return false;
  case MethodFlag::ALL:
  case MethodFlag::CHECK_CONFLICT:
    return true;
  default:
    GALOIS_ERROR(true, "shouldn't get here");
    return false;
  }
}

//! actual locking function.  Will always lock.
void doAcquire(Lockable* C);

//! Master function which handles conflict detection
//! used to acquire a lockable thing
static inline void acquire(Lockable* C, Galois::MethodFlag m) {
  if (shouldLock(m))
    doAcquire(C);
}

bool do_isMagicLock(Lockable* C);

static inline bool isMagicLock(Lockable* C) {
   return do_isMagicLock(C);
}

void do_setMagicLock(Lockable* C);

static inline void setMagicLock(Lockable* C) {
   do_setMagicLock(C);
   return;
}

void *do_getValue(Lockable* C);

static inline void *getValue(Lockable* C) {
   return do_getValue(C);
}

bool do_trylock(Lockable* C);

static inline bool trylock(Lockable* C) {
   return do_trylock(C);
}

void do_unlock(Lockable* C);

static inline void unlock(Lockable* C) {
   do_unlock(C);
   return;
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

//! Actually break for_each loop
void breakLoop();

void signalConflict(Lockable*);

void forceAbort();

}
} // end namespace Galois

#endif
