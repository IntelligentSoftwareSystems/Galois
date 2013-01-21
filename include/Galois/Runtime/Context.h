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

#include "Galois/Callbacks.h"
#include "Galois/MethodFlags.h"
#include "Galois/Runtime/ll/PtrLock.h"
#include "Galois/Runtime/ll/gio.h"
#include <cassert>
#include <cstdlib>
#include <setjmp.h>

//! Throwing exceptions can be a scalability bottleneck.
//! Set to zero to use longjmp hack, otherwise make sure that
//! you use a fixed c++ runtime that improves scalability of
//! exceptions.
//    --- mdhanapal removing all the longjmp hacks for Directory
//    options
#define GALOIS_USE_EXCEPTION_HANDLER 1

namespace Galois {
namespace Runtime {

enum ConflictFlag {
  CONFLICT = -1,
  REACHED_FAILSAFE = 1,
  BREAK = 2,
  REMOTE = 3
};

enum PendingFlag {
  NON_DET,
  PENDING,
  COMMITTING
};

//! Used by deterministic and ordered executor
void setPending(PendingFlag value);

//! used to release lock over exception path
static inline void clearConflictLock() { }

#define CHK_LOCK ((Galois::Runtime::SimpleRuntimeContext*)0x422)
#define USE_LOCK ((Galois::Runtime::SimpleRuntimeContext*)0x423)

class SimpleRuntimeContext;
namespace Distributed {
class LocalDirectory;
class RemoteDirectory;
}
class DeterministicRuntimeContext;

//! All objects that may be locked (nodes primarily) must inherit from Lockable.
//! Use an intrusive list to track objects in a context without allocation overhead
class Lockable {
  LL::PtrLock<SimpleRuntimeContext, true> Owner;
  Lockable* next;
  friend class SimpleRuntimeContext;
  friend class Distributed::LocalDirectory;
  friend class Distributed::RemoteDirectory;
  friend class DeterministicRuntimeContext;
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

class DeterministicRuntimeContext: public SimpleRuntimeContext {
protected:
  //! Iteration id for deterministic execution
  union {
    unsigned long id;
    void* comp_data;
  } data;
  //! Flag to abort other iterations for deterministic and ordered execution
  unsigned long not_ready;

  //! User-defined comparison between iterations for ordered execution
  Galois::CompareCallback* comp;

  virtual void sub_acquire(Lockable* L);

public:
  DeterministicRuntimeContext(): SimpleRuntimeContext(true), not_ready(0), comp(0) { data.id = 0; data.comp_data = 0; }

  void set_id(unsigned long i) { data.id = i; }
  bool is_ready() { return !not_ready; }

  void set_comp_data(void* ptr) { data.comp_data = ptr; }
  void set_comp(Galois::CompareCallback* fn) { comp = fn; }
};

//! Helper function to decide if the conflict detection lock should be taken
static inline bool shouldLock(Galois::MethodFlag g) {
  // Mask out additional "optional" flags
  switch (g & Galois::ALL) {
  case Galois::NONE:
  case Galois::SAVE_UNDO:
    return false;
  case Galois::ALL:
  case Galois::CHECK_CONFLICT:
    return true;
  }
  GALOIS_ERROR(true, "shouldn't get here");
  return false;
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

void signalConflict();

void forceAbort();

} //Runtime
} //Galois



#endif
