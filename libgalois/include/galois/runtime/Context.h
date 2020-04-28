/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_RUNTIME_CONTEXT_H
#define GALOIS_RUNTIME_CONTEXT_H

#include "galois/MethodFlags.h"
#include "galois/substrate/PtrLock.h"
#include "galois/gIO.h"
#include "galois/runtime/config.h"

#include <boost/utility.hpp>

#include <cassert>
#include <cstdlib>

#include <csetjmp>

namespace galois {
namespace runtime {

enum ConflictFlag {
  CONFLICT         = -1,
  NO_CONFLICT      = 0,
  REACHED_FAILSAFE = 1,
  BREAK            = 2
};

extern thread_local std::jmp_buf execFrame;

class Lockable;

[[noreturn]] inline void signalConflict(Lockable* = nullptr) {
#if defined(GALOIS_USE_LONGJMP_ABORT)
  std::longjmp(execFrame, CONFLICT);
  std::abort(); // shouldn't reach here after longjmp
#elif defined(GALOIS_USE_EXCEPTION_ABORT)
  throw CONFLICT;
#endif
}

#ifdef GALOIS_USE_EXP
bool owns(Lockable* lockable, MethodFlag m);
#endif

[[noreturn]] inline void signalFailSafe(void) {
#if defined(GALOIS_USE_LONGJMP_ABORT)
  std::longjmp(galois::runtime::execFrame, galois::runtime::REACHED_FAILSAFE);
  std::abort(); // shouldn't reach here after longjmp
#elif defined(GALOIS_USE_EXCEPTION_ABORT)
  throw REACHED_FAILSAFE;
#endif
}

//! used to release lock over exception path
static inline void clearConflictLock() {}

class LockManagerBase;

/**
 * All objects that may be locked (nodes primarily) must inherit from
 * Lockable.
 */
class Lockable {
  substrate::PtrLock<LockManagerBase> owner;
  //! Use an intrusive list to track neighborhood of a context without
  //! allocation overhead. Works for cases where a Lockable needs to be only in
  //! one context's neighborhood list
  Lockable* next;
  friend class LockManagerBase;
  friend class SimpleRuntimeContext;

public:
  Lockable() : next(0) {}
};

class LockManagerBase : private boost::noncopyable {
protected:
  enum AcquireStatus { FAIL, NEW_OWNER, ALREADY_OWNER };

  AcquireStatus tryAcquire(Lockable* lockable);

  inline bool stealByCAS(Lockable* lockable, LockManagerBase* other) {
    assert(lockable != nullptr);
    return lockable->owner.stealing_CAS(other, this);
  }

  inline bool CASowner(Lockable* lockable, LockManagerBase* other) {
    assert(lockable != nullptr);
    return lockable->owner.CAS(other, this);
  }

  inline void setOwner(Lockable* lockable) {
    assert(lockable != nullptr);
    assert(!lockable->owner.getValue());
    lockable->owner.setValue(this);
  }

  inline void release(Lockable* lockable) {
    assert(lockable != nullptr);
    assert(getOwner(lockable) == this);
    lockable->owner.unlock_and_clear();
  }

  inline static bool tryLock(Lockable* lockable) {
    assert(lockable != nullptr);
    return lockable->owner.try_lock();
  }

  inline static LockManagerBase* getOwner(Lockable* lockable) {
    assert(lockable != nullptr);
    return lockable->owner.getValue();
  }
};

class SimpleRuntimeContext : public LockManagerBase {
  //! The locks we hold
  Lockable* locks;
  bool customAcquire;

protected:
  friend void doAcquire(Lockable*, galois::MethodFlag);

  static SimpleRuntimeContext* getOwner(Lockable* lockable) {
    LockManagerBase* owner = LockManagerBase::getOwner(lockable);
    return static_cast<SimpleRuntimeContext*>(owner);
  }

  virtual void subAcquire(Lockable* lockable, galois::MethodFlag m);

  void addToNhood(Lockable* lockable) {
    assert(!lockable->next);
    lockable->next = locks;
    locks          = lockable;
  }

  void acquire(Lockable* lockable, galois::MethodFlag m) {
    AcquireStatus i;
    if (customAcquire) {
      subAcquire(lockable, m);
    } else if ((i = tryAcquire(lockable)) != AcquireStatus::FAIL) {
      if (i == AcquireStatus::NEW_OWNER) {
        addToNhood(lockable);
      }
    } else {
      signalConflict(lockable);
    }
  }

  void release(Lockable* lockable);

public:
  SimpleRuntimeContext(bool child = false) : locks(0), customAcquire(child) {}
  virtual ~SimpleRuntimeContext() {}

  void startIteration() { assert(!locks); }

  unsigned cancelIteration();
  unsigned commitIteration();
};

//! get the current conflict detection class, may be null if not in parallel
//! region
SimpleRuntimeContext* getThreadContext();

//! used by the parallel code to set up conflict detection per thread
void setThreadContext(SimpleRuntimeContext* n);

//! Helper function to decide if the conflict detection lock should be taken
inline bool shouldLock(const galois::MethodFlag g) {
  // Mask out additional "optional" flags
  switch (g & galois::MethodFlag::INTERNAL_MASK) {
  case MethodFlag::UNPROTECTED:
  case MethodFlag::PREVIOUS:
    return false;

  case MethodFlag::READ:
  case MethodFlag::WRITE:
    return true;

  default:
    // XXX(ddn): Adding error checking code here either upsets the inlining
    // heuristics or icache behavior. Avoid complex code if possible.
    // GALOIS_DIE("shouldn't get here");
    assert(false);
  }
  return false;
}

//! actual locking function.  Will always lock.
inline void doAcquire(Lockable* lockable, galois::MethodFlag m) {
  SimpleRuntimeContext* ctx = getThreadContext();
  if (ctx)
    ctx->acquire(lockable, m);
}

//! Master function which handles conflict detection
//! used to acquire a lockable thing
inline void acquire(Lockable* lockable, galois::MethodFlag m) {
  if (shouldLock(m))
    doAcquire(lockable, m);
}

struct AlwaysLockObj {
  void operator()(Lockable* lockable) const {
    doAcquire(lockable, galois::MethodFlag::WRITE);
  }
};

struct CheckedLockObj {
  galois::MethodFlag m;
  CheckedLockObj(galois::MethodFlag _m) : m(_m) {}
  void operator()(Lockable* lockable) const { acquire(lockable, m); }
};

} // namespace runtime
} // end namespace galois

#endif
