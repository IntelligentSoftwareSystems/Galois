#ifndef GALOIS_RUNTIME_CONTEXT_H
#define GALOIS_RUNTIME_CONTEXT_H

#include "galois/MethodFlags.h"
#include "galois/substrate/PtrLock.h"
#include "galois/gIO.h"

#include <boost/utility.hpp>

#include <cassert>
#include <cstdlib>

#ifdef GALOIS_USE_LONGJMP_ABORT
#include <setjmp.h>
#endif

namespace galois {
namespace runtime {

enum ConflictFlag {
  CONFLICT = -1,
  NO_CONFLICT = 0,
  REACHED_FAILSAFE = 1,
  BREAK = 2
};

#ifdef GALOIS_USE_LONGJMP_ABORT
extern thread_local jmp_buf execFrame;
#endif

//! used to release lock over exception path
static inline void clearConflictLock() { }

class LockManagerBase;

/**
 * All objects that may be locked (nodes primarily) must inherit from
 * Lockable.
 */
class Lockable {
  substrate::PtrLock<LockManagerBase> owner;
  //! Use an intrusive list to track neighborhood of a context without allocation overhead.
  //! Works for cases where a Lockable needs to be only in one context's neighborhood list
  Lockable* next;
  friend class LockManagerBase;
  friend class SimpleRuntimeContext;
public:
  Lockable() :next(0) {}
};

class LockManagerBase: private boost::noncopyable {
protected:
  enum AcquireStatus {
    FAIL, NEW_OWNER, ALREADY_OWNER
  };

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

class SimpleRuntimeContext: public LockManagerBase {
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
    locks = lockable;
  }

  void acquire(Lockable* lockable, galois::MethodFlag m);
  void release(Lockable* lockable);

public:
  SimpleRuntimeContext(bool child = false): locks(0), customAcquire(child) { }
  virtual ~SimpleRuntimeContext() { }

  void startIteration() {
    assert(!locks);
  }

  unsigned cancelIteration();
  unsigned commitIteration();

#ifdef GALOIS_USE_EXP
  virtual bool owns(Lockable* lockable, galois::MethodFlag m) const;
#endif
};

//! get the current conflict detection class, may be null if not in parallel region
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
    //GALOIS_DIE("shouldn't get here");
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
  CheckedLockObj(galois::MethodFlag _m) :m(_m) {}
  void operator()(Lockable* lockable) const {
    acquire(lockable, m);
  }
};

void signalConflict(Lockable* = nullptr);

#ifdef GALOIS_USE_EXP
bool owns(Lockable* lockable, MethodFlag m);
#endif

void signalFailSafe(void);

}
} // end namespace galois

#endif
