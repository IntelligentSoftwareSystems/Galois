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
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/MethodFlags.h"
#include "Galois/Runtime/ll/SimpleLock.h"

#if G_USE_EH
#else
__thread jmp_buf GaloisRuntime::hackjmp;
#endif

//! Global thread context for each active thread
static __thread GaloisRuntime::SimpleRuntimeContext* thread_cnx = 0;

#ifdef GALOIS_USE_CONFLICT_LOCK
static GaloisRuntime::LL::SimpleLock<true> conflictLock;

void GaloisRuntime::clearConflictLock() {
  conflictLock.unlock();
}

static inline void lockConflictLock() {
  conflictLock.lock();
}
#else
static inline void lockConflictLock() { }
#endif

#ifdef GALOIS_DET
static GaloisRuntime::PendingFlag pendingFlag = GaloisRuntime::NON_DET;

void GaloisRuntime::setPending(PendingFlag value) {
  pendingFlag = value;
}

void GaloisRuntime::doCheckWrite() {
  if (pendingFlag == PENDING) {
    lockConflictLock();
    throw GaloisRuntime::REACHED_FAILSAFE;
  }
}
#endif

void GaloisRuntime::setThreadContext(GaloisRuntime::SimpleRuntimeContext* n) {
  thread_cnx = n;
}

GaloisRuntime::SimpleRuntimeContext* GaloisRuntime::getThreadContext() {
  return thread_cnx;
}

void GaloisRuntime::doAcquire(GaloisRuntime::Lockable* C) {
  SimpleRuntimeContext* cnx = getThreadContext();
  if (cnx)
    cnx->acquire(C);
}

unsigned GaloisRuntime::SimpleRuntimeContext::cancel_iteration() {
  //FIXME: not handled yet
  return commit_iteration();
}

unsigned GaloisRuntime::SimpleRuntimeContext::commit_iteration() {
  unsigned numLocks = 0;
  while (locks) {
    //ORDER MATTERS!
    Lockable* L = locks;
    locks = L->next;
    L->next = 0;
    //__sync_synchronize();
    LL::compilerBarrier();
    L->Owner.unlock_and_clear();

    ++numLocks;
  }

  not_ready = false;

  return numLocks;
}

void GaloisRuntime::breakLoop() {
  lockConflictLock();
  throw GaloisRuntime::BREAK;
}

void GaloisRuntime::SimpleRuntimeContext::acquire(GaloisRuntime::Lockable* L) {
#ifdef GALOIS_DET
  if (pendingFlag != NON_DET) {
    if (L->Owner.try_lock()) {
      assert(!L->next);
      L->next = locks;
      locks = L;
    }

    SimpleRuntimeContext* other;
    do {
      other = L->Owner.getValue();
      if (other == this)
        return;
      if (other && other->id < id) {
        if (pendingFlag == PENDING) {
          // A lock I should have but can't get
          not_ready = 1;
          return; 
        } else {
          lockConflictLock();
          throw GaloisRuntime::CONFLICT;
        }
      }
      // Allow new locks on new nodes only
      if (pendingFlag == COMMITTING && other) {
        assert(0 && "Grabbing more locks during commit phase");
        abort();
      }
    } while (!L->Owner.stealing_CAS(other, this));

    // Disable loser
    if (other)
      other->not_ready = 1; // only need atomic write

    return;
  }
#endif
  if (L->Owner.try_lock()) {
    assert(!L->Owner.getValue());
    assert(!L->next);
    L->Owner.setValue(this);
    L->next = locks;
    locks = L;
  } else {
    if (L->Owner.getValue() != this) {
#if G_USE_EH
      //ConflictLock.lock();
      throw GaloisRuntime::CONFLICT; // Conflict
#else
      longjmp(hackjmp, 1);
#endif
    }
  }
}
