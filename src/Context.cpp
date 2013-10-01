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
#include "Galois/Runtime/ll/CacheLineStorage.h"

#include <stdio.h>

#ifdef GALOIS_USE_LONGJMP
#include <setjmp.h>

__thread jmp_buf Galois::Runtime::hackjmp;
static __thread Galois::Runtime::Releasable* releasableHead = 0;

Galois::Runtime::Releasable::Releasable() {
  this->next = releasableHead;
  releasableHead = this;
}

void Galois::Runtime::Releasable::releaseAll() {
  Galois::Runtime::Releasable* head = this;
  while (head) {
    head->release();
    Galois::Runtime::Releasable* next = head->next;
    head->~Releasable();
    head = next;
  }
}

void Galois::Runtime::clearReleasable() {
  releasableHead = 0;
}
#endif

//! Global thread context for each active thread
static __thread Galois::Runtime::SimpleRuntimeContext* thread_cnx = 0;

namespace {

struct PendingStatus {
  Galois::Runtime::LL::CacheLineStorage<Galois::Runtime::PendingFlag> flag;
  PendingStatus(): flag(Galois::Runtime::NON_DET) { }
};

PendingStatus pendingStatus;

}

void Galois::Runtime::setPending(Galois::Runtime::PendingFlag value) {
  pendingStatus.flag.data = value;
}

Galois::Runtime::PendingFlag Galois::Runtime::getPending() {
  return pendingStatus.flag.data;
}

void Galois::Runtime::doCheckWrite() {
  if (Galois::Runtime::getPending() == Galois::Runtime::PENDING) {
#ifdef GALOIS_USE_LONGJMP
    if (releasableHead) releasableHead->releaseAll();
    longjmp(hackjmp, Galois::Runtime::REACHED_FAILSAFE);
#else
    throw Galois::Runtime::REACHED_FAILSAFE;
#endif
  }
}

void Galois::Runtime::setThreadContext(Galois::Runtime::SimpleRuntimeContext* ctx) {
  thread_cnx = ctx;
}

Galois::Runtime::SimpleRuntimeContext* Galois::Runtime::getThreadContext() {
  return thread_cnx;
}

void Galois::Runtime::doAcquire(Galois::Runtime::Lockable* lockable) {
  SimpleRuntimeContext* cnx = getThreadContext();
  if (cnx)
    cnx->acquire(lockable);
}

void Galois::Runtime::breakLoop() {
#ifdef GALOIS_USE_LONGJMP
  if (releasableHead) releasableHead->releaseAll();
  longjmp(hackjmp, Galois::Runtime::BREAK);
#else
  throw Galois::Runtime::BREAK;
#endif
}

void Galois::Runtime::signalConflict(Lockable* lockable) {
#ifdef GALOIS_USE_LONGJMP
  if (releasableHead) releasableHead->releaseAll();
  longjmp(hackjmp, Galois::Runtime::CONFLICT);
#else
  throw Galois::Runtime::CONFLICT; // Conflict
#endif
}

void Galois::Runtime::forceAbort() {
  signalConflict(NULL);
}

////////////////////////////////////////////////////////////////////////////////
// Runtime Context
////////////////////////////////////////////////////////////////////////////////

#if !defined(GALOIS_USE_SEQ_ONLY)
unsigned Galois::Runtime::SimpleRuntimeContext::cancel_iteration() {
  return commit_iteration();
}

unsigned Galois::Runtime::SimpleRuntimeContext::commit_iteration() {
  unsigned numLocks = 0;
  while (locks) {
    //ORDER MATTERS!
    Lockable* L = locks;
    locks = L->next;
    L->next = 0;
    LL::compilerBarrier();
    release(L);
    ++numLocks;
  }

  return numLocks;
}

int Galois::Runtime::SimpleRuntimeContext::tryAcquire(Galois::Runtime::Lockable* lockable) {
  assert(lockable);
  if (tryLockOwner(lockable)) {
    setOwner(lockable);
    return 1;
  } else if (getOwner(lockable) == this) {
    return 2;
  }
  return 0;
}

void Galois::Runtime::SimpleRuntimeContext::release(Galois::Runtime::Lockable* lockable) {
  assert(lockable);
  // The deterministic executor, for instance, steals locks from other
  // iterations
  assert(customAcquire || getOwner(lockable) == this);
  assert(!lockable->next);
  lockable->owner.unlock_and_clear();
}

void Galois::Runtime::SimpleRuntimeContext::acquire(Galois::Runtime::Lockable* lockable) {
  int i;
  if (customAcquire) {
    this->subAcquire(lockable);
  } else if ((i = tryAcquire(lockable))) {
    if (i == 1)
      insertLockable(lockable);
  } else {
    Galois::Runtime::signalConflict(lockable);
  }
}
#endif

void Galois::Runtime::SimpleRuntimeContext::subAcquire(Galois::Runtime::Lockable* lockable) {
  GALOIS_DIE("Shouldn't get here");
}

