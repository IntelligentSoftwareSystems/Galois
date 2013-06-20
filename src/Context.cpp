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

#if GALOIS_USE_EXCEPTION_HANDLER
#else
static __thread jmp_buf Galois::Runtime::hackjmp;
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

Galois::Runtime::PendingFlag Galois::Runtime::getPending () {
  return pendingStatus.flag.data;
}

void Galois::Runtime::doCheckWrite() {
  if (Galois::Runtime::getPending() == Galois::Runtime::PENDING) {
#if GALOIS_USE_EXCEPTION_HANDLER
    throw Galois::Runtime::REACHED_FAILSAFE;
#else
    longjmp(hackjmp, Galois::Runtime::REACHED_FAILSAFE);
#endif
  }
}

void Galois::Runtime::setThreadContext(Galois::Runtime::SimpleRuntimeContext* n) {
  thread_cnx = n;
}

Galois::Runtime::SimpleRuntimeContext* Galois::Runtime::getThreadContext() {
  return thread_cnx;
}

void Galois::Runtime::doAcquire(Galois::Runtime::Lockable* C) {
  SimpleRuntimeContext* cnx = getThreadContext();
  if (cnx)
    cnx->acquire(C);
}

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

void Galois::Runtime::breakLoop() {
#if GALOIS_USE_EXCEPTION_HANDLER
  throw Galois::Runtime::BREAK;
#else
  longjmp(hackjmp, Galois::Runtime::BREAK);
#endif
}

void Galois::Runtime::signalConflict(Lockable* L) {
#if GALOIS_USE_EXCEPTION_HANDLER
  throw Galois::Runtime::CONFLICT; // Conflict
#else
  longjmp(hackjmp, Galois::Runtime::CONFLICT);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Simple Runtime Context
////////////////////////////////////////////////////////////////////////////////

int Galois::Runtime::SimpleRuntimeContext::try_acquire(Galois::Runtime::Lockable* L) {
  assert(L);
  if (L->Owner.try_lock()) {
    assert(!L->Owner.getValue());
    L->Owner.setValue(this);
    return 1;
  } else if (L->Owner.getValue() == this) {
    return 2;
  }
  return 0;
}

void Galois::Runtime::SimpleRuntimeContext::release(Galois::Runtime::Lockable* L) {
  assert(L);
  // The deterministic executor, for instance, steals locks from other
  // iterations
  assert(customAcquire || L->Owner.getValue() == this);
  assert(!L->next);
  L->Owner.unlock_and_clear();
}

void Galois::Runtime::SimpleRuntimeContext::acquire(Galois::Runtime::Lockable* L) {
  int i;
  if (customAcquire) {
    sub_acquire(L);
  } else if ((i = try_acquire(L))) {
    if (i == 1) {
      L->next = locks;
      locks = L;
    }
  } else {
    Galois::Runtime::signalConflict(L);
  }
}

void Galois::Runtime::SimpleRuntimeContext::sub_acquire(Galois::Runtime::Lockable* L) {
  GALOIS_DIE("Shouldn't get here");
}

//anchor vtable
Galois::Runtime::SimpleRuntimeContext::~SimpleRuntimeContext() {}


void Galois::Runtime::forceAbort() {
  signalConflict(NULL);
}
