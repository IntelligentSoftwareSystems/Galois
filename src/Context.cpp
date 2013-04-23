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
  if (getPending () == PENDING) {
    throw failsafe_ex();
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
  //FIXME: not handled yet
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

  // XXX not_ready = false;

  return numLocks;
}

void Galois::Runtime::signalConflict(Lockable* L) {
  throw conflict_ex{L}; // Conflict
}

////////////////////////////////////////////////////////////////////////////////
// Simple Runtime Context
////////////////////////////////////////////////////////////////////////////////

using namespace Galois::Runtime;

int SimpleRuntimeContext::try_acquire(Lockable* L) {
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

void SimpleRuntimeContext::release(Lockable* L) {
  assert(L);
  assert(L->Owner.getValue() == this);
  assert(L->Owner.is_locked());
  assert(!L->next);
  L->Owner.unlock_and_clear();
}

bool SimpleRuntimeContext::swap_lock(Lockable* L, SimpleRuntimeContext* nptr) {
  assert(L);
  if (L->next) return false;
  return L->Owner.stealing_CAS(this,nptr);
}

// Should allow the lock to be taken even if lock has magic value
void SimpleRuntimeContext::acquire(Lockable* L) {
  int i;
  if (customAcquire) {
    sub_acquire(L);
  } else if ((i = try_acquire(L))) {
    if (i == 1) {
      L->next = locks;
      locks = L;
    }
  } else {
    signalConflict(L);
  }
}

bool SimpleRuntimeContext::swap_acquire(Lockable* L, SimpleRuntimeContext* nptr) {
  if (swap_lock(L,nptr)) {
    L->next = nptr->locks;
    nptr->locks = L;
    return true;
  }
  return false;
}

unsigned SimpleRuntimeContext::count_locks() {
  unsigned numLocks = 0;
  Lockable* L = locks;
  while (L) {
    ++numLocks;
    L = L->next;
  }

  return numLocks;
}

void SimpleRuntimeContext::sub_acquire(Lockable* L) {
  assert(0 && "Shouldn't get here");
  abort();
}

//anchor vtable
Galois::Runtime::SimpleRuntimeContext::~SimpleRuntimeContext() {}
