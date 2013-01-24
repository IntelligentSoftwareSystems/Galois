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

namespace HIDDEN {
struct PendingStatus {
  Galois::Runtime::LL::CacheLineStorage<Galois::Runtime::PendingFlag> flag;
  PendingStatus(): flag(Galois::Runtime::NON_DET) { }
};

}

static HIDDEN::PendingStatus pendingStatus;

void Galois::Runtime::setPending(PendingFlag value) {
  pendingStatus.flag.data = value;
}

Galois::Runtime::PendingFlag Galois::Runtime::getPending () {
  return pendingStatus.flag.data;
}

void Galois::Runtime::doCheckWrite() {
  if (Galois::Runtime::getPending () == Galois::Runtime::PENDING) {
    throw Galois::Runtime::REACHED_FAILSAFE;
  }
}

void Galois::Runtime::setThreadContext(Galois::Runtime::SimpleRuntimeContext* n) {
  thread_cnx = n;
}

Galois::Runtime::SimpleRuntimeContext* Galois::Runtime::getThreadContext() {
  return thread_cnx;
}

bool Galois::Runtime::SimpleRuntimeContext::do_isMagicLock(Galois::Runtime::Lockable* L) {
  return (L->Owner.getValue() == CHK_LOCK);
}

bool Galois::Runtime::do_isMagicLock(Galois::Runtime::Lockable* C) {
  SimpleRuntimeContext cnx;
  return cnx.do_isMagicLock(C);
}

void Galois::Runtime::doAcquire(Galois::Runtime::Lockable* C) {
  SimpleRuntimeContext* cnx = getThreadContext();
  if (cnx)
    cnx->acquire(C);
}

void Galois::Runtime::SimpleRuntimeContext::do_setMagicLock(Galois::Runtime::Lockable* L) {
  L->Owner.setValue(USE_LOCK);
  // L->Owner.stealing_CAS(NULL,USE_LOCK);
  return;
}

void Galois::Runtime::do_setMagicLock(Galois::Runtime::Lockable* L) {
  SimpleRuntimeContext cnx;
  cnx.do_setMagicLock(L);
  return;
}

void *Galois::Runtime::SimpleRuntimeContext::do_getValue(Galois::Runtime::Lockable* L) {
  return ((void*)L->Owner.getValue());
}

void *Galois::Runtime::do_getValue(Galois::Runtime::Lockable* L) {
  SimpleRuntimeContext cnx;
  return cnx.do_getValue(L);
}

bool Galois::Runtime::SimpleRuntimeContext::do_trylock(Galois::Runtime::Lockable* L) {
  return L->Owner.try_lock();
}

bool Galois::Runtime::do_trylock(Galois::Runtime::Lockable* L) {
  SimpleRuntimeContext cnx;
  return cnx.do_trylock(L);
}

void Galois::Runtime::SimpleRuntimeContext::do_unlock(Galois::Runtime::Lockable* L) {
  L->Owner.unlock_and_clear();
}

void Galois::Runtime::do_unlock(Galois::Runtime::Lockable* C) {
  SimpleRuntimeContext cnx;
  cnx.do_unlock(C);
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
    //__sync_synchronize();
    LL::compilerBarrier();
    L->Owner.unlock_and_clear();

    ++numLocks;
  }

  // XXX not_ready = false;

  return numLocks;
}

void Galois::Runtime::breakLoop() {
  throw Galois::Runtime::BREAK;
}

void Galois::Runtime::signalConflict() {
        throw Galois::Runtime::CONFLICT; // Conflict
}

/*
void Galois::Runtime::SimpleRuntimeContext::acquire(Galois::Runtime::Lockable* L) {
  if (customAcquire) {
    sub_acquire(L);
    return;
  }
  if (L->Owner.try_lock()) {
    assert(!L->Owner.getValue());
    assert(!L->next);
    L->Owner.setValue(this);
    L->next = locks;
    locks = L;
  } else {
    if (L->Owner.getValue() != this) {
      Galois::Runtime::signalConflict();
    }
  }
}
*/

// Should allow the lock to be taken even if lock has magic value
void Galois::Runtime::SimpleRuntimeContext::acquire(Galois::Runtime::Lockable* L) {
  if (customAcquire) {
    sub_acquire(L);
    return;
  }
  if (L->Owner.try_lock()) {
    assert(!L->Owner.getValue());
    assert(!L->next);
    L->Owner.setValue(this);
    L->next = locks;
    locks = L;
  }
  else if (L->Owner.stealing_CAS(USE_LOCK,this)) {
    assert(!L->next);
    L->next = locks;
    locks = L;
  } else {
    if (L->Owner.getValue() != this) {
      Galois::Runtime::signalConflict();
    }
  }
}

void Galois::Runtime::SimpleRuntimeContext::sub_acquire(Galois::Runtime::Lockable* L) {
  assert(0 && "Shouldn't get here");
  abort();
}

//anchor vtable
Galois::Runtime::SimpleRuntimeContext::~SimpleRuntimeContext() {}

void Galois::Runtime::forceAbort() {
  signalConflict();
}
