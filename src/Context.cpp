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

#if GALOIS_USE_EXCEPTION_HANDLER
#else
__thread jmp_buf GaloisRuntime::hackjmp;
#endif

#define USE_LOCK ((GaloisRuntime::SimpleRuntimeContext*)0x423)

//! Global thread context for each active thread
static __thread GaloisRuntime::SimpleRuntimeContext* thread_cnx = 0;

namespace {
struct PendingFlag {
  GaloisRuntime::LL::CacheLineStorage<GaloisRuntime::PendingFlag> flag;
  PendingFlag(): flag(GaloisRuntime::NON_DET) { }
};

}

static PendingFlag pendingFlag;

void GaloisRuntime::setPending(PendingFlag value) {
  pendingFlag.flag.data = value;
}

void GaloisRuntime::doCheckWrite() {
  if (pendingFlag.flag.data == PENDING) {
#if GALOIS_USE_EXCEPTION_HANDLER
    throw GaloisRuntime::REACHED_FAILSAFE;
#else
    longjmp(hackjmp, GaloisRuntime::REACHED_FAILSAFE);
#endif
  }
}

void GaloisRuntime::setThreadContext(GaloisRuntime::SimpleRuntimeContext* n) {
  thread_cnx = n;
}

GaloisRuntime::SimpleRuntimeContext* GaloisRuntime::getThreadContext() {
  return thread_cnx;
}

void GaloisRuntime::SimpleRuntimeContext::lockAcquire(GaloisRuntime::Lockable* L) {
  if (L->Owner.stealing_CAS(USE_LOCK,this)) {
    assert(!L->next);
    L->next = locks;
    locks = L;
  } else {
    if (L->Owner.getValue() != this) {
      GaloisRuntime::signalConflict();
    }
  }
}

void GaloisRuntime::doLockAcquire(GaloisRuntime::Lockable* C) {
  SimpleRuntimeContext* cnx = getThreadContext();
  if (cnx)
    cnx->lockAcquire(C);
}

void GaloisRuntime::doAcquire(GaloisRuntime::Lockable* C) {
  SimpleRuntimeContext* cnx = getThreadContext();
  if (cnx)
    cnx->acquire(C);
}

void GaloisRuntime::SimpleRuntimeContext::do_setLockValue(GaloisRuntime::Lockable* L) {
  // L->Owner.setValue(USE_LOCK);
  L->Owner.stealing_CAS(NULL,USE_LOCK);
  return;
}

void GaloisRuntime::do_setLockValue(GaloisRuntime::Lockable* L) {
  SimpleRuntimeContext cnx;
  cnx.do_setLockValue(L);
  return;
}

void *GaloisRuntime::SimpleRuntimeContext::do_getValue(GaloisRuntime::Lockable* L) {
  return ((void*)L->Owner.getValue());
}

void *GaloisRuntime::do_getValue(GaloisRuntime::Lockable* L) {
  SimpleRuntimeContext cnx;
  return cnx.do_getValue(L);
}

bool GaloisRuntime::SimpleRuntimeContext::do_trylock(GaloisRuntime::Lockable* L) {
  return L->Owner.try_lock();
}

bool GaloisRuntime::do_trylock(GaloisRuntime::Lockable* L) {
  SimpleRuntimeContext cnx;
  return cnx.do_trylock(L);
}

void GaloisRuntime::SimpleRuntimeContext::do_unlock(GaloisRuntime::Lockable* L) {
  L->Owner.unlock_and_clear();
}

void GaloisRuntime::do_unlock(GaloisRuntime::Lockable* C) {
  SimpleRuntimeContext cnx;
  cnx.do_unlock(C);
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

  // XXX not_ready = false;

  return numLocks;
}

void GaloisRuntime::breakLoop() {
#if GALOIS_USE_EXCEPTION_HANDLER
  throw GaloisRuntime::BREAK;
#else
  longjmp(hackjmp, GaloisRuntime::BREAK);
#endif
}

void GaloisRuntime::signalConflict() {
#if GALOIS_USE_EXCEPTION_HANDLER
        throw GaloisRuntime::CONFLICT; // Conflict
#else
        longjmp(hackjmp, GaloisRuntime::CONFLICT);
#endif
}

void GaloisRuntime::SimpleRuntimeContext::acquire(GaloisRuntime::Lockable* L) {
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
      GaloisRuntime::signalConflict();
    }
  }
}

void GaloisRuntime::SimpleRuntimeContext::sub_acquire(GaloisRuntime::Lockable* L) {
  assert(0 && "Shouldn't get here");
  abort();
}

//anchor vtable
GaloisRuntime::SimpleRuntimeContext::~SimpleRuntimeContext() {}

void GaloisRuntime::DeterministicRuntimeContext::sub_acquire(GaloisRuntime::Lockable* L) {
  // Normal path
  if (pendingFlag.flag.data == NON_DET) {
    GaloisRuntime::SimpleRuntimeContext::acquire(L);
    return;
  }

  // Ordered and deterministic path
  if (pendingFlag.flag.data == COMMITTING)
    return;

  if (L->Owner.try_lock()) {
    assert(!L->next);
    L->next = locks;
    locks = L;
  }

  DeterministicRuntimeContext* other;
  do {
    other = static_cast<DeterministicRuntimeContext*>(L->Owner.getValue());
    if (other == this)
      return;
    if (other) {
      bool conflict = comp ? comp->compare(other->data.comp_data, data.comp_data) :  other->data.id < data.id;
      if (conflict) {
        // A lock that I want but can't get
        not_ready = 1;
        return; 
      }
    }
  } while (!L->Owner.stealing_CAS(other, this));

  // Disable loser
  if (other)
    other->not_ready = 1; // Only need atomic write

  return;
}


void GaloisRuntime::forceAbort() {
  signalConflict();
}
