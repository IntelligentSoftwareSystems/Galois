/** simple galois context and contention manager -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/Context.h"
#include "Galois/Substrate/SimpleLock.h"
#include "Galois/Substrate/CacheLineStorage.h"

#include <stdio.h>

//! Global thread context for each active thread
static __thread galois::Runtime::SimpleRuntimeContext* thread_ctx = 0;


void galois::Runtime::signalFailSafe(void) {
  throw galois::Runtime::REACHED_FAILSAFE;
}

void galois::Runtime::setThreadContext(galois::Runtime::SimpleRuntimeContext* ctx) {
  thread_ctx = ctx;
}

galois::Runtime::SimpleRuntimeContext* galois::Runtime::getThreadContext() {
  return thread_ctx;
}

void galois::Runtime::signalConflict(Lockable* lockable) {
  throw galois::Runtime::CONFLICT; // Conflict
}

#ifdef GALOIS_USE_EXP
bool galois::Runtime::owns(Lockable* lockable, MethodFlag m) {
  SimpleRuntimeContext* ctx = getThreadContext();
  if (ctx) {
    return ctx->owns(lockable, m);
  } 
  return false;
}
#endif


////////////////////////////////////////////////////////////////////////////////
// LockManagerBase & SimpleRuntimeContext
////////////////////////////////////////////////////////////////////////////////

galois::Runtime::LockManagerBase::AcquireStatus
galois::Runtime::LockManagerBase::tryAcquire(galois::Runtime::Lockable* lockable) 
{
  assert(lockable);
  // XXX(ddn): Hand inlining this code makes a difference on 
  // delaunaytriangulation (GCC 4.7.2)
#if 0
  if (tryLock(lockable)) {
    assert(!getOwner(lockable));
    setOwner(lockable);
    return NEW_OWNER;
#else
  if (lockable->owner.try_lock()) {
    lockable->owner.setValue(this);
    return NEW_OWNER;
#endif
  } else if (getOwner(lockable) == this) {
    return ALREADY_OWNER;
  }
  return FAIL;
}

void galois::Runtime::SimpleRuntimeContext::acquire(galois::Runtime::Lockable* lockable, galois::MethodFlag m) {
  AcquireStatus i;
  if (customAcquire) {
    subAcquire(lockable, m);
  } else if ((i = tryAcquire(lockable)) != AcquireStatus::FAIL) {
    if (i == AcquireStatus::NEW_OWNER) {
      addToNhood(lockable);
    }
  } else {
    galois::Runtime::signalConflict(lockable);
  }
}

void galois::Runtime::SimpleRuntimeContext::release(galois::Runtime::Lockable* lockable) {
  assert(lockable);
  // The deterministic executor, for instance, steals locks from other
  // iterations
  assert(customAcquire || getOwner(lockable) == this);
  assert(!lockable->next);
  lockable->owner.unlock_and_clear();
}

unsigned galois::Runtime::SimpleRuntimeContext::commitIteration() {
  unsigned numLocks = 0;
  while (locks) {
    //ORDER MATTERS!
    Lockable* lockable = locks;
    locks = lockable->next;
    lockable->next = 0;
    Substrate::compilerBarrier();
    release(lockable);
    ++numLocks;
  }

  return numLocks;
}

unsigned galois::Runtime::SimpleRuntimeContext::cancelIteration() {
  return commitIteration();
}

void galois::Runtime::SimpleRuntimeContext::subAcquire(galois::Runtime::Lockable* lockable, galois::MethodFlag) {
  GALOIS_DIE("Shouldn't get here");
}

#ifdef GALOIS_USE_EXP
bool galois::Runtime::SimpleRuntimeContext::owns(galois::Runtime::Lockable* lockable, galois::MethodFlag) const {
  GALOIS_DIE("SimpleRuntimeContext::owns Not Implemented");
}
#endif
