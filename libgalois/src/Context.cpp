/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/runtime/Context.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/CacheLineStorage.h"

#include <stdio.h>

//! Global thread context for each active thread
static thread_local galois::runtime::SimpleRuntimeContext* thread_ctx = 0;

thread_local jmp_buf galois::runtime::execFrame;

void galois::runtime::setThreadContext(
    galois::runtime::SimpleRuntimeContext* ctx) {
  thread_ctx = ctx;
}

galois::runtime::SimpleRuntimeContext* galois::runtime::getThreadContext() {
  return thread_ctx;
}

////////////////////////////////////////////////////////////////////////////////
// LockManagerBase & SimpleRuntimeContext
////////////////////////////////////////////////////////////////////////////////

galois::runtime::LockManagerBase::AcquireStatus
galois::runtime::LockManagerBase::tryAcquire(
    galois::runtime::Lockable* lockable) {
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

void galois::runtime::SimpleRuntimeContext::release(
    galois::runtime::Lockable* lockable) {
  assert(lockable);
  // The deterministic executor, for instance, steals locks from other
  // iterations
  assert(customAcquire || getOwner(lockable) == this);
  assert(!lockable->next);
  lockable->owner.unlock_and_clear();
}

unsigned galois::runtime::SimpleRuntimeContext::commitIteration() {
  unsigned numLocks = 0;
  while (locks) {
    // ORDER MATTERS!
    Lockable* lockable = locks;
    locks              = lockable->next;
    lockable->next     = 0;
    substrate::compilerBarrier();
    release(lockable);
    ++numLocks;
  }

  return numLocks;
}

unsigned galois::runtime::SimpleRuntimeContext::cancelIteration() {
  return commitIteration();
}

void galois::runtime::SimpleRuntimeContext::subAcquire(
    galois::runtime::Lockable*, galois::MethodFlag) {
  GALOIS_DIE("unreachable");
}
