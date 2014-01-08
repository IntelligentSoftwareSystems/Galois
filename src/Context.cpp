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

using namespace Galois::Runtime;

//! Global thread context for each active thread
static __thread SimpleRuntimeContext* thread_ctx = 0;

void Galois::Runtime::doCheckWrite() {
  // if (getPending () == PENDING) {
  //   throw failsafe_ex();
  // }
}

void Galois::Runtime::setThreadContext(SimpleRuntimeContext* ctx) {
  thread_ctx = ctx;
}

SimpleRuntimeContext* Galois::Runtime::getThreadContext() {
  return thread_ctx;
}

void Galois::Runtime::signalConflict(Lockable* lockable) {
  throw conflict_ex{lockable}; // Conflict
}

void Galois::Runtime::forceAbort() {
  throw conflict_ex{nullptr}; // Conflict
}

////////////////////////////////////////////////////////////////////////////////
// SimpleRuntimeContext
////////////////////////////////////////////////////////////////////////////////

bool SimpleRuntimeContext::acquire(Lockable* lockable) {
  if (customAcquire)
    return subAcquire(lockable);
  else
    return (locks.tryAcquire(lockable) != LockManagerBase::FAIL);
}

void SimpleRuntimeContext::release(Lockable* lockable) {
  assert(lockable);
  // The deterministic executor, for instance, steals locks from other
  // iterations
  assert(customAcquire);// || getOwner(lockable) == this);
  //assert(!lockable->next);
  locks.releaseOne(lockable);
}

unsigned SimpleRuntimeContext::commitIteration() {
  if (customAcquire)
    return subCommit();
  else
    return locks.releaseAll();
}

unsigned SimpleRuntimeContext::cancelIteration() { 
  if (customAcquire)
    return subCancel();
  else
    return commitIteration();
}

bool SimpleRuntimeContext::subAcquire(Lockable* lockable) {
  GALOIS_DIE("Shouldn't get here");
}

unsigned SimpleRuntimeContext::subCommit() {
  GALOIS_DIE("Shouldn't get here");
}

unsigned SimpleRuntimeContext::subCancel() {
  GALOIS_DIE("Shouldn't get here");
}


