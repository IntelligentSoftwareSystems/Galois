/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/
#include <cassert>
#include <stdlib.h>

#include "Galois/Runtime/Context.h"

//! Global thread context for each active thread
static __thread GaloisRuntime::SimpleRuntimeContext* thread_cnx = 0;

//! Helper function to decide if the conflict detection lock should be taken
static inline bool shouldLock(Galois::MethodFlag g) {
  switch(g) {
  case Galois::NONE:
  case Galois::SAVE_UNDO:
    return false;
  case Galois::ALL:
  case Galois::CHECK_CONFLICT:
    return true;
  }
  assert(0 && "Shouldn't get here");
  abort();
}


void GaloisRuntime::setThreadContext(GaloisRuntime::SimpleRuntimeContext* n)
{
  thread_cnx = n;
}

GaloisRuntime::SimpleRuntimeContext* GaloisRuntime::getThreadContext() 
{
  return thread_cnx;
}

void GaloisRuntime::doAcquire(GaloisRuntime::Lockable* C) {
  SimpleRuntimeContext* cnx = getThreadContext();
  if (cnx)
    cnx->acquire(C);
}

void GaloisRuntime::SimpleRuntimeContext::cancel_iteration() {
  //FIXME: not handled yet
  commit_iteration();
}

void GaloisRuntime::SimpleRuntimeContext::commit_iteration() {
  while (locks) {
    //ORDER MATTERS!
    //FIXME: compiler optimization barrier
    Lockable* L = locks;
    locks = L->next;
    L->next = 0;
    L->Owner.unlock_and_clear();
  }
}

void GaloisRuntime::SimpleRuntimeContext::acquire(GaloisRuntime::Lockable* L) {
  bool suc = L->Owner.try_lock();
  if (suc) {
    L->Owner.setValue(this);
    L->next = locks;
    locks = L;
  } else {
    if (L->Owner.getValue() != this)
      throw -1; //CONFLICT
  }
}
