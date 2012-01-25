// simple galois context and contention manager -*- C++ -*-
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

#ifndef _GALOIS_RUNTIME_CONTEXT_H
#define _GALOIS_RUNTIME_CONTEXT_H

#include "Galois/Runtime/ll/PtrLock.h"
#include "Galois/ConflictFlags.h"
#include <cassert>
#include <cstdlib>

namespace GaloisRuntime {

class SimpleRuntimeContext;

//! All objects that may be locked (nodes primarily) must inherit from Lockable.
//! Use an intrusive list to track objects in a context without allocation overhead
class Lockable {
  LL::PtrLock<SimpleRuntimeContext*, true> Owner;
  Lockable* next;
  friend class SimpleRuntimeContext;
public:
  Lockable() :next(0) {}
};

class SimpleRuntimeContext {
  //! The locks we hold
  Lockable* locks;

public:
  SimpleRuntimeContext() :locks(0) {}

  void start_iteration() {
    assert(!locks);
  }
  
  unsigned cancel_iteration();
  unsigned commit_iteration();
  void acquire(Lockable* L);
};

//! get the current conflict detection class, may be null if not in parallel region
SimpleRuntimeContext* getThreadContext();

//! used by the parallel code to set up conflict detection per thread
void setThreadContext(SimpleRuntimeContext* n);

//! used to release the conflict lock
//! gcc exception handling can deadlock :(
void clearConflictLock();

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

void doAcquire(Lockable* C);

//! Master function which handles conflict detection
//! used to acquire a lockable thing
static inline void acquire(Lockable* C, Galois::MethodFlag m) {
  if (shouldLock(m))
    doAcquire(C);
}

}

#endif
