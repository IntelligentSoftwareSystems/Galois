/** User Facing loop api -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_USERCONTEXT_H
#define GALOIS_USERCONTEXT_H

#include "Galois/Mem.h"
#include "Galois/gdeque.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/MethodFlags.h"

namespace Galois {

/** 
 * This is the object passed to the user's parallel loop.  This
 * provides the in-loop api.
 */
template<typename T>
class UserContext: private boost::noncopyable {
protected:
  //! Allocator stuff
  IterAllocBaseTy IterationAllocatorBase;
  PerIterAllocTy PerIterationAllocator;

  void __resetAlloc() {
    IterationAllocatorBase.clear();
  }

  //! push stuff
  typedef gdeque<T> PushBufferTy;
  PushBufferTy pushBuffer;

  PushBufferTy& __getPushBuffer() {
    return pushBuffer;
  }
  
  void __resetPushBuffer() {
    pushBuffer.clear();
  }

  void* localState;
  bool localStateUsed;
  void __setLocalState(void *p, bool used) {
    localState = p;
    localStateUsed = used;
  }

public:
  UserContext()
    :IterationAllocatorBase(), 
     PerIterationAllocator(&IterationAllocatorBase)
  { }

  //! Signal break in parallel loop
  void breakLoop() {
    GaloisRuntime::breakLoop();
  }

  //! Acquire a per-iteration allocator
  PerIterAllocTy& getPerIterAlloc() {
    return PerIterationAllocator;
  }

  //! Push new work 
  void push(T val) {
    GaloisRuntime::checkWrite(Galois::WRITE);
    pushBuffer.push_back(val);
  }

  //! force the abort of this iteration
  void abort() { GaloisRuntime::forceAbort(); }

  //! Store and retrieve local state for deterministic and ordered executor
  void* getLocalState(bool& used) { used = localStateUsed; return localState; }
};

}

#endif
