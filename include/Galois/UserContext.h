/** User Facing loop api -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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

  static const unsigned int fastPushBackLimit = 64;

  typedef std::function<void(PushBufferTy&)> FastPushBack; 
  FastPushBack fastPushBack;
  void __setFastPushBack(FastPushBack f) {
    fastPushBack = f;
  }

public:
  UserContext()
    :IterationAllocatorBase(), 
     PerIterationAllocator(&IterationAllocatorBase)
  { }

  //! Signal break in parallel loop
  void breakLoop() {
    Galois::Runtime::breakLoop();
  }

  //! Acquire a per-iteration allocator
  PerIterAllocTy& getPerIterAlloc() {
    return PerIterationAllocator;
  }

  //! Push new work 
  template<typename... Args>
  void push(Args&&... args) {
    Galois::Runtime::checkWrite(MethodFlag::WRITE, true);
    pushBuffer.emplace_back(std::forward<Args>(args)...);
    if (fastPushBack && pushBuffer.size() > fastPushBackLimit)
      fastPushBack(pushBuffer);
  }

  //! Force the abort of this iteration
  void abort() { Galois::Runtime::forceAbort(); }

  //! Store and retrieve local state for deterministic
  void* getLocalState(bool& used) { used = localStateUsed; return localState; }
};

}

#endif
