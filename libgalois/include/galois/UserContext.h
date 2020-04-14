/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_USERCONTEXT_H
#define GALOIS_USERCONTEXT_H

#include "galois/Mem.h"
#include "galois/gdeque.h"

#include "galois/runtime/Context.h"

#include <functional>

namespace galois {

/**
 * This is the object passed to the user's parallel loop.  This
 * provides the in-loop api.
 */
template <typename T>
class UserContext : private boost::noncopyable {
protected:
  //! push stuff
  typedef gdeque<T> PushBufferTy;
  static const unsigned int fastPushBackLimit = 64;
  typedef std::function<void(PushBufferTy&)> FastPushBack;

  PushBufferTy pushBuffer;
  //! Allocator stuff
  IterAllocBaseTy IterationAllocatorBase;
  PerIterAllocTy PerIterationAllocator;

  //! used by all
  bool* didBreak = nullptr;
  FastPushBack fastPushBack;

  //! some flags used by deterministic
  bool firstPassFlag = false;
  void* localState   = nullptr;

  void __resetAlloc() { IterationAllocatorBase.clear(); }

  void __setFirstPass(void) { firstPassFlag = true; }

  void __resetFirstPass(void) { firstPassFlag = false; }

  PushBufferTy& __getPushBuffer() { return pushBuffer; }

  void __resetPushBuffer() { pushBuffer.clear(); }

  void __setLocalState(void* p) { localState = p; }

  void __setFastPushBack(FastPushBack f) { fastPushBack = f; }

public:
  UserContext()
      : IterationAllocatorBase(),
        PerIterationAllocator(&IterationAllocatorBase), didBreak(0) {}

  //! Signal break in parallel loop, current iteration continues
  //! untill natural termination
  void breakLoop() { *didBreak = true; }

  //! Acquire a per-iteration allocator
  PerIterAllocTy& getPerIterAlloc() { return PerIterationAllocator; }

  //! Push new work
  template <typename... Args>
  void push(Args&&... args) {
    // galois::runtime::checkWrite(MethodFlag::WRITE, true);
    pushBuffer.emplace_back(std::forward<Args>(args)...);
    if (fastPushBack && pushBuffer.size() > fastPushBackLimit)
      fastPushBack(pushBuffer);
  }

  //! Push new work
  template <typename... Args>
  inline void push_back(Args&&... args) {
    this->push(std::forward<Args>(args)...);
  }

  //! Push new work
  template <typename... Args>
  inline void insert(Args&&... args) {
    this->push(std::forward<Args>(args)...);
  }

  //! Force the abort of this iteration
  void abort() { galois::runtime::signalConflict(); }

  //! Store and retrieve local state for deterministic
  template <typename LS>
  LS* getLocalState(void) {
    return reinterpret_cast<LS*>(localState);
  }

  template <typename LS, typename... Args>
  LS* createLocalState(Args&&... args) {
    new (localState) LS(std::forward<Args>(args)...);
    return getLocalState<LS>();
  }

  //! used by deterministic and ordered
  //! @returns true when the operator is invoked for the first time. The
  //! operator can use this information and choose to expand the neighborhood
  //! only in the first pass.
  bool isFirstPass(void) const { return firstPassFlag; }

  //! declare that the operator has crossed the cautious point.  This
  //! implies all data has been touched thus no new locks will be
  //! acquired.
  void cautiousPoint() {
    if (isFirstPass()) {
      galois::runtime::signalFailSafe();
    }
  }
};

} // namespace galois

#endif
