/** User Facing loop api -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * User-facing API for inside a loop
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_USERCONTEXT_H
#define GALOIS_USERCONTEXT_H

#include "Galois/Mem.h"
#include "Galois/Runtime/gdeque.h"

#include "Galois/Runtime/SyncContext.h"

#include <functional>

namespace Galois {

/** 
 * This is the object passed to the user's parallel loop.  This
 * provides the in-loop api.
 */
template<typename T>
class UserContext: private boost::noncopyable {
protected:
// TODO(ddn): move to a separate class for dedicated speculative executors
#ifdef GALOIS_USE_EXP
  typedef std::function<void (void)> Closure;
  typedef Runtime::gdeque<Closure, 8> UndoLog;
  typedef UndoLog CommitLog;

  UndoLog undoLog;
  CommitLog commitLog;
#endif 
  //! push stuff
  typedef Runtime::gdeque<T> PushBufferTy;
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
  void* localState = nullptr;

  void __resetAlloc() {
    IterationAllocatorBase.clear();
  }

  void __setFirstPass (void) {
    firstPassFlag = true;
  }

  void __resetFirstPass (void) {
    firstPassFlag = false;
  }

#ifdef GALOIS_USE_EXP
  void __rollback() {
    for (auto ii = undoLog.begin(), ei = undoLog.end(); ii != ei; ++ii) {
      (*ii)();
    }
  }

  void __commit() {
    for (auto ii = commitLog.begin(), ei = commitLog.end(); ii != ei; ++ii) {
      (*ii)();
    }
  }

  void __resetUndoLog() {
    undoLog.clear();
  }

  void __resetCommitLog() {
    commitLog.clear();
  }
#endif 


  PushBufferTy& __getPushBuffer() {
    return pushBuffer;
  }
  
  void __resetPushBuffer() {
    pushBuffer.clear();
  }

  void __setLocalState(void *p) {
    localState = p;
  }

  void __setFastPushBack(FastPushBack f) {
    fastPushBack = f;
  }


public:
  UserContext()
    :IterationAllocatorBase(), 
     PerIterationAllocator(&IterationAllocatorBase),
     didBreak(0)
  { }

  //! Signal break in parallel loop, current iteration continues
  //! untill natural termination
  void breakLoop() {
    *didBreak = true;
  }

  //! Acquire a per-iteration allocator
  PerIterAllocTy& getPerIterAlloc() {
    return PerIterationAllocator;
  }

  //! Push new work 
  template<typename... Args>
  void push(Args&&... args) {
    // Galois::Runtime::checkWrite(MethodFlag::WRITE, true);
    pushBuffer.emplace_back(std::forward<Args>(args)...);
    if (fastPushBack && pushBuffer.size() > fastPushBackLimit)
      fastPushBack(pushBuffer);
  }

  //! Force the abort of this iteration
  void abort() { Galois::Runtime::signalConflict(); }

  //! Store and retrieve local state for deterministic
  void* getLocalState(void) { return localState; }
 
#ifdef GALOIS_USE_EXP
  void addUndoAction(const Closure& f) {
    undoLog.push_front(f);
  }

  void addCommitAction(const Closure& f) {
    commitLog.push_back(f);
  }
#endif 

  //! used by deterministic and ordered
  //! @returns true when the operator is invoked for the first time. The operator
  //! can use this information and choose to expand the neighborhood only in the first pass. 
  bool isFirstPass (void) const { 
    return firstPassFlag;
  }

  //! declare that the operator has crossed the cautious point.  This
  //! implies all data has been touched thus no new locks will be
  //! acquired.
  void cautiousPoint() {
    if (isFirstPass()) {
      Galois::Runtime::signalFailSafe();
    }
  }

};

}

#endif
