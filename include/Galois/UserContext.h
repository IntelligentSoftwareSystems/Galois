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

#include <vector>

#include "Galois/Mem.h"

namespace GaloisRuntime {
template <typename WorkListTy, typename T, typename FunctionTy, bool isSimple>
class ForEachWork;

template <typename WorkListTy, typename FunctionTy> 
class ParaMeterExecutor;
}

namespace Galois {

template<typename T>
class UserContext: private boost::noncopyable {
  template <typename WorkListTy, typename TT, typename FunctionTy, bool isSimple>
  friend class GaloisRuntime::ForEachWork;

  template <typename WorkListTy, typename FunctionTy>
  friend class GaloisRuntime::ParaMeterExecutor;

  //! Allocator stuff
  IterAllocBaseTy IterationAllocatorBase;
  PerIterAllocTy PerIterationAllocator;

  void __resetAlloc() {
    IterationAllocatorBase.clear();
  }

  //! break stuff
  int breakFlag;
  bool __breakHappened() {
    return breakFlag;
  }

  void __resetBreak() {
    breakFlag = 0;
  }

  //! push stuff
  typedef std::vector<T> pushBufferTy;
  pushBufferTy pushBuffer;

  pushBufferTy& __getPushBuffer() {
    return pushBuffer;
  }
  
  void __resetPushBuffer() {
    pushBuffer.clear();
  }

public:
  UserContext()
    :IterationAllocatorBase(), 
     PerIterationAllocator(&IterationAllocatorBase),
     breakFlag(0)
  { }

  //! Signal break in parallel loop
  void breakLoop() {
    breakFlag = 1;
  }

  PerIterAllocTy& getPerIterAlloc() {
    return PerIterationAllocator;
  }

  //! Push new work 
  void push(T val) {
    pushBuffer.push_back(val);
  }
};

}
#endif
