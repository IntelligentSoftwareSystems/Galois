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

namespace GaloisRuntime {
template<class WorkListTy, class Function>
class ForEachWork;
}

namespace Galois {

template<typename T>
class UserContext {
  template<class WorkListTy, class Function>
  friend class GaloisRuntime::ForEachWork;

  //! Allocator stuff
  ItAllocBaseTy IterationAllocatorBase;
  PerIterAllocTy PerIterationAllocator;

  void __resetAlloc() {
    IterationAllocatorBase.clear();
  }

  //! break stuff
  GaloisRuntime::cache_line_storage<int> breakFlag;
  bool __breakHappened() {
    return breakFlag.data;
  }

  //! push stuff
  typedef std::vector<T> pushBufferTy;
  pushBufferTy pushBuffer;

  pushBufferTy& __getPushBuffer() {
    return pushBuffer;
  }

public:
  UserContext()
    :IterationAllocatorBase(), 
     PerIterationAllocator(&IterationAllocatorBase),
     breakFlag(0)
  {}

  void breakLoop() {
    breakFlag.data = 1;
  }

  PerIterAllocTy& getPerIterAlloc() {
    return PerIterationAllocator;
  }

  void push(T val) {
    pushBuffer.push_back(val);
  }


};

}
#endif
