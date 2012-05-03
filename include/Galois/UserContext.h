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
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/MethodFlags.h"

namespace GaloisRuntime {

template<typename T>
class UserContextAccess;

}

namespace Galois {

template<typename T>
class UserContext: private boost::noncopyable {
  template<typename TT>
  friend class GaloisRuntime::UserContextAccess;

  //! Allocator stuff
  IterAllocBaseTy IterationAllocatorBase;
  PerIterAllocTy PerIterationAllocator;

  void __resetAlloc() {
    IterationAllocatorBase.clear();
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
     PerIterationAllocator(&IterationAllocatorBase)
  { }

  //! Signal break in parallel loop
  void breakLoop() {
    throw GaloisRuntime::BREAK;
  }

  PerIterAllocTy& getPerIterAlloc() {
    return PerIterationAllocator;
  }

  //! Push new work 
  void push(T val) {
    GaloisRuntime::checkWrite(Galois::WRITE);
    pushBuffer.push_back(val);
  }
};

}

namespace GaloisRuntime {

//! Backdoor to allow runtime methods to access private data in UserContext
template<typename T>
class UserContextAccess {
  Galois::UserContext<T> ctx;
public:
  typedef typename Galois::UserContext<T>::pushBufferTy pushBufferTy;

  void resetAlloc() { ctx.__resetAlloc(); }
  pushBufferTy& getPushBuffer() { return ctx.__getPushBuffer(); }
  void resetPushBuffer() { ctx.__resetPushBuffer(); }
  Galois::UserContext<T>& data() { return ctx; }
};

}
#endif
