/** Galois Simple Function Executor -*- C++ -*-
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
 * Simple wrapper for the thread pool
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_EXECUTOR_ONEACH_H
#define GALOIS_RUNTIME_EXECUTOR_ONEACH_H

#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/ll/gio.h"

#include "Galois/Runtime/ThreadPool.h"

namespace Galois {
namespace Runtime {

extern unsigned int activeThreads;
extern bool inGaloisForEach;

namespace detail {

template<typename FunctionTy>
struct WOnEach {
  const FunctionTy& origFunction;
  explicit WOnEach(const FunctionTy& f): origFunction(f) { }
  void operator()(void) {
    FunctionTy fn(origFunction);
    fn(LL::getTID(), activeThreads);   
  }
};

} // end namespace detail

template<typename FunctionTy>
void on_each_impl(const FunctionTy& fn, const char* loopname = nullptr) {
  if (inGaloisForEach)
    GALOIS_DIE("Nested parallelism not supported");
  
  inGaloisForEach = true;
  getSystemThreadPool().run(activeThreads, detail::WOnEach<FunctionTy>(fn));
  inGaloisForEach = false;
}

} // end namespace Runtime
} // end namespace Galois

#endif

