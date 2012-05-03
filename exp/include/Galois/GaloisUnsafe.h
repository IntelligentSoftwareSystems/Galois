/** Galois user interface -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
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
 */

#ifndef GALOIS_GALOISUNSAFE_H
#define GALOIS_GALOISUNSAFE_H

#include "Galois/Galois.h"

namespace Galois {

namespace hidden {

  template <typename ExecutorTy>
  static inline void for_each_wl_impl (ExecutorTy& exec, const bool isParallel) {
    assert(!GaloisRuntime::inGaloisForEach);

    GaloisRuntime::inGaloisForEach = true;

    GaloisRuntime::RunCommand w[2];

    // disabling profiling because this loop is invoked multiple times typically
    // and that crashes vtune
    w[0].work = GaloisRuntime::Config::ref(exec);
    w[0].isParallel = isParallel;
    w[0].barrierAfter = true;
    w[0].profile = false;
    w[1].work = &GaloisRuntime::runAllLoopExitHandlers;
    w[1].isParallel = false;
    w[1].barrierAfter = true;
    w[1].profile = false;
    GaloisRuntime::getSystemThreadPool().run(&w[0], &w[2]);

    GaloisRuntime::inGaloisForEach = false;
  }
}

// XXX: the basic idea of treating executor type as a WorkList e.g. WorkList::ParaMeter
// is broken
// However, following should work:
// When using ParaMeter say: for_each_wl <ParaMeter<WLTy> > (wl, ...)
// and when usin galois say: for_each_wl (wl,...)
template <typename ExecTy, typename WLTy, typename FunctionTy>
static inline void for_each_wl (WLTy& wl, FunctionTy f, const char* loopname=0) {

  typedef typename WLTy::value_type T;

  typedef GaloisRuntime::ForEachWork<ExecTy, T, FunctionTy> WorkTy;

  WorkTy W (wl, f, loopname);

  hidden::for_each_wl_impl (W, false);
 
}

template <typename WLTy, typename FunctionTy>
static inline void for_each_wl (WLTy& wl, FunctionTy f, const char* loopname=0) {

  typedef typename WLTy::value_type T;

  typedef GaloisRuntime::ForEachWork<WLTy, T, FunctionTy> WorkTy;

  WorkTy W (wl, f, loopname);

  hidden::for_each_wl_impl (W, true);
}





}

#endif //  GALOIS_GALOISUNSAFE_H

