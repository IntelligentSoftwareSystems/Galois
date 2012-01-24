/** Galois user interface -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
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
 */
#ifndef GALOIS_GALOIS_H
#define GALOIS_GALOIS_H

#include "Galois/Threads.h"
#include "Galois/UserContext.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/ParaMeter.h"

namespace Galois {

////////////////////////////////////////////////////////////////////////////////
// Foreach
////////////////////////////////////////////////////////////////////////////////

//Iterator based versions
template<typename WLTy, typename IterTy, typename Function>
static inline void for_each(IterTy b, IterTy e, Function f, const char* loopname = 0) {

  if (GaloisRuntime::usingParaMeter ()) {
    GaloisRuntime::ParaMeter::for_each_impl <WLTy> (b, e, f, loopname);
  } else {
    GaloisRuntime::for_each_impl<WLTy>(b, e, f, loopname);
  }
}

template<typename IterTy, typename Function>
static inline void for_each(IterTy b, IterTy e, Function f, const char* loopname = 0) {
  typedef GaloisRuntime::WorkList::dChunkedFIFO<1024> WLTy;
  for_each<WLTy, IterTy, Function>(b, e, f, loopname);
}

//Single initial item versions
template<typename WLTy, typename InitItemTy, typename Function>
static inline void for_each(InitItemTy i, Function f, const char* loopname = 0) {
  InitItemTy wl[1];
  wl[0] = i;
  for_each<WLTy>(&wl[0], &wl[1], f, loopname);
}

template<typename InitItemTy, typename Function>
static inline void for_each(InitItemTy i, Function f, const char* loopname = 0) {
  typedef GaloisRuntime::WorkList::ChunkedFIFO<256> WLTy;
  for_each<WLTy, InitItemTy, Function>(i, f, loopname);
}


////////////////////////////////////////////////////////////////////////////////
// PreAlloc
////////////////////////////////////////////////////////////////////////////////

struct WPreAlloc {
  int n;
  void operator()(void) {
    GaloisRuntime::MM::pagePreAlloc(n);
  }
};

static inline void preAlloc(int num) {
  WPreAlloc fw;
  int a = GaloisRuntime::getSystemThreadPool().getActiveThreads();
  fw.n = (num + a - 1) / a;
  GaloisRuntime::RunCommand w[1];
  w[0].work = GaloisRuntime::config::ref(fw);
  w[0].isParallel = true;
  w[0].barrierAfter = true;
  GaloisRuntime::getSystemThreadPool().run(&w[0], &w[1]);
}

}
#endif
