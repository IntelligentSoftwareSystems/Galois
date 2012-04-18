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

template<typename WLTy, typename FunctionTy>
static inline void for_each_wl(WLTy& wl, FunctionTy f, const char* loopname = 0) {
  assert(!inGaloisForEach);

  inGaloisForEach = true;

  typedef typename WLTy::value_type T;

  typedef ForEachWork<WLTy,T,FunctionTy,false> WorkTy;

  WorkTy W(wl, f, loopname);
  RunCommand w[2];

  w[0].work = Config::ref(W);
  w[0].isParallel = true;
  w[0].barrierAfter = true;
  w[0].profile = true;
  w[1].work = &runAllLoopExitHandlers;
  w[1].isParallel = false;
  w[1].barrierAfter = true;
  w[1].profile = true;
  getSystemThreadPool().run(&w[0], &w[2]);

  inGaloisForEach = false;
}


}

#endif //  GALOIS_GALOISUNSAFE_H
