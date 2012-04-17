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

#ifndef GALOIS_GALOIS_UNSAFE_H
#define GALOIS_GALOIS_UNSAFE_H

#include "Galois/Galois.h"

namespace Galois {

// WorkList based version
template <ForeachOpts::ExecutorMode mode_tp, typename WLTy, typename Function>
static inline void for_each_wl (WLTy& wl, Function f, const char* loopname = 0) {

  if (mode_tp == ForeachOpts::PARAMETER) {
    GaloisRuntime::ParaMeter::for_each_impl (wl, f, loopname);
  } else {
    GaloisRuntime::for_each_impl (wl, f, loopname);
  }
}

template <typename WLTy, typename Function>
static inline void for_each_wl (WLTy& wl, Function f, const char* loopname = 0) {

  Galois::for_each_wl<ForeachOpts::GALOIS> (wl, f, loopname);
}


}

#endif //  GALOIS_GALOIS_UNSAFE_H
