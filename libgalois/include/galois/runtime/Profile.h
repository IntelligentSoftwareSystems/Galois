/** HW Runtime Sampling Control -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_SAMPLING_H
#define GALOIS_RUNTIME_SAMPLING_H

#include "galois/Timer.h"
#include "galois/gIO.h"

namespace galois {
namespace runtime {

#ifdef GALOIS_USE_VTUNE
#include "ittnotify.h"

template <typename F>
void profileVtune(const F& func, const char* region) {

  region = region ? region : "(NULL)";

  GALOIS_ASSERT(galois::substrate::ThreadPool::getTID() == 0
      , "profileVtune can only be called from master thread (thread 0)");

  __itt_resume();

  timeThis(func, region);

  __itt_pause();

}

#else

template <typename F>
void profileVtune(const F& func, const char* region) {

  region = region ? region : "(NULL)";
  galois::gWarn("Vtune not enabled or found");

  timeThis(func, region);
}

#endif


}
} // end namespace galois

#endif
