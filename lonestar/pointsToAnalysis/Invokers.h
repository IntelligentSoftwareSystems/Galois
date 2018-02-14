/** Invokers -*- C++ -*-
 * @file
 *
 * Functors for different loops available in Galois. 
 *
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef _GALOIS_INVOKERS_
#define _GALOIS_INVOKERS_

#include <galois/Galois.h>

namespace galois {

/**
 * Functor to invoke C++'s for each construct from a Galois-like argument
 * interface.
 */
struct InvokeStdForEach {
  template <typename I, typename F, typename... Args>
  void operator() (const I& iter, const F& f, Args&&... args) {
    // creates the range object from iter
    auto ranges = iter(std::make_tuple(args...)); // args are a dummy variable
    std::for_each(ranges.begin(), ranges.end(), f);
  }
};

/**
 * Functor to invoke Galois's doall construct.
 */
struct InvokeDoAll {
  template <typename I, typename F, typename... Args>
  void operator() (const I& iter, const F& f, Args&&... args) {
    galois::do_all(iter, f, args...);
  }
};

} // end galois namespace

#endif
