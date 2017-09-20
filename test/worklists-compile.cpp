/** Check worklist instantiations -*- C++ -*-
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
#include "galois/runtime/Range.h"

#include <cstdlib>

int run = 1;

template<typename T2>
struct checker {
  typedef typename T2::template retype<int> T;
  T wl;
  typename T::template rethread<true> wl2;
  typename T::template rethread<false> wl3;

  checker() {
    int a[4] = {1,2,3,0};

    // Don't actually run this code as some worklists don't support
    // the full worklist API
    if (run)
      return;

    wl.push(0);
    wl.push_initial(galois::runtime::makeStandardRange(&a[0], &a[4]));
    wl.push(&a[0], &a[4]);
    wl.pop();

    wl2.push(0);
    wl2.push_initial(galois::runtime::makeStandardRange(&a[0], &a[4]));
    wl2.push(&a[0], &a[4]);
    wl2.pop();

    wl3.push(0);
    wl3.push_initial(galois::runtime::makeStandardRange(&a[0], &a[4]));
    wl3.push(&a[0], &a[4]);
    wl3.pop();
  }
};

#define GALOIS_WLCOMPILECHECK(name) checker<name<> > ck_##name;
#include "galois/worklists/WorkList.h"

int main(int argc, char** argv) {
  if (argc > 1)
    run = atoi(argv[1]);

  return 0;
}
