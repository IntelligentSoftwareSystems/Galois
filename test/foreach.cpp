/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
 */

#include "galois/Galois.h"
#include "galois/Bag.h"
#include <vector>
#include <iostream>

void function_pointer(int x, galois::UserContext<int>& ctx) {
  std::cout << x << "\n";
}

struct function_object {
  void operator()(int x, galois::UserContext<int>& ctx) const {
    function_pointer(x, ctx);
  }
};

int main() {
  galois::SharedMemSys Galois_runtime;
  std::vector<int> v(10);
  galois::InsertBag<int> b;

  galois::for_each(galois::iterate(v), &function_pointer,
                   galois::loopname("func-pointer"));
  galois::for_each(galois::iterate(v), function_object(),
                   galois::loopname("with function object and options"));
  galois::do_all(galois::iterate(v), [&b](int x) { b.push(x); });
  galois::for_each(galois::iterate(b), function_object());

  // Works without context as well
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1400
#else
  // Don't support Context-free versions yet (gcc 4.7 problem)
  //  galois::for_each(v.begin(), v.end(), [](int x) { std::cout << x << "\n";
  //  });
  // galois::for_each(b, [](int x) { std::cout << x << "\n"; });
#endif

  return 0;
}
