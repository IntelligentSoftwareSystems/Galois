/** Galois Compiler Helper Functions-*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */
#ifndef GALOIS_RUNTIME_COMPILER_HELPER_FUNCTIONS_H
#define GALOIS_RUNTIME_COMPILER_HELPER_FUNCTIONS_H
#include <atomic>
#include <algorithm>
#include <vector>
namespace galois {
  template<typename... Args>
  int read_set(Args... args) {
    // Nothing for now.
    return 0;
  }

  template<typename... Args>
  int write_set(Args... args) {
    // Nothing for now.
    return 0;
  }
} // end namespace galois
#endif
