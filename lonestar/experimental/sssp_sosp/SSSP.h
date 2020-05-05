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

#ifndef APPS_SSSP_SSSP_H
#define APPS_SSSP_SSSP_H

#include "llvm/Support/CommandLine.h"

#include <limits>
#include <string>
#include <sstream>
#include <stdint.h>

typedef unsigned int Dist;
static const Dist DIST_INFINITY = std::numeric_limits<Dist>::max() - 1;

template <typename GrNode>
struct UpdateRequestCommon {
  GrNode n;
  Dist w;

  UpdateRequestCommon(const GrNode& N, Dist W) : n(N), w(W) {}

  UpdateRequestCommon() : n(), w(0) {}

  bool operator>(const UpdateRequestCommon& rhs) const {
    if (w > rhs.w)
      return true;
    if (w < rhs.w)
      return false;
    return n > rhs.n;
  }

  bool operator<(const UpdateRequestCommon& rhs) const {
    if (w < rhs.w)
      return true;
    if (w > rhs.w)
      return false;
    return n < rhs.n;
  }

  bool operator!=(const UpdateRequestCommon& other) const {
    if (w != other.w)
      return true;
    return n != other.n;
  }

  uintptr_t getID() const { return reinterpret_cast<uintptr_t>(n); }
};

struct SNode {
  std::atomic<Dist> dist;
};

template <typename Graph>
void readInOutGraph(Graph& graph);

extern llvm::cl::opt<unsigned int> memoryLimit;

#endif
