/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

// NOTE: must be before DistBenchStart.h as that relies on some cuda
// calls

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

namespace cll = llvm::cl;

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

struct NodeData {
  uint32_t dummy;
};

typedef galois::graphs::DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "Partition";
constexpr static const char* const desc = "Partitions a normal graph.";
constexpr static const char* const url  = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);
  distGraphInitialization<NodeData, void>();
  return 0;
}
