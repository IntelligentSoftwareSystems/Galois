/** -*- C++ -*-
 * @file
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
 * @section Description
 *
 * Prints node ids that have an outgoing edge.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "galois/graphs/OfflineGraph.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

/**
 * Dummy node data struct.
 */
struct NodeData {
  char dummy;
};

typedef galois::graphs::DistGraph<NodeData, unsigned> Graph;
typedef galois::graphs::DistGraphEdgeCut<NodeData, unsigned> Graph_edgeCut;

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "Print Nodes with Edges";
constexpr static const char* const desc = "Print node ids with edges.";
constexpr static const char* const url  = 0;

/**
 * Partitions the graph with an outgoing edges cut, then loops through owned
 * nodes to determine which node has the highest out-degree. Prints
 * it to a line.
 */
int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  galois::graphs::OfflineGraph graph(inputFile);

  uint64_t count = 0;
  for (size_t i = 0; i < graph.size(); i++) {
    uint64_t numEdges = std::distance(graph.edge_begin(i), graph.edge_end(i));
    if (numEdges > 0) {
      galois::gPrint(i, "\n");
      count++;
    }
  }

  galois::gPrint(count, " nodes with edges.\n");
  return 0;
}
