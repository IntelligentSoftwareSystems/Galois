/** Highest out degree finder -*- C++ -*-
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
 * @section Description
 * 
 * Finds the node with the highest outdegree in a graph.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

/**
 * Dummy node data struct.
 */
struct NodeData {
  char dummy;
};

typedef galois::graphs::DistGraph<NodeData, void> Graph;
typedef galois::graphs::DistGraph_edgeCut<NodeData, void> Graph_edgeCut;

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "FindHighDegree";
constexpr static const char* const desc = "Find highest owned outdegree node.";
constexpr static const char* const url = 0;

/**
 * Partitions the graph with an outgoing edges cut, then loops through owned 
 * nodes to determine which node has the highest out-degree. Prints
 * it to a line.
 */
int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  const auto& net = galois::runtime::getSystemNetworkInterface();
  std::vector<unsigned> dummyScale;

  partitionScheme = galois::graphs::PARTITIONING_SCHEME::OEC;

  Graph* regular = new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                                     dummyScale, false);

  uint32_t curMaxNode = 0;
  uint64_t curHighEdges = 0;

  for (auto i = regular->masterNodesRange().begin();
       i < regular->masterNodesRange().end();
       i++) {
    uint64_t numEdges = 
       std::distance(regular->edge_begin(*i), regular->edge_end(*i));
   
    if (numEdges > curHighEdges) {
      curHighEdges = numEdges;
      curMaxNode = *i;
    }
  }

  galois::gPrint(regular->L2G(curMaxNode), " ", curHighEdges, "\n");
  return 0;
}
