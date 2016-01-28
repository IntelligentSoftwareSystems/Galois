/** Single source shortest paths -*- C++ -*-
 * @file
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
 *
 * @section Description
 *
 * Single source shortest paths, push based.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/Graphs/LCGraph.h"

#include <iostream>

//! [Define LC Graph]
typedef Galois::Graph::LC_Linear_Graph<unsigned int, unsigned int> Graph;
//! [Define LC Graph]
typedef Graph::GraphNode GNode;
typedef std::pair<unsigned, GNode> UpdateRequest;

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max();

unsigned stepShift = 11;

//! [Operator in SSSPPushSimple]
struct SSSP {
  Graph& graph;
  SSSP(Graph& _g) :graph(_g) {}

  void operator()(GNode active_node, Galois::UserContext<GNode>& ctx) const {
    //![Get the value on the node]
    unsigned data = graph.getData(active_node);

    //![loop over neighbors to compute new value]
    for (auto ii : graph.edges(active_node)) {
      GNode dst = graph.getEdgeDst(ii);
      if (graph.getData(dst) > graph.getEdgeData(ii) + data) {
        graph.getData(dst) = graph.getEdgeData(ii) + data;
        ctx.push(dst);
      }
    }
  }
};
//! [Operator in SSSPPushSimple]

//! [Priority Function in SSSPPushSimple]
struct UpdateRequestIndexer {
  Graph& g;
  UpdateRequestIndexer(Graph& _g) :g(_g) {}

  unsigned int operator() (const GNode N) const {
    return g.getData(N, Galois::MethodFlag::UNPROTECTED) >> stepShift;
  }
};


int main(int argc, char **argv) {
  Galois::StatManager statManager;
  Galois::setActiveThreads(256); // Galois will cap at hw max

  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " filename\n";
    return 1;
  } else {
    std::cout << "Note: This is just a very simple example and provides no useful information for performance\n";
  }

//! [ReadGraph]
  Graph graph;
  Galois::Graph::readGraph(graph, argv[1]);
//! [ReadGraph]

  //! Use a lambda as the operator
  Galois::do_all(graph.begin(), graph.end(), [&graph] (GNode& N) { graph.getData(N) = DIST_INFINITY; });

  using namespace Galois::WorkList;
  typedef dChunkedLIFO<16> dChunk;
  typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;
//! [OrderedByIntegerMetic in SSSPPushSimple]

  Galois::StatTimer T;
  T.start();
  //clear source
  graph.getData(*graph.begin()) = 0;
  //! [for_each in SSSPPushSimple]
  std::array<GNode,1> init = {*graph.begin()};
  //!use a structure as an operator and pass a loopname for stats
  Galois::for_each(init.begin(), init.end(), SSSP{graph}, Galois::wl<OBIM>(UpdateRequestIndexer{graph}), Galois::loopname("sssp_run_loop"));
  //! [for_each in SSSPPullsimple]
  T.stop();
  return 0;
}
