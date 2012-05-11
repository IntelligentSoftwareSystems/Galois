/** Spanning-tree application -*- C++ -*-
 * @file
 *
 * A simple spanning tree algorithm to demostrate the Galois system.
 *
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "boost/optional.hpp"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

namespace cll = llvm::cl;

const char* name = "Spanning-tree Algorithm";
const char* desc = "Compute the spanning tree (not mimimal) of a graph";
const char* url = NULL;

static cll::opt<int> rootNode(cll::Positional, cll::desc("<root id>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

struct Node {
  bool in_mst;
  Node() : in_mst(false) { }
};

typedef Galois::Graph::LC_Linear_Graph<Node,void> Graph;
typedef Graph::GraphNode GNode;
typedef std::pair<GNode,GNode> Edge;

Graph graph;

struct Process {
  Galois::InsertBag<Edge>& result;
  Process(Galois::InsertBag<Edge>& _result): result(_result) { }

  void operator()(GNode src, Galois::UserContext<GNode>& ctx) {
    for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::ALL),
        ei = graph.edge_end(src, Galois::ALL); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, Galois::ALL);
      if (data.in_mst)
        continue;
      ctx.push(dst);
      result.push(Edge(src, dst));
      data.in_mst = true;
    }
  }
};

struct not_in_mst {
  bool operator()(const GNode& n) const {
    return !graph.getData(n).in_mst;
  }
};

bool verify(Galois::InsertBag<Edge>& result) {
  if (std::distance(result.begin(), result.end()) != graph.size() - 1)
    return false;

  return Galois::find_if(graph.begin(), graph.end(), not_in_mst()) == graph.end();
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

  boost::optional<GNode> root;
  graph.structureFromFile(filename.c_str());
  int n = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii, n++) {
    if (n == rootNode) {
      root = boost::optional<GNode>(*ii);
      break;
    }
  }
  if (!root) {
    std::cerr << "Unknown root node\n";
    abort();
  }

  Galois::InsertBag<Edge> result;

  Galois::StatTimer T;
  T.start();
  graph.getData(*root).in_mst = true;
  Galois::for_each(*root, Process(result));
  T.stop();

  std::cout << "Edges in spanning tree: " 
    << std::distance(result.begin(), result.end()) << "\n";

  if (!skipVerify && !verify(result)) {
    std::cerr << "If graph was connected, verification failed\n";
    assert(0 && "If graph was connected, verification failed");
    abort();
  }

  return 0;
}

