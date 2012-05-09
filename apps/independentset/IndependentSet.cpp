/** Maximal independent set application -*- C++ -*-
 * @file
 *
 * A simple spanning tree algorithm to demostrate the Galois system.
 *
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
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

const char* name = "Maximal Independent Set";
const char* desc = "Compute a maximal independent set (not maximum) of nodes in a graph";
const char* url = NULL;

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

enum MatchFlag {
  UNMATCHED, OTHER_MATCHED, MATCHED
};

struct Node {
  MatchFlag flag; 
  Node() : flag(UNMATCHED) { }
};

typedef Galois::Graph::LC_Linear_Graph<Node,void> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

struct Process {
  typedef int tt_does_not_need_parallel_push;

  void operator()(GNode src, Galois::UserContext<GNode>& ctx) {
    for (Graph::edge_iterator ii = graph.edge_begin(src),
        ei = graph.edge_end(src); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      if (data.flag != UNMATCHED)
        return;
    }

    Node& me = graph.getData(src, Galois::WRITE);
    if (me.flag != UNMATCHED)
      return;

    for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::NONE),
        ei = graph.edge_end(src, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, Galois::NONE);
      data.flag = OTHER_MATCHED;
    }

    me.flag = MATCHED;
  }
};

struct is_bad {
  bool operator()(GNode n) const {
    Node& me = graph.getData(n);
    if (me.flag == MATCHED) {
      for (Graph::edge_iterator ii = graph.edge_begin(n),
          ei = graph.edge_end(n); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& data = graph.getData(dst);
        if (dst != n && data.flag == MATCHED) {
          std::cerr << "double match\n";
          return true;
        }
      }
    } else if (me.flag == UNMATCHED) {
      bool ok = false;
      for (Graph::edge_iterator ii = graph.edge_begin(n),
          ei = graph.edge_end(n); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& data = graph.getData(dst);
        if (data.flag != UNMATCHED) {
          ok = true;
        }
      }
      if (!ok) {
        std::cerr << "not maximal\n";
        return true;
      }
    }
    return false;
  }
};

struct is_matched {
  bool operator()(const GNode& n) const {
    return graph.getData(n).flag == MATCHED;
  }
};

bool verify() {
  return Galois::find_if(graph.begin(), graph.end(), is_bad()) == graph.end();
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

  graph.structureFromFile(filename.c_str());

  Galois::StatTimer T;
  T.start();
  Galois::for_each(graph.begin(), graph.end(), Process());
  T.stop();

  std::cout << "Cardinality of maximal independent set: " 
    << Galois::count_if(graph.begin(), graph.end(), is_matched()) 
    << "\n";

  if (!skipVerify && !verify()) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  return 0;
}

