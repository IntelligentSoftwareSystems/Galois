/** Spanning-tree application -*- C++ -*-
 * @file
 *
 * A simple spanning tree algorithm to demonstrate the Galois system.
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
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/UnionFind.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <algorithm>
#include <iostream>

namespace cll = llvm::cl;

const char* name = "Spanning Tree Algorithm";
const char* desc = "Compute the spanning forest of a graph";
const char* url = NULL;

enum Algo {
  demo,
  asynchronous
};

static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(demo, "Demonstration algorithm"),
      clEnumVal(asynchronous, "Asynchronous"),
      clEnumValEnd), cll::init(asynchronous));

struct Node: public Galois::UnionFindNode<Node> {
  Node* component;
};

#ifdef GALOIS_USE_NUMA
typedef Galois::Graph::LC_Numa_Graph<Node,void> Graph;
#else
typedef Galois::Graph::LC_CSR_Graph<Node,void> Graph;
#endif

typedef Graph::GraphNode GNode;

Graph graph;

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "[id: " << &n << "]";
  return os;
}

typedef std::pair<GNode,GNode> Edge;

Galois::InsertBag<Edge> mst;

/** 
 * Construct a spanning forest via a modified BFS algorithm. Intended as a
 * simple introduction to the Galois system and not intended to particularly
 * fast. Restrictions: graph must be strongly connected. In this case, the
 * spanning tree is over the undirected graph created by making the directed
 * graph symmetric.
 */
struct DemoAlgo {
  Node* root;

  void operator()(GNode src, Galois::UserContext<GNode>& ctx) {
    for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::ALL),
        ei = graph.edge_end(src, Galois::ALL); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& ddata = graph.getData(dst, Galois::NONE);
      if (ddata.component == root)
        continue;
      ddata.component = root;
      mst.push(std::make_pair(src, dst));
      ctx.push(dst);
    }
  }

  void operator()() {
    Graph::iterator ii = graph.begin(), ei = graph.end();
    if (ii != ei) {
      root = &graph.getData(*ii);
      Galois::for_each(*ii, *this);
    }
  }
};

/**
 * Like asynchronous connected components algorithm. 
 */
struct AsynchronousAlgo {
  struct Merge {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_parallel_push;
    typedef int tt_does_not_need_stats;

    Galois::Statistic& emptyMerges;
    Merge(Galois::Statistic& e): emptyMerges(e) { }

    //! Add the next edge between components to the worklist
    void operator()(const GNode& src, Galois::UserContext<GNode>&) const {
      (*this)(src);
    }

    void operator()(const GNode& src) const {
      Node& sdata = graph.getData(src, Galois::NONE);
      for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::NONE),
          ei = graph.edge_end(src, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, Galois::NONE);
        if (sdata.merge(&ddata)) {
          mst.push(std::make_pair(src, dst));
        } else {
          emptyMerges += 1;
        }
      }
    }
  };

  //! Normalize component by doing find with path compression
  struct Normalize {
    void operator()(const GNode& src) const {
      Node& sdata = graph.getData(src, Galois::NONE);
      sdata.component = sdata.findAndCompress();
    }
  };

  void operator()() {
    Galois::Statistic emptyMerges("EmptyMerges");
    Galois::for_each_local(graph, Merge(emptyMerges));
    Galois::do_all_local(graph, Normalize());
  }
};

struct is_bad_graph {
  bool operator()(const GNode& n) const {
    Node& me = graph.getData(n);
    for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      if (me.component != data.component) {
        std::cerr << "not in same component: " << me << " and " << data << "\n";
        return true;
      }
    }
    return false;
  }
};

struct is_bad_mst {
  bool operator()(const Edge& e) const {
    return graph.getData(e.first).component != graph.getData(e.second).component;
  }
};

struct CheckAcyclic {
  struct Accum {
    Galois::GAccumulator<unsigned> roots;
  };

  Accum* accum;

  void operator()(const GNode& n) {
    Node& data = graph.getData(n);
    if (data.component == &data)
      accum->roots += 1;
  }

  bool operator()() {
    Accum a;
    accum = &a;
    Galois::do_all_local(graph, *this);
    unsigned numRoots = a.roots.reduce();
    unsigned numEdges = std::distance(mst.begin(), mst.end());
    if (graph.size() - numRoots != numEdges) {
      std::cerr << "Generated graph is not a forest. "
        << "Expected " << graph.size() - numRoots << " edges but "
        << "found " << numEdges << "\n";
      return false;
    }

    std::cout << "Num trees: " << numRoots << "\n";
    std::cout << "Tree edges: " << numEdges << "\n";
    return true;
  }
};

bool verify() {
  if (Galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad_graph()) == graph.end()) {
    if (Galois::ParallelSTL::find_if(mst.begin(), mst.end(), is_bad_mst()) == mst.end()) {
      CheckAcyclic c;
      return c();
    }
  }
  return false;
}

template<typename Algo>
void run() {
  Algo algo;

  Galois::StatTimer T;
  T.start();
  algo();
  T.stop();
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();
  graph.structureFromFile(inputFilename.c_str());
  std::cout << "Num nodes: " << graph.size() << "\n";
  Tinitial.stop();

  //Galois::preAlloc(numThreads);
  Galois::Statistic("MeminfoPre", GaloisRuntime::MM::pageAllocInfo());
  switch (algo) {
    case demo: run<DemoAlgo>(); break;
    case asynchronous: run<AsynchronousAlgo>(); break;
    default: std::cerr << "Unknown algo: " << algo << "\n";
  }
  Galois::Statistic("MeminfoPost", GaloisRuntime::MM::pageAllocInfo());

  if (!skipVerify && !verify()) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  return 0;
}
