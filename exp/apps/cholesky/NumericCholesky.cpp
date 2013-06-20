/** Elimination game -*- C++ -*-
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
 * Play the elimination game on a graph.
 *
 * @author Noah Anderson <noah@ices.utexas.edu>
 */

// A bunch of this is copied from SpanningTree

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/UnionFind.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>

namespace cll = llvm::cl;

const char* name = "Numeric Cholesky Factorization";
const char* desc = "Compute the numeric cholesky factorization of a filled graph";
const char* url = NULL;

enum Algo {
  demo//,
  //asynchronous
};

static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<filled graph file>"), cll::Required);
static cll::opt<std::string> depFilename(cll::Positional, cll::desc("<dependency graph file>"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(demo, "Demonstration algorithm"),
      //clEnumVal(asynchronous, "Asynchronous"),
      clEnumValEnd), cll::init(demo));

struct Node {
  unsigned id;
  bool seen;
  Node(): seen(false) { };
};

typedef float edgedata;

typedef Galois::Graph::LC_Linear_Graph<Node,edgedata>::with_numa_alloc<true>::type Graph;

typedef Graph::GraphNode GNode;

Graph graph;

// The dependency list is stored as a total ordering
typedef unsigned int DepItem;
DepItem *depgraph;


std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "[id: " << &n << "]";
  return os;
}

typedef std::pair<GNode,GNode> Edge;

Galois::InsertBag<Edge> mst;

// Copied from preflowpush/Preflowpush.cpp
Graph::edge_iterator findEdge(Graph& g, GNode src, GNode dst) {
  Graph::edge_iterator ii = g.edge_begin(src, Galois::MethodFlag::NONE),
                       ei = g.edge_end(src, Galois::MethodFlag::NONE);
  for (; ii != ei; ++ii) {
    if (g.getEdgeDst(ii) == dst)
      break;
  }
  return ii;
}

// include/Galois/Graphs/Serialize.h
bool outputTextEdgeData(const char* ofile, Graph& G) {
  std::ofstream file(ofile);
  for (Graph::iterator ii = G.begin(),
         ee = G.end(); ii != ee; ++ii) {
    unsigned src = G.getData(*ii).id;
    // FIXME: Version in include/Galois/Graphs/Serialize.h is wrong.
    for (Graph::edge_iterator jj = G.edge_begin(*ii),
           ej = G.edge_end(*ii); jj != ej; ++jj) {
      unsigned dst = G.getData(G.getEdgeDst(jj)).id;
      file << src << ' ' << dst << ' ' << G.getEdgeData(jj) << '\n';
    }
  }
  return true;
}

void probe_graph() {
#if 1
  unsigned int n = graph.size(), i = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    Node& data = graph.getData(*ii, Galois::MethodFlag::NONE);
    assert(data.id == i++);
  }
  assert(i == n);
#endif
}

struct Cmp {
  bool operator()(const GNode& item1, const GNode& item2) const {
    probe_graph();
    Node &a = graph.getData(item1, Galois::MethodFlag::NONE);
    Node &b = graph.getData(item2, Galois::MethodFlag::NONE);
    unsigned int posa = -1, posb = -1;

    // Check if path exists from A to B. Then A <= B.
    for ( unsigned int n = graph.size(), i = 0; i < n; i++ ) {
      if ( depgraph[i] == a.id ) posa = i;
      else if ( depgraph[i] == b.id ) posb = i;
      else continue;
      if ( posa >= 0 && posb >= 0 ) break;
    }
    assert(posa >= 0 && posb >= 0); // FIXME: Implements a total ordering
    bool result = posa <= posb;
    std::cout << "Cmp: " << a.id << " <= " << b.id << ": " << (result ? "true" : "false") << "\n";
    return result;
  }
};

struct NhFunc {
  void operator()(GNode& item, Galois::UserContext<GNode>& ctx) {
    (*this)(item);
  }
  void operator()(GNode& item, int &something) {
    // include/Galois/Runtime/LCordered.h:477
    (*this)(item);
  }
  void operator()(GNode& item) {
    // Touch all neighbors
    probe_graph();
    Graph::edge_iterator ii = graph.edge_begin(item, Galois::MethodFlag::ALL); // This seems to be enough
    probe_graph();
#if 0
    unsigned a = graph.getData(item, Galois::MethodFlag::NONE).id;
    std::cout << "Neighbors of " << a << "\n";
    for (Graph::edge_iterator ei = graph.edge_end(item, Galois::MethodFlag::ALL); ii != ei; ++ii) {
      std::cout << a << "-" << graph.getData(graph.getEdgeDst(ii), Galois::MethodFlag::NONE).id << "\n";
    }
#endif
  }
};

/** 
 * Do stuff
 */
struct DemoAlgo {
  typedef int tt_has_fixed_neighborhood;
  Node* root;

  void operator()(GNode src, Galois::UserContext<GNode>& ctx) {
    probe_graph();
    // Find self-edge for this node, update it
    edgedata& factor = graph.getEdgeData(findEdge(graph, src, src), Galois::MethodFlag::NONE);
    factor = sqrt(factor);
    assert(factor != 0);

    // Update seen flag on node
    Node &srcd = graph.getData(src);
    assert(!srcd.seen);
    srcd.seen = true;

    // Update all edges (except self-edge)
    
    for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::MethodFlag::ALL),
         ei = graph.edge_end(src, Galois::MethodFlag::ALL); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      if ( !graph.getData(dst).seen )
        graph.getEdgeData(ii, Galois::MethodFlag::NONE) /= factor;
    }

    // Update all edges between neighbors (we're operating on the filled graph,
    // so we can assume they form a clique)
    for (Graph::edge_iterator ii1 = graph.edge_begin(src, Galois::MethodFlag::ALL),
         ei1 = graph.edge_end(src, Galois::MethodFlag::ALL); ii1 != ei1; ++ii1) {
      GNode srcn = graph.getEdgeDst(ii1);
      if ( graph.getData(srcn).seen ) continue;
      edgedata& ed1 = graph.getEdgeData(ii1, Galois::MethodFlag::NONE);
      for (Graph::edge_iterator ii2 = graph.edge_begin(src, Galois::MethodFlag::ALL),
           ei2 = graph.edge_end(src, Galois::MethodFlag::ALL); ii2 != ei2; ++ii2) {
        GNode dstn = graph.getEdgeDst(ii2);
        if ( graph.getData(dstn).seen ) continue;
        edgedata &ed2 = graph.getEdgeData(ii2, Galois::MethodFlag::NONE),
          &ed3 = graph.getEdgeData(findEdge(graph, srcn, dstn), Galois::MethodFlag::NONE);
        // Update the edge between these two neighbors
        ed3 -= ed1*ed2;
      }
    }
    std::cout << "OPERATED ON " << srcd.id << "\n";
    //sleep(1); // Use this to help debug parallelism
    probe_graph();
  }

  void operator()() {
    Graph::iterator ii = graph.begin(), ei = graph.end();
    if (ii != ei) {
      //Galois::for_each_ordered(ii, ei, Cmp(), NhFunc(), *this);
      Galois::for_each(ii, ei, *this);
      //Galois::for_each(ii, ei, *this);
    }
  }
};
static_assert(Galois::has_fixed_neighborhood<DemoAlgo>::value, "OOps");

// FIXME: implement verify, etc. See SpanningTree.

bool verify() {
  outputTextEdgeData("choleskyedges.txt", graph);
  std::cout << "\n\n\nPlease verify by comparing choleskyedges.txt against expected contents.\n\n\n\n"; 
  return true;
  /*
  if (Galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad_graph()) == graph.end()) {
    if (Galois::ParallelSTL::find_if(mst.begin(), mst.end(), is_bad_mst()) == mst.end()) {
      CheckAcyclic c;
      return c();
    }
  }
  return false;
  */
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

  // Load filled graph with edge data
  Galois::Graph::readGraph(graph, inputFilename.c_str());
  std::cout << "Num nodes: " << graph.size() << "\n";

  // Assign IDs to each node
  {
    unsigned int n = graph.size(), i = 0;
    for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      Node& data = graph.getData(*ii);
      data.id = i++;
    }
    assert(i == n);
#if 0
    // Load dependence tree
    depgraph = new DepItem[n];
    assert(depgraph);
    std::ifstream depfile(depFilename.c_str());
    i = 0;
    while (depfile) {
      unsigned int node;
      depfile >> node;
      if ( !depfile ) break;
      assert(node < n);
      if ( i < 0 || i >= n ) {
        std::cout << "Error loading dependency graph.\n";
        abort();
      }
      depgraph[i] = node;
      i++;
    }
    assert(i == n);
    depfile.close();
#endif
  }
  //probe_graph();

  Tinitial.stop();
  
  //Galois::preAlloc(numThreads);
  Galois::reportPageAlloc("MeminfoPre");

  switch (algo) {
    case demo: run<DemoAlgo>(); break;
    //case asynchronous: run<AsynchronousAlgo>(); break;
    default: std::cerr << "Unknown algo: " << algo << "\n";
  }
  Galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify && !verify()) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  return 0;
}
