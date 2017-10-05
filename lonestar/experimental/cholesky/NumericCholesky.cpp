/** NumericCholesky application -*- C++ -*-
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
 * Compute numeric Cholesky factorization of a filled graph.
 *
 * @author Noah Anderson <noah@ices.utexas.edu>
 */

// A bunch of this is copied from SpanningTree

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/UnionFind.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdio> // For certain debugging output

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
  int seen;
  Node(): seen(0) { };
};

// WARNING: Will silently behave oddly when given wrong data type
typedef double edgedata;
//typedef float edgedata;

typedef galois::graphs::LC_Linear_Graph<Node,edgedata>::with_numa_alloc<true>::type Graph;

typedef Graph::GraphNode GNode;

Graph graph;

// The dependency list is stored as a total ordering
typedef unsigned int DepItem;
DepItem *depgraph;


std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "[id: " << &n << "]";
  return os;
}

// Adapted from preflowpush/Preflowpush.cpp
Graph::edge_iterator findEdge(Graph& g, GNode src, GNode dst, bool *hasEdge) {
  Graph::edge_iterator ii = g.edge_begin(src, galois::MethodFlag::UNPROTECTED),
                       ei = g.edge_end(src, galois::MethodFlag::UNPROTECTED);
  *hasEdge = false;
  for (; ii != ei; ++ii) {
    if (g.getEdgeDst(ii) == dst) {
      *hasEdge = true;
      break;
    }
  }
  return ii;
}

// include/galois/graphs/Serialize.h
bool outputTextEdgeData(const char* ofile, Graph& G) {
  std::ofstream file(ofile);
  for (Graph::iterator ii = G.begin(),
         ee = G.end(); ii != ee; ++ii) {
    unsigned src = G.getData(*ii).id;
    // FIXME: Version in include/galois/graphs/Serialize.h is wrong.
    for (Graph::edge_iterator jj = G.edge_begin(*ii),
           ej = G.edge_end(*ii); jj != ej; ++jj) {
      unsigned dst = G.getData(G.getEdgeDst(jj)).id;
      file << src << ' ' << dst << ' ' << G.getEdgeData(jj) << '\n';
    }
  }
  return true;
}

/**
 * Comparison function. The symbolic factorization produces a total
 * ordering of the nodes. This defines the traversal order for the
 * numeric factorization.
 */
struct Cmp {
  bool operator()(const GNode& node1, const GNode& node2) const {
    Node &node1d = graph.getData(node1, galois::MethodFlag::UNPROTECTED);
    Node &node2d = graph.getData(node2, galois::MethodFlag::UNPROTECTED);
    int pos1 = -1, pos2 = -1;

    // Check the total ordering to determine if item1 <= item2
    for ( int n = graph.size(), i = 0; i < n; i++ ) {
      if ( depgraph[i] == node1d.id ) pos1 = i;
      if ( depgraph[i] == node2d.id ) pos2 = i; // FIXME: make else if
      if ( pos1 >= 0 && pos2 >= 0 ) break;      // FIXME: eliminate
    }
    assert(pos1 >= 0 && pos2 >= 0);
    bool result = pos1 <= pos2;
    /*
    std::cout << "Cmp: " << node1d.id << " <= " << node2d.id << ": " <<
      (result ? "true" : "false") << "\n";
    */
    return result;
  }
};

/**
 * Defining the neighborhood of the operator. The operator touches all
 * of the edges to and between neighbors.
 */
struct NhFunc {
  /*
  typedef int tt_has_fixed_neighborhood;
  static_assert(galois::has_fixed_neighborhood<NhFunc>::value, "Oops!");
  */

  template<typename C>
  void operator()(GNode& node, C& ctx) {
    (*this)(node);
  }
  void operator()(GNode& node) {
    // Touch all neighbors (this seems to be good enough)
    graph.edge_begin(node);
  }
};

/**
 * Perform the numeric factorization. Assumes the graph is a directed
 * graph produced by symbolic factorization.
 */
struct DemoAlgo {
  //typedef int tt_does_not_need_push;
  //static_assert(galois::does_not_need_push<DemoAlgo>::value, "Oops!");

  void operator()(GNode node, galois::UserContext<GNode>& ctx) {
    // Find self-edge for this node, update it
    bool hasEdge = false;
    edgedata& factor = graph.getEdgeData(findEdge(graph, node, node, &hasEdge),
                                         galois::MethodFlag::UNPROTECTED);
    assert(hasEdge);
    assert(factor > 0);
    factor = sqrt(factor);
    assert(factor != 0 && !isnan(factor));

    //std::cout << "STARTING " << node << "\n";

    // Check seen flag on node
    Node &noded = graph.getData(node);
    assert(noded.seen == 0);
    // DO NOT UPDATE THE SEEN FLAG YET, it may cause problems.

    //std::cout << "STARTING " << noded.id << " " << factor << "\n";
    //printf("STARTING %4d %10.5f\n", noded.id, factor);

    // Update all edges (except self-edge)
    for (Graph::edge_iterator ii = graph.edge_begin(node),
           ei = graph.edge_end(node); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node &dstd = graph.getData(dst);
      if ( !dstd.seen && dst != node ) {
        edgedata &ed = graph.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);
        ed /= factor;
        //printf("N-EDGE %4d %4d %10.5f\n", noded.id, graph.getData(dst).id, ed);
        //std::cout << noded.id << " " << dstd.id << " " << ed << "\n";
      }
    }

    // Update all edges between neighbors (we're operating on the filled graph,
    // so we they form a (directed) clique)
    for (Graph::edge_iterator iis = graph.edge_begin(node),
         eis = graph.edge_end(node);
         iis != eis; ++iis) {
      GNode src = graph.getEdgeDst(iis);
      Node &srcd = graph.getData(src);
      if ( srcd.seen || src == node ) continue;
      edgedata& eds = graph.getEdgeData(iis, galois::MethodFlag::UNPROTECTED);

      // Enumerate all other neighbors
      for (Graph::edge_iterator iid = graph.edge_begin(node),
           eid = graph.edge_end(node);
           iid != eid; ++iid) {
        GNode dst = graph.getEdgeDst(iid);
        Node &dstd = graph.getData(dst);
        if ( dstd.seen || dst == node ) continue;

        // Find the edge that bridges these two neighbors
        hasEdge = false; // FIXME: There must be a better way
        Graph::edge_iterator bridge = findEdge(graph, src, dst, &hasEdge);
        if ( !hasEdge ) continue;

        // Update the weight of the bridge edge
        edgedata &edd = graph.getEdgeData(iid, galois::MethodFlag::UNPROTECTED),
          &edb = graph.getEdgeData(bridge, galois::MethodFlag::UNPROTECTED);
        edb -= eds*edd;

        //printf("I-EDGE %4d %4d %10.5f\n", srcd.id, dstd.id, edb);
        //std::cout << srcd.id << " " << dstd.id << " " << edb << "\n";
      }
    }
    //std::cout << "OPERATED ON " << noded.id << "\n";
    //sleep(1); // Use this to help debug parallelism

    // Now update the seen flag.
    assert(noded.seen == 0);
    noded.seen = 1;
    assert(noded.seen == 1);
    //printf("FINISHED %4d %d\n", noded.id, noded.seen);
  }

  void operator()() {
    Graph::iterator ii = graph.begin(), ei = graph.end();
    if (ii != ei) { // Ensure there is at least one node in the graph.
      galois::for_each_ordered(ii, ei, Cmp(), NhFunc(), *this);
      //galois::for_each(ii, ei, *this);
    }
  }
};

// FIXME: implement verify, etc. See SpanningTree.

bool verify() {
  outputTextEdgeData("choleskyedges.txt", graph);
  std::cout << "\n\n\nPlease verify by comparing choleskyedges.txt against expected contents.\n\n\n\n"; 
  return true;
  /*
  if (galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad_graph()) == graph.end()) {
    if (galois::ParallelSTL::find_if(mst.begin(), mst.end(), is_bad_mst()) == mst.end()) {
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

  galois::StatTimer T;
  T.start();
  algo();
  T.stop();
}

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();

  // Load filled graph with edge data
  galois::graphs::readGraph(graph, inputFilename.c_str());
  std::cout << "Num nodes: " << graph.size() << "\n";

  // Assign IDs to each node
  {
    unsigned int n = graph.size(), i = 0;
    for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      Node& data = graph.getData(*ii);
      data.id = i++;
      assert(!data.seen);
    }
    assert(i == n);

    // Load dependence ordering
    depgraph = new DepItem[n];
    assert(depgraph);
    std::ifstream depfile(depFilename.c_str());
    i = 0;
    while (depfile) {
      unsigned int node;
      depfile >> node;
      if ( !depfile ) break;
      assert(node < n);
      if ( /* i < 0  || */ i >= n ) {
        std::cout << "Error loading dependencies.\n";
        abort();
      }
      depgraph[i] = node;
      i++;
    }
    assert(i == n);
    depfile.close();
  }

  Tinitial.stop();

  //galois::preAlloc(numThreads);
  galois::reportPageAlloc("MeminfoPre");

  switch (algo) {
    case demo: run<DemoAlgo>(); break;
    //case asynchronous: run<AsynchronousAlgo>(); break;
    default: std::cerr << "Unknown algo: " << algo << "\n";
  }
  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify && !verify()) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  return 0;
}
