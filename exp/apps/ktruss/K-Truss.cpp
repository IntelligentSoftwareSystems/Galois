/** Maximal k-trusses for a given undirected graph -*- C++ -*-
 * @example K-Truss.cpp
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * Maximal k-trusses for a given undirected graph
 *
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <deque>
#include <algorithm>
#include <fstream>

enum Algo {
  bsp,
  async
};

namespace cll = llvm::cl;

static const char* name = "Maximal k-trusses";
static const char* desc =
  "Computes the maximal k-trusses for a given undirected graph";
static const char* url = "k_truss";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int> trussNum("trussNum", cll::desc("report trussNum-trusses"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"), 
  cll::values(
    clEnumValN(Algo::bsp, "bsp", "Bulk-synchronous parallel (default)"), 
    clEnumValN(Algo::async, "async", "Asynchronous"), 
    clEnumValEnd), cll::init(Algo::bsp));

static const unsigned int valid = 0x0;
static const unsigned int removed = 0x1;

// edge weight: (# triangles supported << 1) | removal
//   set LSB of an edge weight to indicate the removal of the edge.
//   << 1 to track # triangles an edge supports, 
//   >> 1 when computing edge supports
typedef Galois::Graph::LC_CSR_Graph<void, unsigned int>
  ::template with_numa_alloc<true>::type Graph;
typedef Graph::GraphNode GNode;

typedef std::pair<GNode, GNode> Edge;
typedef Galois::InsertBag<Edge> EdgeVec;

template<typename Graph>
void initialize(Graph& g) {
  g.sortAllEdgesByDst();

  // initializa all edges to valid
  Galois::do_all_local(
    g, 
    [&g] (typename Graph::GraphNode N) { 
      for (auto e: g.edges(N)) {
        g.getEdgeData(e) = valid;
      }
    },
    Galois::do_all_steal<true>()
  );
}

template<typename Graph>
void printGraph(Graph& g) {
  for (auto n: g) {
    std::cout << "node " << n << std::endl;
    for (auto e: g.edges(n)) {
      auto d = g.getEdgeDst(e);
      if (d >= n) continue;
      std::cout << "  edge to " << d << ((g.getEdgeData(e) & removed) ? " removed" : "") << std::endl;
    }
  }
}

template<typename G>
size_t countValidNodes(G& g) {
  Galois::GAccumulator<size_t> numNodes;

  Galois::do_all_local(g, 
    [&g, &numNodes] (GNode n) {
      for (auto e: g.edges(n)) {
        if (!(g.getEdgeData(e) & removed)) {
          numNodes += 1;
          break;
        }
      }
    },
    Galois::do_all_steal<true>()
  );

  return numNodes.reduce();
}

template<typename G>
size_t countValidEqual(G& g, typename G::GraphNode src, typename G::GraphNode dst) {
  size_t retval = 0;
  auto srcI = g.edge_begin(src, Galois::MethodFlag::UNPROTECTED), 
    srcE = g.edge_end(src, Galois::MethodFlag::UNPROTECTED), 
    dstI = g.edge_begin(dst, Galois::MethodFlag::UNPROTECTED), 
    dstE = g.edge_end(dst, Galois::MethodFlag::UNPROTECTED);

  while (true) {
    // find the first valid edge
    while (srcI != srcE && (g.getEdgeData(srcI) & removed)) {
      ++srcI;
    }
    while (dstI != dstE && (g.getEdgeData(dstI) & removed)) {
      ++dstI;
    }

    if (srcI == srcE || dstI == dstE) {
      return retval;
    }

    // check for intersection
    auto sN = g.getEdgeDst(srcI), dN = g.getEdgeDst(dstI);
    if (sN < dN) {
      ++srcI;
    } else if (dN < sN) {
      ++dstI;
    } else {
      retval += 1;
      ++srcI;
      ++dstI;
    }
  }
  return retval;
}

void reportKTruss(Graph& g, unsigned int k, std::string algoName) {
  std::string outName = algoName + "-" + std::to_string(k) + "-truss.txt";
  std::ofstream of(outName);
  auto unprotected = Galois::MethodFlag::UNPROTECTED;
  for (auto n: g) {
    for (auto e: g.edges(n, unprotected)) {
      auto dst = g.getEdgeDst(e);
      if (n < dst && (g.getEdgeData(e) & 0x1) != removed) {
        of << n << " " << dst << std::endl;
      }
    }
  }
}

struct BSPAlgo {
  std::string name() { return "bsp"; }

  struct PickUnsupportedEdges {
    Graph& g;
    unsigned int j;
    EdgeVec& r;
    EdgeVec& s;
    PickUnsupportedEdges(Graph& g, unsigned int j, EdgeVec& r, EdgeVec& s): g(g), j(j), r(r), s(s) {}

    void operator()(Edge e) {
      if (countValidEqual(g, e.first, e.second) < j) {
        r.push_back(e);
      } else {
        s.push_back(e);
      }
    }
  };

  void operator()(Graph& g, unsigned int k) {
    if (0 == k-2) {
      return;
    }

    EdgeVec unsupported, work[2];
    EdgeVec *cur = &work[0], *next = &work[1];

    // symmetry breaking: 
    // consider only edges (i, j) where i < j
    Galois::do_all_local(g, 
      [&g, cur] (GNode n) {
        for (auto e: g.edges(n)) {
          auto dst = g.getEdgeDst(e);
          if (dst > n) {
            cur->push_back(std::make_pair(n, dst));
          }
        }
      },
      Galois::do_all_steal<true>()
    );

    size_t iter = 0;

    while (true) {
      std::cout << "Iteration " << iter << ": " << std::distance(cur->begin(), cur->end()) << " edges" << std::endl;

      Galois::do_all_local(*cur, 
        PickUnsupportedEdges{g, k-2, unsupported, *next},
        Galois::do_all_steal<true>()
      );

      if (0 == std::distance(unsupported.begin(), unsupported.end())) {
        break;
      }

      // mark unsupported edges as removed
      Galois::do_all_local(unsupported, 
        [&g] (Edge e) {
          g.getEdgeData(g.findEdgeSortedByDst(e.first, e.second)) = removed;
          g.getEdgeData(g.findEdgeSortedByDst(e.second, e.first)) = removed;
        },
        Galois::do_all_steal<true>()
      );

      unsupported.clear();
      cur->clear();
      std::swap(cur, next);
      ++iter;
    } 

    std::cout << "Ends at iteration " << iter << " with " << countValidNodes(g) << " nodes." << std::endl;
  } // end operator()
}; // end struct BSPAlgo

template<typename Algo>
void run() {
  Algo algo;
  Graph g;

  Galois::Graph::readGraph(g, filename);
  std::cout << "Read " << g.size() << " nodes" << std::endl;

  initialize(g);
//  printGraph(g);

  std::cout << "Running " << algo.name() << " algorithm for maximal " << trussNum << "-truss" << std::endl;

  Galois::StatTimer T;
  T.start();
  algo(g, trussNum);
  T.stop();
  reportKTruss(g, trussNum, algo.name());
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if (2 > trussNum) {
    std::cerr << "trussNum >= 2" << std::endl;
    return -1;
  }

  Galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
  case bsp: 
    run<BSPAlgo>(); 
    break;
  case async: 
//    run<AsyncAlgo>(); 
    break;
  default: 
    std::cerr << "Unknown algorithm\n"; 
    abort();
  }
  T.stop();

  return 0;
}
