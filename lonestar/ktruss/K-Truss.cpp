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

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <deque>
#include <algorithm>
#include <fstream>
#include <memory>

enum Algo {
  bspJacobi,
  bsp,
  bspCoreThenTruss,
};

namespace cll = llvm::cl;

static const char* name = "Maximal k-trusses";
static const char* desc =
    "Computes the maximal k-trusses for a given undirected graph";
static const char* url = "k_truss";

static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int>
    trussNum("trussNum", cll::desc("report trussNum-trusses"), cll::Required);
static cll::opt<std::string>
    outName("o", cll::desc("output file for the edgelist of resulting truss"));
static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(
        clEnumValN(Algo::bspJacobi, "bspJacobi",
                   "Bulk-synchronous parallel with separated edge removal"),
        clEnumValN(Algo::bsp, "bsp", "Bulk-synchronous parallel (default)"),
        clEnumValN(Algo::bspCoreThenTruss, "bspCoreThenTruss",
                   "Compute k-1 core and then k-truss"),
        clEnumValEnd),
    cll::init(Algo::bsp));

static const uint32_t valid   = 0x0;
static const uint32_t removed = 0x1;

template <typename Graph>
void initialize(Graph& g) {
  g.sortAllEdgesByDst();

  // initializa all edges to valid
  galois::do_all(galois::iterate(g),
                 [&g](typename Graph::GraphNode N) {
                   for (auto e : g.edges(N, galois::MethodFlag::UNPROTECTED)) {
                     g.getEdgeData(e) = valid;
                   }
                 },
                 galois::steal());
}

#if 0
template<typename Graph>
void printGraph(Graph& g) {
  for (auto n: g) {
    std::cout << "node " << n << std::endl;
    for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
      auto d = g.getEdgeDst(e);
      if (d >= n) continue;
      std::cout << "  edge to " << d << ((g.getEdgeData(e) & removed) ? " removed" : "") << std::endl;
    }
  }
}

template<typename G>
size_t countValidNodes(G& g) {
  galois::GAccumulator<size_t> numNodes;

  galois::do_all(galois::iterate(g), 
    [&g, &numNodes] (typename G::GraphNode n) {
      for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        if (!(g.getEdgeData(e) & removed)) {
          numNodes += 1;
          break;
        }
      }
    },
    galois::steal()
  );

  return numNodes.reduce();
}

template<typename G>
size_t countValidEdges(G& g) {
  galois::GAccumulator<size_t> numEdges;

  galois::do_all(galois::iterate(g), 
    [&g, &numEdges] (typename G::GraphNode n) {
      for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        if (n < g.getEdgeDst(e) && !(g.getEdgeData(e) & removed)) {
          numEdges += 1;
        }
      }
    },
    galois::steal()
  );

  return numEdges.reduce();
}
#endif

template <typename Graph>
void reportKTruss(Graph& g) {
  if (outName.empty()) {
    return;
  }

  std::ofstream of(outName);
  if (!of.is_open()) {
    std::cerr << "Cannot open " << outName << " for output." << std::endl;
    return;
  }

  for (auto n : g) {
    for (auto e : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
      auto dst = g.getEdgeDst(e);
      if (n < dst && (g.getEdgeData(e) & 0x1) != removed) {
        of << n << " " << dst << std::endl;
      }
    }
  }
}

template <typename G>
bool isSupportNoLessThanJ(G& g, typename G::GraphNode src,
                          typename G::GraphNode dst, unsigned int j) {
  size_t numValidEqual = 0;
  auto srcI            = g.edge_begin(src, galois::MethodFlag::UNPROTECTED),
       srcE            = g.edge_end(src, galois::MethodFlag::UNPROTECTED),
       dstI            = g.edge_begin(dst, galois::MethodFlag::UNPROTECTED),
       dstE            = g.edge_end(dst, galois::MethodFlag::UNPROTECTED);

  while (true) {
    // find the first valid edge
    while (srcI != srcE && (g.getEdgeData(srcI) & removed)) {
      ++srcI;
    }
    while (dstI != dstE && (g.getEdgeData(dstI) & removed)) {
      ++dstI;
    }

    if (srcI == srcE || dstI == dstE) {
      return numValidEqual >= j;
    }

    // check for intersection
    auto sN = g.getEdgeDst(srcI), dN = g.getEdgeDst(dstI);
    if (sN < dN) {
      ++srcI;
    } else if (dN < sN) {
      ++dstI;
    } else {
      numValidEqual += 1;
      if (numValidEqual >= j) {
        return true;
      }
      ++srcI;
      ++dstI;
    }
  }
  return numValidEqual >= j;
}

// BSPTrussJacobiAlgo:
// 1. Scan for unsupported edges.
// 2. If no unsupported edges are found, done.
// 3. Remove unsupported edges in a separated loop.
// 4. Go back to 1.
struct BSPTrussJacobiAlgo {
  // set LSB of an edge weight to indicate the removal of the edge.
  typedef galois::graphs::LC_CSR_Graph<void, uint32_t>::
      template with_numa_alloc<true>::type ::template with_no_lockable<
          true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef std::pair<GNode, GNode> Edge;
  typedef galois::InsertBag<Edge> EdgeVec;

  std::string name() { return "bsp"; }

  struct PickUnsupportedEdges {
    Graph& g;
    unsigned int j;
    EdgeVec& r;
    EdgeVec& s;

    PickUnsupportedEdges(Graph& g, unsigned int j, EdgeVec& r, EdgeVec& s)
        : g(g), j(j), r(r), s(s) {}

    void operator()(Edge e) {
      EdgeVec& w = isSupportNoLessThanJ(g, e.first, e.second, j) ? s : r;
      w.push_back(e);
    }
  };

  void operator()(Graph& g, unsigned int k) {
    if (0 == k - 2) {
      return;
    }

    EdgeVec unsupported, work[2];
    EdgeVec *cur = &work[0], *next = &work[1];

    // symmetry breaking:
    // consider only edges (i, j) where i < j
    galois::do_all(galois::iterate(g),
                   [&g, cur](GNode n) {
                     for (auto e :
                          g.edges(n, galois::MethodFlag::UNPROTECTED)) {
                       auto dst = g.getEdgeDst(e);
                       if (dst > n) {
                         cur->push_back(std::make_pair(n, dst));
                       }
                     }
                   },
                   galois::steal());

    while (true) {
      galois::do_all(galois::iterate(*cur), PickUnsupportedEdges{g, k - 2, unsupported, *next},
                     galois::steal());

      if (0 == std::distance(unsupported.begin(), unsupported.end())) {
        break;
      }

      // mark unsupported edges as removed
      galois::do_all(galois::iterate(unsupported),
                     [&g](Edge e) {
                       g.getEdgeData(g.findEdgeSortedByDst(e.first, e.second)) =
                           removed;
                       g.getEdgeData(g.findEdgeSortedByDst(e.second, e.first)) =
                           removed;
                     },
                     galois::steal());

      unsupported.clear();
      cur->clear();
      std::swap(cur, next);
    }
  } // end operator()
};  // end struct BSPTrussJacobiAlgo

// BSPTrussAlgo:
// 1. Keep supported edges and remove unsupported edges.
// 2. If all edges are kept, done.
// 3. Go back to 3.
struct BSPTrussAlgo {
  // set LSB of an edge weight to indicate the removal of the edge.
  typedef galois::graphs::LC_CSR_Graph<void, uint32_t>::
      template with_numa_alloc<true>::type ::template with_no_lockable<
          true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef std::pair<GNode, GNode> Edge;
  typedef galois::InsertBag<Edge> EdgeVec;

  std::string name() { return "bsp"; }

  struct KeepSupportedEdges {
    Graph& g;
    unsigned int j;
    EdgeVec& s;

    KeepSupportedEdges(Graph& g, unsigned int j, EdgeVec& s)
        : g(g), j(j), s(s) {}

    void operator()(Edge e) {
      if (isSupportNoLessThanJ(g, e.first, e.second, j)) {
        s.push_back(e);
      } else {
        g.getEdgeData(g.findEdgeSortedByDst(e.first, e.second)) = removed;
        g.getEdgeData(g.findEdgeSortedByDst(e.second, e.first)) = removed;
      }
    }
  };

  void operator()(Graph& g, unsigned int k) {
    if (0 == k - 2) {
      return;
    }

    EdgeVec work[2];
    EdgeVec *cur = &work[0], *next = &work[1];
    size_t curSize, nextSize;

    // symmetry breaking:
    // consider only edges (i, j) where i < j
    galois::do_all(galois::iterate(g),
                   [&g, cur](GNode n) {
                     for (auto e :
                          g.edges(n, galois::MethodFlag::UNPROTECTED)) {
                       auto dst = g.getEdgeDst(e);
                       if (dst > n) {
                         cur->push_back(std::make_pair(n, dst));
                       }
                     }
                   },
                   galois::steal());
    curSize = std::distance(cur->begin(), cur->end());

    // remove unsupported edges until no more edges can be removed
    while (true) {
      galois::do_all(galois::iterate(*cur), KeepSupportedEdges{g, k - 2, *next},
                     galois::steal());
      nextSize = std::distance(next->begin(), next->end());

      if (curSize == nextSize) {
        // every edge in *cur is kept, done
        break;
      }

      cur->clear();
      curSize = nextSize;
      std::swap(cur, next);
    }
  } // end operator()
};  // end struct BSPTrussAlgo

template <typename G>
bool isValidDegreeNoLessThanJ(G& g, typename G::GraphNode n, unsigned int j) {
  size_t numValid = 0;
  for (auto e : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
    if (!(g.getEdgeData(e) & removed)) {
      numValid += 1;
      if (numValid >= j) {
        return true;
      }
    }
  }
  return numValid >= j;
}

// BSPCoreAlgo:
// 1. Keep nodes w/ degree >= k and remove all edges for nodes whose degree < k.
// 2. If all nodes are kept, done.
// 3. Go back to 1.
struct BSPCoreAlgo {
  // set LSB of an edge weight to indicate the removal of the edge.
  typedef galois::graphs::LC_CSR_Graph<void, uint32_t>::
      template with_numa_alloc<true>::type ::template with_no_lockable<
          true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef galois::InsertBag<GNode> NodeVec;

  std::string name() { return "bspCore"; }

  struct KeepValidNodes {
    Graph& g;
    unsigned int j;
    NodeVec& s;

    KeepValidNodes(Graph& g, unsigned int j, NodeVec& s) : g(g), j(j), s(s) {}

    void operator()(GNode n) {
      if (isValidDegreeNoLessThanJ(g, n, j)) {
        s.push_back(n);
      } else {
        for (auto e : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
          auto dst                                     = g.getEdgeDst(e);
          g.getEdgeData(g.findEdgeSortedByDst(n, dst)) = removed;
          g.getEdgeData(g.findEdgeSortedByDst(dst, n)) = removed;
        }
      }
    }
  };

  void operator()(Graph& g, unsigned int k) {
    NodeVec work[2];
    NodeVec *cur = &work[0], *next = &work[1];
    size_t curSize = g.size(), nextSize;

    galois::do_all(galois::iterate(g), KeepValidNodes{g, k, *next}, galois::steal());
    nextSize = std::distance(next->begin(), next->end());

    while (curSize != nextSize) {
      cur->clear();
      curSize = nextSize;
      std::swap(cur, next);

      galois::do_all(galois::iterate(*cur), KeepValidNodes{g, k, *next}, galois::steal());
      nextSize = std::distance(next->begin(), next->end());
    }
  }
}; // end BSPCoreAlgo

// BSPCoreThenTrussAlgo:
// 1. Reduce the graph to k-1 core
// 2. Compute k-truss from k-1 core
struct BSPCoreThenTrussAlgo {
  // set LSB of an edge weight to indicate the removal of the edge.
  typedef galois::graphs::LC_CSR_Graph<void, uint32_t>::
      template with_numa_alloc<true>::type ::template with_no_lockable<
          true>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() { return "bspCoreThenTruss"; }

  void operator()(Graph& g, unsigned int k) {
    if (0 == k - 2) {
      return;
    }

    galois::StatTimer TCore("Reduce_to_(k-1)-core");
    TCore.start();

    BSPCoreAlgo bspCore;
    bspCore(g, k - 1);

    TCore.stop();

    galois::StatTimer TTruss("Reduce_to_k-truss");
    TTruss.start();

    BSPTrussAlgo bspTrussIm;
    bspTrussIm(g, k);

    TTruss.stop();
  } // end operator()
};  // end struct BSPCoreThenTrussAlgo

template <typename T>
using PerIterAlloc = typename galois::PerIterAllocTy::rebind<T>::other;

template <typename G>
std::deque<typename G::GraphNode, PerIterAlloc<typename G::GraphNode>>
getValidCommonNeighbors(G& g, typename G::GraphNode src,
                        typename G::GraphNode dst, galois::PerIterAllocTy& a,
                        galois::MethodFlag flag = galois::MethodFlag::WRITE) {
  using GNode = typename G::GraphNode;

  auto srcI = g.edge_begin(src, flag), srcE = g.edge_end(src, flag),
       dstI = g.edge_begin(dst, flag), dstE = g.edge_end(dst, flag);
  std::deque<GNode, PerIterAlloc<GNode>> commonNeighbors(a);

  while (true) {
    // find the first valid edge
    while (srcI != srcE && (g.getEdgeData(srcI) & removed)) {
      ++srcI;
    }
    while (dstI != dstE && (g.getEdgeData(dstI) & removed)) {
      ++dstI;
    }

    if (srcI == srcE || dstI == dstE) {
      break;
    }

    // check for intersection
    auto sN = g.getEdgeDst(srcI), dN = g.getEdgeDst(dstI);
    if (sN < dN) {
      ++srcI;
    } else if (dN < sN) {
      ++dstI;
    } else {
      commonNeighbors.push_back(sN);
      ++srcI;
      ++dstI;
    }
  }
  return commonNeighbors;
}

template <typename Algo>
void run() {
  Algo algo;
  typename Algo::Graph g;

  galois::reportPageAlloc("MeminfoPre");

  galois::graphs::readGraph(g, filename);
  std::cout << "Read " << g.size() << " nodes" << std::endl;

  initialize(g);
  //  printGraph(g);

  std::cout << "Running " << algo.name() << " algorithm for maximal "
            << trussNum << "-truss" << std::endl;

  galois::StatTimer T;
  T.start();
  algo(g, trussNum);
  T.stop();
  galois::reportPageAlloc("MeminfoPost");
  reportKTruss(g);
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  if (2 > trussNum) {
    std::cerr << "trussNum >= 2" << std::endl;
    return -1;
  }

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
  case bspJacobi:
    run<BSPTrussJacobiAlgo>();
    break;
  case bsp:
    run<BSPTrussAlgo>();
    break;
  case bspCoreThenTruss:
    run<BSPCoreThenTrussAlgo>();
    break;
  default:
    std::cerr << "Unknown algorithm\n";
    abort();
  }
  T.stop();

  return 0;
}
