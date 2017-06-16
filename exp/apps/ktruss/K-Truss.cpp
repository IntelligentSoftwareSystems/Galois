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
  bspIm,
  bspCoreThenTruss,
  async
};

namespace cll = llvm::cl;

static const char* name = "Maximal k-trusses";
static const char* desc =
  "Computes the maximal k-trusses for a given undirected graph";
static const char* url = "k_truss";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int> trussNum("trussNum", cll::desc("report trussNum-trusses"), cll::Required);
static cll::opt<std::string> outName("o", cll::desc("output file for the edgelist of resulting truss"));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"), 
  cll::values(
    clEnumValN(Algo::bsp, "bsp", "Bulk-synchronous parallel (default)"), 
    clEnumValN(Algo::bspIm, "bspIm", "Bulk-synchronous parallel improved"),
    clEnumValN(Algo::bspCoreThenTruss, "bspCoreThenTruss", "Compute k-1 core and then k-truss in bspIm way"),
    clEnumValN(Algo::async, "async", "Asynchronous"), 
    clEnumValEnd), cll::init(Algo::bsp));

static const uint32_t valid = 0x0;
static const uint32_t removed = 0x1;

template<typename Graph>
void initialize(Graph& g) {
  g.sortAllEdgesByDst();

  // initializa all edges to valid
  Galois::do_all_local(
    g, 
    [&g] (typename Graph::GraphNode N) { 
      for (auto e: g.edges(N, Galois::MethodFlag::UNPROTECTED)) {
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
    for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
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
    [&g, &numNodes] (typename G::GraphNode n) {
      for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
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

template<typename Graph>
void reportKTruss(Graph& g, unsigned int k, std::string algoName) {
  if (outName.empty()) {
    return;
  }

  std::ofstream of(outName);
  if (!of.is_open()) {
    std::cerr << "Cannot open " << outName << " for output." << std::endl;
    return;
  }

  for (auto n: g) {
    for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
      auto dst = g.getEdgeDst(e);
      if (n < dst && (g.getEdgeData(e) & 0x1) != removed) {
        of << n << " " << dst << std::endl;
      }
    }
  }
}

template<typename G>
bool isValidDegreeNoLessThanJ(G& g, typename G::GraphNode n, unsigned int j) {
  size_t numValid = 0;
  for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
    if (!(g.getEdgeData(e) & removed)) {
      numValid += 1;
      if (numValid >= j) {
        return true;
      }
    }
  }
  return numValid >= j;
}

template<typename G>
bool isSupportNoLessThanJ(G& g, typename G::GraphNode src, typename G::GraphNode dst, unsigned int j) {
  size_t numValidEqual = 0;
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

// BSPAlgo:
// 1. Scan for unsupported edges.
// 2. If no unsupported edges are found, done.
// 3. Remove unsupported edges in a separated loop.
// 4. Go back to 1.
struct BSPAlgo {
  // edge weight: (# triangles supported << 1) | removal
  //   set LSB of an edge weight to indicate the removal of the edge.
  //   << 1 to track # triangles an edge supports, 
  //   >> 1 when computing edge supports
  typedef Galois::Graph::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef std::pair<GNode, GNode> Edge;
  typedef Galois::InsertBag<Edge> EdgeVec;

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
    if (0 == k-2) {
      return;
    }

    EdgeVec unsupported, work[2];
    EdgeVec *cur = &work[0], *next = &work[1];

    // symmetry breaking: 
    // consider only edges (i, j) where i < j
    Galois::do_all_local(g, 
      [&g, cur] (GNode n) {
        for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
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
      std::cout << "Iteration " << iter << ": ";
      std::cout << countValidNodes(g) << " valid nodes, ";
      std::cout << std::distance(cur->begin(), cur->end()) << " valid edges" << std::endl;

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

    std::cout << "Ends at iteration " << iter << std::endl;
  } // end operator()
}; // end struct BSPAlgo

// BSPImprovedAlgo:
// 1. Keep supported edges and remove unsupported edges.
// 2. If all edges are kept, done.
// 3. Go back to 3.
struct BSPImprovedAlgo {
  // edge weight: (# triangles supported << 1) | removal
  //   set LSB of an edge weight to indicate the removal of the edge.
  //   << 1 to track # triangles an edge supports, 
  //   >> 1 when computing edge supports
  typedef Galois::Graph::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef std::pair<GNode, GNode> Edge;
  typedef Galois::InsertBag<Edge> EdgeVec;

  std::string name() { return "bspIm"; }

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
    if (0 == k-2) {
      return;
    }

    EdgeVec work[2];
    EdgeVec *cur = &work[0], *next = &work[1];
    size_t curSize, nextSize;

    // symmetry breaking: 
    // consider only edges (i, j) where i < j
    Galois::do_all_local(g, 
      [&g, cur] (GNode n) {
        for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
          auto dst = g.getEdgeDst(e);
          if (dst > n) {
            cur->push_back(std::make_pair(n, dst));
          }
        }
      },
      Galois::do_all_steal<true>()
    );
    curSize = std::distance(cur->begin(), cur->end());

    size_t iter = 0;

    // remove unsupported edges until no more edges can be removed
    while (true) {
      std::cout << "Iteration " << iter << ": ";
      std::cout << countValidNodes(g) << " valid nodes, ";
      std::cout << std::distance(cur->begin(), cur->end()) << " valid edges" << std::endl;

      Galois::do_all_local(*cur, 
        KeepSupportedEdges{g, k-2, *next},
        Galois::do_all_steal<true>()
      );
      nextSize = std::distance(next->begin(), next->end());

      if (curSize == nextSize) {
        // every edge in *cur is kept, done
        break;
      }

      cur->clear();
      curSize = nextSize;
      std::swap(cur, next);
      ++iter;
    } 

    std::cout << "Ends at iteration " << iter << std::endl;
  } // end operator()
}; // end struct BSPImprovedAlgo

// BSPCoreThenTrussAlgo:
// 1. Reduce the graph to k-1 core
//    a. Keep nodes w/ degree >= k-1 and remove all edges for nodes whose degree < k-1.
//    b. If all nodes are kept, done.
//    c. Go back to a.
// 2. Compute k-truss from k-1 core
//    a. Keep supported edges and remove unsupported edges.
//    b. If all edges are kept, done.
//    c. Go back to a.
struct BSPCoreThenTrussAlgo {
  // edge weight: (# triangles supported << 1) | removal
  //   set LSB of an edge weight to indicate the removal of the edge.
  //   << 1 to track # triangles an edge supports, 
  //   >> 1 when computing edge supports
  typedef Galois::Graph::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef std::pair<GNode, GNode> Edge;
  typedef Galois::InsertBag<Edge> EdgeVec;
  typedef Galois::InsertBag<GNode> NodeVec;

  std::string name() { return "bspCoreThenTruss"; }

  struct KeepValidNodes {
    Graph& g;
    unsigned int j;
    NodeVec& s;

    KeepValidNodes(Graph& g, unsigned int j, NodeVec& s)
      : g(g), j(j), s(s) {}

    void operator()(GNode n) {
      if (isValidDegreeNoLessThanJ(g, n, j)) {
        s.push_back(n);
      } else {
        for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
          auto dst = g.getEdgeDst(e);
          g.getEdgeData(g.findEdgeSortedByDst(n, dst)) = removed;
          g.getEdgeData(g.findEdgeSortedByDst(dst, n)) = removed;
        }
      }
    }
  };

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
    if (0 == k-2) {
      return;
    }

    size_t iter = 0;
    NodeVec nWork[2];
    NodeVec *nCur = &nWork[0], *nNext = &nWork[1];
    size_t nCurSize = g.size(), nNextSize;

    Galois::StatTimer TCore("Reduce_to_k-1_core");
    TCore.start();

    std::cout << "Core iteration " << iter << ": ";
    std::cout << nCurSize << " valid nodes" << std::endl;

    // reduce to k-1 core
    Galois::do_all_local(g, 
      KeepValidNodes{g, k-1, *nNext}, 
      Galois::do_all_steal<true>()
    );
    nNextSize = std::distance(nNext->begin(), nNext->end());

    while (nCurSize != nNextSize) {
      nCur->clear();
      nCurSize = nNextSize;
      std::swap(nCur, nNext);
      ++iter;

      std::cout << "Core iteration " << iter << ": ";
      std::cout << nCurSize << " valid nodes" << std::endl;

      Galois::do_all_local(*nCur, 
        KeepValidNodes{g, k-1, *nNext}, 
        Galois::do_all_steal<true>()
      );
      nNextSize = std::distance(nNext->begin(), nNext->end());
    }

    std::cout << "Ends core computation at iteration " << iter << std::endl;
    TCore.stop();

    iter = 0;
    EdgeVec eWork[2];
    EdgeVec *eCur = &eWork[0], *eNext = &eWork[1];
    size_t eCurSize, eNextSize;

    // symmetry breaking: 
    // consider only valid edges (i, j) where i < j
    Galois::do_all_local(g, 
      [&g, eCur] (GNode n) {
        for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
          auto dst = g.getEdgeDst(e);
          if (dst > n && !(g.getEdgeData(e) & removed)) {
            eCur->push_back(std::make_pair(n, dst));
          }
        }
      },
      Galois::do_all_steal<true>()
    );
    eCurSize = std::distance(eCur->begin(), eCur->end());

    Galois::StatTimer TTruss("Reduce_to_k_truss");
    TTruss.start();

    // remove unsupported edges until no more edges can be removed
    while (true) {
      std::cout << "Truss iteration " << iter << ": ";
      std::cout << countValidNodes(g) << " valid nodes, ";
      std::cout << eCurSize << " valid edges" << std::endl;

      Galois::do_all_local(*eCur,
        KeepSupportedEdges{g, k-2, *eNext},
        Galois::do_all_steal<true>()
      );
      eNextSize = std::distance(eNext->begin(), eNext->end());

      if (eCurSize == eNextSize) {
        // every edge in *eCur is kept, done
        break;
      }

      eCur->clear();
      eCurSize = eNextSize;
      std::swap(eCur, eNext);
      ++iter;
    }

    std::cout << "Ends truss computation at iteration " << iter << std::endl;
    TTruss.stop();
  } // end operator()
}; // end struct BSPCoreThenTrussAlgo

template<typename Algo>
void run() {
  Algo algo;
  typename Algo::Graph g;

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
  case bspIm:
    run<BSPImprovedAlgo>();
    break;
  case bspCoreThenTruss:
    run<BSPCoreThenTrussAlgo>();
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
