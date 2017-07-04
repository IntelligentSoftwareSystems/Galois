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
  bspJacobi,
  bsp,
  bspCoreThenTruss,
  async,
  asyncCoreThenTruss
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
    clEnumValN(Algo::bspJacobi, "bspJacobi", "Bulk-synchronous parallel with separated edge removal"), 
    clEnumValN(Algo::bsp, "bsp", "Bulk-synchronous parallel (default)"),
    clEnumValN(Algo::bspCoreThenTruss, "bspCoreThenTruss", "Compute k-1 core and then k-truss"),
    clEnumValN(Algo::async, "async", "Asynchronous"), 
    clEnumValN(Algo::asyncCoreThenTruss, "asyncCoreThenTruss", "Compute k-1 core and then k-truss in async way"),
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

template<typename G>
size_t countValidEdges(G& g) {
  Galois::GAccumulator<size_t> numEdges;

  Galois::do_all_local(g, 
    [&g, &numEdges] (typename G::GraphNode n) {
      for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
        if (n < g.getEdgeDst(e) && !(g.getEdgeData(e) & removed)) {
          numEdges += 1;
        }
      }
    },
    Galois::do_all_steal<true>()
  );

  return numEdges.reduce();
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

// BSPTrussJacobiAlgo:
// 1. Scan for unsupported edges.
// 2. If no unsupported edges are found, done.
// 3. Remove unsupported edges in a separated loop.
// 4. Go back to 1.
struct BSPTrussJacobiAlgo {
  // set LSB of an edge weight to indicate the removal of the edge.
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

    while (true) {
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
    } 
  } // end operator()
}; // end struct BSPTrussJacobiAlgo

// BSPTrussAlgo:
// 1. Keep supported edges and remove unsupported edges.
// 2. If all edges are kept, done.
// 3. Go back to 3.
struct BSPTrussAlgo {
  // set LSB of an edge weight to indicate the removal of the edge.
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

    // remove unsupported edges until no more edges can be removed
    while (true) {
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
    } 
  } // end operator()
}; // end struct BSPTrussAlgo

// BSPCoreImprovedAlgo:
// 1. Keep nodes w/ degree >= k and remove all edges for nodes whose degree < k.
// 2. If all nodes are kept, done.
// 3. Go back to 1.
struct BSPCoreImprovedAlgo {
  // set LSB of an edge weight to indicate the removal of the edge.
  typedef Galois::Graph::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef Galois::InsertBag<GNode> NodeVec;

  std::string name() { return "bspCoreIm"; }

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

  void operator()(Graph& g, unsigned int k) {
    NodeVec work[2];
    NodeVec *cur = &work[0], *next = &work[1];
    size_t curSize = g.size(), nextSize;

    Galois::do_all_local(g, 
      KeepValidNodes{g, k, *next}, 
      Galois::do_all_steal<true>()
    );
    nextSize = std::distance(next->begin(), next->end());

    while (curSize != nextSize) {
      cur->clear();
      curSize = nextSize;
      std::swap(cur, next);

      Galois::do_all_local(*cur, 
        KeepValidNodes{g, k, *next}, 
        Galois::do_all_steal<true>()
      );
      nextSize = std::distance(next->begin(), next->end());
    }
  }
};

// BSPCoreThenTrussAlgo:
// 1. Reduce the graph to k-1 core
// 2. Compute k-truss from k-1 core
struct BSPCoreThenTrussAlgo {
  // set LSB of an edge weight to indicate the removal of the edge.
  typedef Galois::Graph::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() { return "bspCoreThenTruss"; }

  void operator()(Graph& g, unsigned int k) {
    if (0 == k-2) {
      return;
    }

    Galois::StatTimer TCore("Reduce_to_k-1_core");
    TCore.start();

    BSPCoreImprovedAlgo bspCoreIm;
    bspCoreIm(g, k-1);

    TCore.stop();

    Galois::StatTimer TTruss("Reduce_to_k_truss");
    TTruss.start();

    BSPTrussAlgo bspTrussIm;
    bspTrussIm(g, k);

    TTruss.stop();
  } // end operator()
}; // end struct BSPCoreThenTrussAlgo

template<typename G>
std::deque<typename G::GraphNode> getValidCommonNeighbors(G& g,
  typename G::GraphNode src, typename G::GraphNode dst, Galois::MethodFlag flag = Galois::MethodFlag::WRITE)
{
  auto srcI = g.edge_begin(src, flag), srcE = g.edge_end(src, flag), 
    dstI = g.edge_begin(dst, flag), dstE = g.edge_end(dst, flag);
  std::deque<typename G::GraphNode> commonNeighbors;

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

// AsyncTrussAlgo:
// 1. Compute support for all edges and pick out unsupported ones.
// 2. Remove unsupported edges, decrease the support for affected edges and pick out those becomeing unsupported.
// 3. Repeat 2. until no more unsupported edges are found.
struct AsyncTrussAlgo {
  // edge weight: (# triangles supported << 1) | removal
  //   set LSB of an edge weight to indicate the removal of the edge.
  //   << 1 to track # triangles an edge supports, 
  //   >> 1 when computing edge supports
  typedef Galois::Graph::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef std::pair<GNode, GNode> Edge;
  typedef Galois::InsertBag<Edge> EdgeVec;

  std::string name() { return "async"; }

  struct PickUnsupportedEdges {
    Graph& g;
    unsigned int j;
    EdgeVec& r;

    PickUnsupportedEdges(Graph& g, unsigned int j, EdgeVec& r)
      : g(g), j(j), r(r) {}

    void operator()(Edge e) {
      auto src = e.first, dst = e.second;
      std::deque<GNode> commonNeighbors = getValidCommonNeighbors(g, src, dst, Galois::MethodFlag::UNPROTECTED);
      auto numValidCommonNeighbors = commonNeighbors.size();

      g.getEdgeData(g.findEdgeSortedByDst(src, dst)) = (numValidCommonNeighbors << 1);
      g.getEdgeData(g.findEdgeSortedByDst(dst, src)) = (numValidCommonNeighbors << 1);
      if (numValidCommonNeighbors < j) {
        r.push_back(e);
      }
    }
  };

  struct PropagateEdgeRemoval {
    Graph& g;
    unsigned int j;

    PropagateEdgeRemoval(Graph& g, unsigned int j): g(g), j(j) {}

    void removeUnsupportedEdge(GNode src, GNode dst, Galois::UserContext<Edge>& ctx) {
      auto& oeData = g.getEdgeData(g.findEdgeSortedByDst(src, dst));
      auto& ieData = g.getEdgeData(g.findEdgeSortedByDst(dst, src));

      auto newSupport = (oeData >> 1) - 1;
      oeData = (newSupport << 1);
      ieData = (newSupport << 1);
      if (newSupport < j) {
        ctx.push(std::make_pair(src, dst));
      }
    }

    void operator()(Edge e, Galois::UserContext<Edge>& ctx) {
      auto src = e.first, dst = e.second;

      // lock nodes within 2 hops from src
      auto& oeData = g.getEdgeData(g.findEdgeSortedByDst(src, dst));
      for (auto ei: g.edges(src)) {
        g.edges(g.getEdgeDst(ei));
      }

      // lock nodes within 2 hops from dst
      auto& ieData = g.getEdgeData(g.findEdgeSortedByDst(dst, src));
      for (auto ei: g.edges(dst)) {
        g.edges(g.getEdgeDst(ei));
      }

      // avoid repeated processing
      if (oeData & removed) {
        return;
      }

      // mark as removed
      oeData = removed;
      ieData = removed;

      // propagate edge removal
      std::deque<GNode> commonNeighbors = getValidCommonNeighbors(g, src, dst);
      for (auto n: commonNeighbors) {
        removeUnsupportedEdge(((n < src) ? n : src), ((n < src) ? src: n), ctx);
        removeUnsupportedEdge(((n < dst) ? n : dst), ((n < dst) ? dst: n), ctx);
      }
    }
  };

  void operator()(Graph& g, unsigned int k) {
    if (0 == k-2) {
      return;
    }

    EdgeVec work, unsupported;

    // symmetry breaking: 
    // consider only edges (i, j) where i < j
    Galois::do_all_local(g, 
      [&g, &work] (GNode n) {
        for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
          auto dst = g.getEdgeDst(e);
          if (dst > n) {
            work.push_back(std::make_pair(n, dst));
          }
        }
      },
      Galois::do_all_steal<true>()
    );

    Galois::do_all_local(work, 
      PickUnsupportedEdges{g, k-2, unsupported},
      Galois::do_all_steal<true>()
    );

    Galois::for_each_local(unsupported,
      PropagateEdgeRemoval{g, k-2},
      Galois::loopname("PropagateEdgeRemoval")
    );
  } // end operator()
}; // end AsyncTrussAlgo

template<typename Algo>
void run() {
  Algo algo;
  typename Algo::Graph g;

  Galois::reportPageAlloc("MeminfoPre");

  Galois::Graph::readGraph(g, filename);
  std::cout << "Read " << g.size() << " nodes" << std::endl;

  initialize(g);
//  printGraph(g);

  std::cout << "Running " << algo.name() << " algorithm for maximal " << trussNum << "-truss" << std::endl;

  Galois::StatTimer T;
  T.start();
  algo(g, trussNum);
  T.stop();
  Galois::reportPageAlloc("MeminfoPost");
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
  case bspJacobi: 
    run<BSPTrussJacobiAlgo>(); 
    break;
  case bsp:
    run<BSPTrussAlgo>();
    break;
  case bspCoreThenTruss:
    run<BSPCoreThenTrussAlgo>();
    break;
  case async: 
    run<AsyncTrussAlgo>(); 
    break;
  case asyncCoreThenTruss:
//    run<AsyncCoreThenTrussAlgo>();
    break;
  default: 
    std::cerr << "Unknown algorithm\n"; 
    abort();
  }
  T.stop();

  return 0;
}
