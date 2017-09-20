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
#include "Galois/Timer.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/TypeTraits.h"
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
  asyncTx,
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
    clEnumValN(Algo::asyncTx, "asyncTx", "Asynchronous with Transactional Semantics"),
    clEnumValN(Algo::async, "async", "Asynchronous"), 
    clEnumValN(Algo::asyncCoreThenTruss, "asyncCoreThenTruss", "Compute k-1 core and then k-truss in async way"),
    clEnumValEnd), cll::init(Algo::bsp));

static const uint32_t valid = 0x0;
static const uint32_t removed = 0x1;

template<typename Graph>
void initialize(Graph& g) {
  g.sortAllEdgesByDst();

  // initializa all edges to valid
  galois::do_all_local(
    g, 
    [&g] (typename Graph::GraphNode N) { 
      for (auto e: g.edges(N, galois::MethodFlag::UNPROTECTED)) {
        g.getEdgeData(e) = valid;
      }
    },
    galois::do_all_steal<true>()
  );
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

  galois::do_all_local(g, 
    [&g, &numNodes] (typename G::GraphNode n) {
      for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        if (!(g.getEdgeData(e) & removed)) {
          numNodes += 1;
          break;
        }
      }
    },
    galois::do_all_steal<true>()
  );

  return numNodes.reduce();
}

template<typename G>
size_t countValidEdges(G& g) {
  galois::GAccumulator<size_t> numEdges;

  galois::do_all_local(g, 
    [&g, &numEdges] (typename G::GraphNode n) {
      for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        if (n < g.getEdgeDst(e) && !(g.getEdgeData(e) & removed)) {
          numEdges += 1;
        }
      }
    },
    galois::do_all_steal<true>()
  );

  return numEdges.reduce();
}
#endif

template<typename Graph>
void reportKTruss(Graph& g) {
  if (outName.empty()) {
    return;
  }

  std::ofstream of(outName);
  if (!of.is_open()) {
    std::cerr << "Cannot open " << outName << " for output." << std::endl;
    return;
  }

  for (auto n: g) {
    for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
      auto dst = g.getEdgeDst(e);
      if (n < dst && (g.getEdgeData(e) & 0x1) != removed) {
        of << n << " " << dst << std::endl;
      }
    }
  }
}

template<typename G>
bool isSupportNoLessThanJ(G& g, typename G::GraphNode src, typename G::GraphNode dst, unsigned int j) {
  size_t numValidEqual = 0;
  auto srcI = g.edge_begin(src, galois::MethodFlag::UNPROTECTED), 
    srcE = g.edge_end(src, galois::MethodFlag::UNPROTECTED), 
    dstI = g.edge_begin(dst, galois::MethodFlag::UNPROTECTED), 
    dstE = g.edge_end(dst, galois::MethodFlag::UNPROTECTED);

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
  typedef galois::graphs::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
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
    if (0 == k-2) {
      return;
    }

    EdgeVec unsupported, work[2];
    EdgeVec *cur = &work[0], *next = &work[1];

    // symmetry breaking: 
    // consider only edges (i, j) where i < j
    galois::do_all_local(g, 
      [&g, cur] (GNode n) {
        for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
          auto dst = g.getEdgeDst(e);
          if (dst > n) {
            cur->push_back(std::make_pair(n, dst));
          }
        }
      },
      galois::do_all_steal<true>()
    );

    while (true) {
      galois::do_all_local(*cur, 
        PickUnsupportedEdges{g, k-2, unsupported, *next},
        galois::do_all_steal<true>()
      );

      if (0 == std::distance(unsupported.begin(), unsupported.end())) {
        break;
      }

      // mark unsupported edges as removed
      galois::do_all_local(unsupported, 
        [&g] (Edge e) {
          g.getEdgeData(g.findEdgeSortedByDst(e.first, e.second)) = removed;
          g.getEdgeData(g.findEdgeSortedByDst(e.second, e.first)) = removed;
        },
        galois::do_all_steal<true>()
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
  typedef galois::graphs::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
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
    if (0 == k-2) {
      return;
    }

    EdgeVec work[2];
    EdgeVec *cur = &work[0], *next = &work[1];
    size_t curSize, nextSize;

    // symmetry breaking: 
    // consider only edges (i, j) where i < j
    galois::do_all_local(g, 
      [&g, cur] (GNode n) {
        for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
          auto dst = g.getEdgeDst(e);
          if (dst > n) {
            cur->push_back(std::make_pair(n, dst));
          }
        }
      },
      galois::do_all_steal<true>()
    );
    curSize = std::distance(cur->begin(), cur->end());

    // remove unsupported edges until no more edges can be removed
    while (true) {
      galois::do_all_local(*cur, 
        KeepSupportedEdges{g, k-2, *next},
        galois::do_all_steal<true>()
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

template<typename G>
bool isValidDegreeNoLessThanJ(G& g, typename G::GraphNode n, unsigned int j) {
  size_t numValid = 0;
  for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
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
  typedef galois::graphs::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef galois::InsertBag<GNode> NodeVec;

  std::string name() { return "bspCore"; }

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
        for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
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

    galois::do_all_local(g, 
      KeepValidNodes{g, k, *next}, 
      galois::do_all_steal<true>()
    );
    nextSize = std::distance(next->begin(), next->end());

    while (curSize != nextSize) {
      cur->clear();
      curSize = nextSize;
      std::swap(cur, next);

      galois::do_all_local(*cur, 
        KeepValidNodes{g, k, *next}, 
        galois::do_all_steal<true>()
      );
      nextSize = std::distance(next->begin(), next->end());
    }
  }
}; // end BSPCoreAlgo

// BSPCoreThenTrussAlgo:
// 1. Reduce the graph to k-1 core
// 2. Compute k-truss from k-1 core
struct BSPCoreThenTrussAlgo {
  // set LSB of an edge weight to indicate the removal of the edge.
  typedef galois::graphs::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() { return "bspCoreThenTruss"; }

  void operator()(Graph& g, unsigned int k) {
    if (0 == k-2) {
      return;
    }

    galois::StatTimer TCore("Reduce_to_(k-1)-core");
    TCore.start();

    BSPCoreAlgo bspCore;
    bspCore(g, k-1);

    TCore.stop();

    galois::StatTimer TTruss("Reduce_to_k-truss");
    TTruss.start();

    BSPTrussAlgo bspTrussIm;
    bspTrussIm(g, k);

    TTruss.stop();
  } // end operator()
}; // end struct BSPCoreThenTrussAlgo

template<typename T>
using PerIterAlloc = typename galois::PerIterAllocTy::rebind<T>::other;

template<typename G>
std::deque<typename G::GraphNode, PerIterAlloc<typename G::GraphNode> >
getValidCommonNeighbors(
  G& g,
  typename G::GraphNode src, typename G::GraphNode dst, 
  galois::PerIterAllocTy& a, galois::MethodFlag flag = galois::MethodFlag::WRITE)
{
  using GNode = typename G::GraphNode;

  auto srcI = g.edge_begin(src, flag), srcE = g.edge_end(src, flag), 
    dstI = g.edge_begin(dst, flag), dstE = g.edge_end(dst, flag);
  std::deque<GNode, PerIterAlloc<GNode> > commonNeighbors(a);

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

// AsyncTrussTxAlgo:
// 1. Compute support for all edges and pick out unsupported ones.
// 2. Remove unsupported edges, decrease the support for affected edges and pick out those becomeing unsupported.
// 3. Repeat 2. until no more unsupported edges are found.
//
// edges update in default Galois sync model, i.e. transactional semantics
struct AsyncTrussTxAlgo {
  // edge weight: (# triangles supported << 1) | removal
  //   set LSB of an edge weight to indicate the removal of the edge.
  //   << 1 to track # triangles an edge supports, 
  //   >> 1 when computing edge supports
  typedef galois::graphs::LC_CSR_Graph<void, uint32_t>
    ::template with_numa_alloc<true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef std::pair<GNode, GNode> Edge;
  typedef galois::InsertBag<Edge> EdgeVec;

  std::string name() { return "asyncTx"; }

  struct PickUnsupportedEdges {
    Graph& g;
    unsigned int j;
    EdgeVec& r;

    PickUnsupportedEdges(Graph& g, unsigned int j, EdgeVec& r)
      : g(g), j(j), r(r) {}

    void operator()(Edge e, galois::UserContext<Edge>& ctx) {
      auto src = e.first, dst = e.second;
      std::deque<GNode, PerIterAlloc<GNode> > commonNeighbors
        = getValidCommonNeighbors(g, src, dst, ctx.getPerIterAlloc(), galois::MethodFlag::UNPROTECTED);
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

    void removeUnsupportedEdge(GNode src, GNode dst, galois::UserContext<Edge>& ctx) {
      auto& oeData = g.getEdgeData(g.findEdgeSortedByDst(src, dst));
      auto& ieData = g.getEdgeData(g.findEdgeSortedByDst(dst, src));

      auto newSupport = (oeData >> 1) - 1;
      oeData = (newSupport << 1);
      ieData = (newSupport << 1);
      if (newSupport < j) {
        ctx.push(std::make_pair(src, dst));
      }
    }

    void operator()(Edge e, galois::UserContext<Edge>& ctx) {
      auto src = e.first, dst = e.second;

      // lock src's neighbors
      auto& oeData = g.getEdgeData(g.findEdgeSortedByDst(src, dst));
      // lock src's neighbors' neighbors for back edges from them to src's neighbors
      for (auto ei: g.edges(src)) {
        g.edges(g.getEdgeDst(ei));
      }

      // lock dst's neighbors
      auto& ieData = g.getEdgeData(g.findEdgeSortedByDst(dst, src));
      // lock dst's neighbors' neighbors for back edge from them to dst's neighbors
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
      std::deque<GNode, PerIterAlloc<GNode> > commonNeighbors
        = getValidCommonNeighbors(g, src, dst, ctx.getPerIterAlloc());
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
    galois::do_all_local(g, 
      [&g, &work] (GNode n) {
        for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
          auto dst = g.getEdgeDst(e);
          if (dst > n) {
            work.push_back(std::make_pair(n, dst));
          }
        }
      },
      galois::do_all_steal<true>()
    );

    galois::for_each_local(work, 
      PickUnsupportedEdges{g, k-2, unsupported},
      galois::loopname("PickUnsupportedEdges"),
      galois::does_not_need_aborts<>(),
      galois::does_not_need_push<>(),
      galois::needs_per_iter_alloc<>()
    );

    galois::for_each_local(unsupported,
      PropagateEdgeRemoval{g, k-2},
      galois::loopname("PropagateEdgeRemoval"),
      galois::needs_per_iter_alloc<>()
    );
  } // end operator()
}; // end AsyncTrussTxAlgo

// AsyncTrussAlgo:
// 1. Compute support for all edges and pick out unsupported ones.
// 2. Remove unsupported edges, decrease the support for affected edges and pick out those becomeing unsupported.
// 3. Repeat 2. until no more unsupported edges are found.
//
// edges update using atomic operations in C++
/*struct AsyncTrussAlgo {
  // edge weight: (# triangles supported << 1) | removal
  //   set LSB of an edge weight to indicate the removal of the edge.
  //   << 1 to track # triangles an edge supports, 
  //   >> 1 when computing edge supports
  typedef galois::graphs::LC_CSR_Graph<void, std::atomic<uint32_t> >
    ::template with_no_lockable<true>::type
    ::template with_numa_alloc<true>::type Graph;
  typedef Graph::GraphNode GNode;

  typedef std::pair<GNode, GNode> Edge;
  typedef galois::InsertBag<Edge> EdgeVec;

  std::string name() { return "async"; }

  struct PickUnsupportedEdges {
    Graph& g;
    unsigned int j;
    EdgeVec& r;

    PickUnsupportedEdges(Graph& g, unsigned int j, EdgeVec& r)
      : g(g), j(j), r(r) {}

    void operator()(Edge e, galois::UserContext<Edge>& ctx) {
      auto src = e.first, dst = e.second;
      std::deque<GNode, PerIterAlloc<GNode> > commonNeighbors
        = getValidCommonNeighbors(g, src, dst, ctx.getPerIterAlloc(), galois::MethodFlag::UNPROTECTED);
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

    void removeUnsupportedEdge(GNode src, GNode dst, galois::UserContext<Edge>& ctx) {
      auto& oeData = g.getEdgeData(g.findEdgeSortedByDst(src, dst));
      auto& ieData = g.getEdgeData(g.findEdgeSortedByDst(dst, src));

      auto newSupport = (oeData >> 1) - 1;
      oeData = (newSupport << 1);
      ieData = (newSupport << 1);
      if (newSupport < j) {
        ctx.push(std::make_pair(src, dst));
      }
    }

    void operator()(Edge e, galois::UserContext<Edge>& ctx) {
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
      std::deque<GNode, PerIterAlloc<GNode> > commonNeighbors
        = getValidCommonNeighbors(g, src, dst, ctx.getPerIterAlloc(), galois::MethodFlag::UNPROTECTED);
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
    galois::do_all_local(g, 
      [&g, &work] (GNode n) {
        for (auto e: g.edges(n, galois::MethodFlag::UNPROTECTED)) {
          auto dst = g.getEdgeDst(e);
          if (dst > n) {
            work.push_back(std::make_pair(n, dst));
          }
        }
      },
      galois::do_all_steal<true>()
    );

    galois::for_each_local(work, 
      PickUnsupportedEdges{g, k-2, unsupported},
      galois::loopname("PickUnsupportedEdges"),
      galois::does_not_need_aborts<>(),
      galois::does_not_need_push<>(),
      galois::needs_per_iter_alloc<>()
    );

    galois::for_each_local(unsupported,
      PropagateEdgeRemoval{g, k-2},
      galois::loopname("PropagateEdgeRemoval"),
      galois::needs_per_iter_alloc<>()
    );
  } // end operator()
}; // end AsyncTrussAlgo
*/
template<typename Algo>
void run() {
  Algo algo;
  typename Algo::Graph g;

  galois::reportPageAlloc("MeminfoPre");

  galois::graphs::readGraph(g, filename);
  std::cout << "Read " << g.size() << " nodes" << std::endl;

  initialize(g);
//  printGraph(g);

  std::cout << "Running " << algo.name() << " algorithm for maximal " << trussNum << "-truss" << std::endl;

  galois::StatTimer T;
  T.start();
  algo(g, trussNum);
  T.stop();
  galois::reportPageAlloc("MeminfoPost");
  reportKTruss(g);
}

int main(int argc, char **argv) {
  galois::StatManager statManager;
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
  case asyncTx:
    run<AsyncTrussTxAlgo>();
    break;
  case async: 
//    run<AsyncTrussAlgo>(); 
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
