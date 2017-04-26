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

namespace cll = llvm::cl;

static const char* name = "Maximal k-trusses";
static const char* desc =
  "Computes the maximal k-trusses for a given undirected graph";
static const char* url = "k_truss";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int> k("k", cll::desc("k for the k-truss"), cll::Required);
static cll::opt<bool> reportAllTheWay("reportAllTheWay", cll::desc("report all maximal p-trusses for p in [2, k]"), cll::init(false));

// undirected graph with sorted neighbors
// read only the lower-triangle or upper-triangle, or the edges are duplicated
typedef Galois::Graph::FirstGraph<unsigned int, unsigned int, false>
  ::template with_sorted_neighbors<true>::type Graph;
typedef Graph::GraphNode GNode;

void initialize(Graph& g) {
  unsigned int i = 0;
  for (auto n: g) {
    g.getData(n, Galois::MethodFlag::UNPROTECTED) = i++;
  }
}

void printGraph(Graph& g) {
  for (auto n: g) {
    std::cout << "node " << g.getData(n) << std::endl;
    for (auto e: g.edges(n)) {
      std::cout << "  edge to " << g.getData(g.getEdgeDst(e)) << std::endl;
    }
  }
}

struct RemoveNodeLessDegreeJ {
  Graph& g;
  unsigned int j;
  RemoveNodeLessDegreeJ(Graph& g, unsigned int j): g(g), j(j) {}

  void operator()(GNode n, Galois::UserContext<GNode>& ctx) {
    if (!g.containsNode(n)) {
      return;
    }

    auto deg = std::distance(g.edge_begin(n), g.edge_end(n));
//    std::cout << "node " << g.getData(n) << ".degree = " << deg << std::endl;
    if (deg >= j) {
      return;
    }

    for (auto e: g.edges(n)) {
      ctx.push(g.getEdgeDst(e));
    }
//    std::cout << "remove non-" << j << "-core node " << g.getData(n) << std::endl;
    g.removeNode(n);
  }
};

void reduce2JCore(Graph& g, unsigned int j) {
  Galois::for_each_local(g, 
    RemoveNodeLessDegreeJ{g, j}, 
    Galois::loopname("reduce2JCore")
  );
}

typedef std::pair<GNode, Graph::edge_iterator> Edge;

// for counting occurences only. no space allocation is required.
template<typename T>
class counter : public std::iterator<std::output_iterator_tag, T> {
  T dummy;
  unsigned int num;
public:
  counter(): num(0) {}
  counter& operator++() { ++num; return *this; }
  counter  operator++(int) { auto retval = *this; ++num; return retval; }
  T& operator*() { return dummy; }
  unsigned int get() { return num; }
};

struct ComputeEdgeSupport {
  Graph& g;
  Galois::InsertBag<Edge>& w;
  unsigned int j;
  ComputeEdgeSupport(Graph& g, Galois::InsertBag<Edge>& w, unsigned int j): g(g), w(w), j(j) {}

  void operator()(GNode n) {
    Galois::MethodFlag unprotected = Galois::MethodFlag::UNPROTECTED;

    for (auto e: g.edges(n, unprotected)) {
      auto dst = g.getEdgeDst(e);

      // symmetry breaking
      if (n > dst) {
        continue;
      }

      using Iter = typename Graph::edge_iterator;
      auto l = [=] (Iter i) { return (this->g).getEdgeDst(i); };
      counter<GNode> support;

      std::set_intersection(
        // Galois::NoDerefIterator lets dereference return the wrapped iterator itself
        // boost::make_transform_iterator gives an iterator, dereferenced to func(in_iter)
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_begin(n)), l),
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_end(n)), l),
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_begin(dst)), l),
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_end(dst)), l),
        support
      );

      g.getEdgeData(e) = support.get();
      if (support.get() < j) {
        w.push_back(std::make_pair(n, e));
      }
    } // end for e
  }
};

Galois::InsertBag<Edge> pickUnsupportedEdges(Graph& g, unsigned int j) {
  Galois::InsertBag<Edge> unsupported;

  Galois::do_all_local(g, 
    ComputeEdgeSupport{g, unsupported, j},
    Galois::do_all_steal<true>()
  );

  return unsupported;
}

struct RemoveEdgeLessSupportJ {
  Graph& g;
  unsigned int j;
  RemoveEdgeLessSupportJ(Graph& g, unsigned int j): g(g), j(j) {}

  void pickUnsupported(GNode n1, GNode n2, Galois::UserContext<Edge>& ctx) {
    auto e = g.findEdge(n1, n2);
    if (e == g.edge_end(n1)) {
      return;
    }

    auto& support = g.getEdgeData(e);
    support--;
    if (support < j) {
      auto src = (n1 < n2) ? n1 : n2;
      auto dst = (src == n1) ? n2 : n1;
      ctx.push(std::make_pair(src, g.findEdge(src, dst)));
    }
  }

  void operator()(Edge e, Galois::UserContext<Edge>& ctx) {
    auto src = e.first, dst = (e.second)->first();
    auto ei = g.findEdge(src, dst);
    if (ei == g.edge_end(src)) {
      // e is invalid
      return;
    }

    using Iter = typename Graph::edge_iterator;
    auto l = [=] (Iter i) { return (this->g).getEdgeDst(i); };
    std::deque<GNode, Galois::PerIterAllocTy::rebind<GNode>::other> 
      support(ctx.getPerIterAlloc());

    std::set_intersection(
      // Galois::NoDerefIterator lets dereference return the wrapped iterator itself
      // boost::make_transform_iterator gives an iterator, dereferenced to func(in_iter)
      boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_begin(src)), l),
      boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_end(src)), l),
      boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_begin(dst)), l),
      boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_end(dst)), l),
      std::back_inserter(support)
    );

    g.getEdgeData(ei) = support.size();
    if (support.size() >= j) {
      return;
    }

    for (auto n: support) {
      pickUnsupported(src, n, ctx);
      pickUnsupported(dst, n, ctx);
    }
//    std::cout << "remove edge " << src << " -> " << dst << std::endl;
    g.removeEdge(src, ei);
  }
};

void removeUnsupportedEdges(Graph& g, unsigned int j) {
  if (0 == j) {
    return;
  }

  Galois::InsertBag<Edge> unsupported = pickUnsupportedEdges(g, j); 

  Galois::for_each_local(unsupported, 
    RemoveEdgeLessSupportJ{g, j}, 
    Galois::loopname("removeUnsupportedEdges"),
    Galois::needs_per_iter_alloc<>()
  );
}

void removeIsolatedVertices(Graph& g) {
  Galois::do_all_local(g, 
    [&g] (GNode n) { 
      if (0 == std::distance(g.edge_begin(n), g.edge_end(n))) { 
//        std::cout << "remove isolated node " << g.getData(n) << std::endl;
        g.removeNode(n); 
      } 
    },
    Galois::do_all_steal<true>()
  );
}

void reportJTruss(Graph& g, unsigned int j) {
  std::string outName = std::to_string(j) + "-truss.txt";
  std::ofstream of(outName);
  Galois::MethodFlag unprotected = Galois::MethodFlag::UNPROTECTED;
  for (auto n: g) {
    auto id = g.getData(n, unprotected);
    for (auto e: g.edges(n, unprotected)) {
      auto dst = g.getEdgeDst(e);
      auto dstId = g.getData(dst, unprotected);
      if (id < dstId) {
        of << id << " " << dstId << std::endl;
      }
    }
  }
}

// Algorithm from the following reference: 
//   Jonathan Cohen. Trusses: cohesive subgraphs for social netowrk analysis.
//   National Security Agency Technical Report, page 16, 2008.
void reduce2KTruss(Graph& g, unsigned int k) {
  reduce2JCore(g, k-1);
  removeUnsupportedEdges(g, k-2);
  removeIsolatedVertices(g);
  reportJTruss(g, k);
}

void reduce2KTrussAllTheWay(Graph& g, unsigned int k) {
  for (unsigned int i = 2; i <= k; ++i) {
    reduce2KTruss(g, i);
  }
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if(k < 2) {
    std::cerr << "k must be >= 2" << std::endl;
    return -1;
  }

  Galois::StatTimer T("TotalTime");
  T.start();
  Graph g;
  Galois::Graph::readGraph(g, filename);
  initialize(g);
//  printGraph(g);
  if (reportAllTheWay) {
    reduce2KTrussAllTheWay(g, k);
  } else {
    reduce2KTruss(g, k);
  }
  T.stop();

  return 0;
}
