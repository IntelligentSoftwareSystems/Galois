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
static cll::opt<unsigned int> toK("toK", cll::desc("report up to toK-trusses"), cll::Required);
static cll::opt<unsigned int> fromK("fromK", cll::desc("report from fromK-trusses"), cll::init(2));
static cll::opt<bool> reportAllTheWay("reportAllTheWay", 
  cll::desc("report all maximal p-trusses for p in [fromK, toK]"), cll::init(false));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"), 
  cll::values(
    clEnumValN(Algo::bsp, "bsp", "Bulk-synchronous parallel (default)"), 
    clEnumValN(Algo::async, "async", "Asynchronous"), 
    clEnumValEnd), cll::init(Algo::bsp));

template<typename Graph>
void initialize(Graph& g) {
  unsigned int i = 0;
  for (auto n: g) {
    g.getData(n, Galois::MethodFlag::UNPROTECTED) = i++;
  }
}

template<typename Graph>
void printGraph(Graph& g) {
  for (auto n: g) {
    std::cout << "node " << g.getData(n) << std::endl;
    for (auto e: g.edges(n)) {
      std::cout << "  edge to " << g.getData(g.getEdgeDst(e)) << std::endl;
    }
  }
}

template<typename Graph>
void reportKTruss(Graph& g, unsigned int k, std::string algoName) {
  std::string outName = algoName + "-" + std::to_string(k) + "-truss.txt";
  std::ofstream of(outName);
  auto unprotected = Galois::MethodFlag::UNPROTECTED;
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
  unsigned int get() { return num; } //FIXME: always return 0 for BSPAlgo. why?
};

template<typename Algo>
struct ListEdge {
  typedef typename Algo::Graph Graph;
  typedef typename Graph::GraphNode GNode;
  typedef typename Algo::Edge Edge;
  typedef typename Algo::EdgeVec EdgeVec;

  Graph& g;
  EdgeVec& w;
  ListEdge(Graph& g, EdgeVec& w): g(g), w(w) {}

  void operator() (const GNode n) const {
    for (auto e: g.edges(n, Galois::MethodFlag::UNPROTECTED)) {
      auto dst = g.getEdgeDst(e);
      // symmetry breaking
      if (n < dst) {
        w.push_back(std::make_pair(n, dst));
      }
    }
  }
};

template<typename Algo>
typename Algo::EdgeVec ListAllEdges(typename Algo::Graph& g) {
  typename Algo::EdgeVec w;
  Galois::do_all_local(g, ListEdge<Algo>{g, w}, Galois::do_all_steal<true>());
  return w;
}

struct AsyncAlgo {
  // undirected graph with sorted neighbors
  // read only the lower-triangle or upper-triangle, or the edges are duplicated
  typedef Galois::Graph::FirstGraph<unsigned int, unsigned int, false>
    ::template with_sorted_neighbors<true>::type Graph;
  typedef Graph::GraphNode GNode;
  typedef std::pair<GNode, GNode> Edge;
  typedef Galois::InsertBag<Edge> EdgeVec;

  std::string name() { return "async"; }

  struct RemoveNodeLessDegreeJ {
    Graph& g;
    unsigned int j;
    RemoveNodeLessDegreeJ(Graph& g, unsigned int j): g(g), j(j) {}

    void operator()(GNode n, Galois::UserContext<GNode>& ctx) {
      if (!g.containsNode(n)) {
        return;
      }

      auto deg = std::distance(g.edge_begin(n), g.edge_end(n));
//      std::cout << "node " << g.getData(n) << ".degree = " << deg << std::endl;
      if (deg >= j) {
        return;
      }

      for (auto e: g.edges(n)) {
        ctx.push(g.getEdgeDst(e));
      }
//      std::cout << "remove non-" << j << "-core node " << g.getData(n) << std::endl;
      g.removeNode(n);
    }
  };

  struct PickUnsupportedEdge {
    Graph& g;
    EdgeVec& r;
    unsigned int j;
    PickUnsupportedEdge(Graph& g, EdgeVec& r, unsigned int j): g(g), r(r), j(j) {}

    void operator() (Edge e) {
      auto unprotected = Galois::MethodFlag::UNPROTECTED;
      auto src = e.first, dst = e.second;
      auto ei = g.findEdge(src, dst);
      if (ei == g.edge_end(src)) {
        return;
      }

      using Iter = typename Graph::edge_iterator;
      auto l = [=] (Iter i) { return (this->g).getEdgeDst(i); };
      counter<GNode> support;

      std::set_intersection(
        // Galois::NoDerefIterator lets dereference return the wrapped iterator itself
        // boost::make_transform_iterator gives an iterator, dereferenced to func(in_iter)
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_begin(src, unprotected)), l),
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_end(src, unprotected)), l),
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_begin(dst, unprotected)), l),
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_end(dst, unprotected)), l),
        support
      );

      g.getEdgeData(ei) = support.get();
      if (support.get() < j) {
        r.push_back(e);
//        std::cout << "edge " << g.getData(src) << " -> " << g.getData(dst) << ": " << g.getEdgeData(e.second) << " < " << j << std::endl;
      }
    }
  };

  struct RemoveEdgeSupportLessThanJ {
    Graph& g;
    unsigned int j;
    RemoveEdgeSupportLessThanJ(Graph& g, unsigned int j): g(g), j(j) {}

    void pickUnsupported (GNode n1, GNode n2, Galois::UserContext<Edge>& ctx) {
      auto e = g.findEdge(n1, n2);
      if (e == g.edge_end(n1)) {
        return;
      }

      auto& support = g.getEdgeData(e);
      support--;
      if (support < j) {
        auto src = (n1 < n2) ? n1 : n2;
        auto dst = (n1 < n2) ? n2 : n1;
        ctx.push(std::make_pair(src, dst));
      }
    }

    void operator() (Edge e, Galois::UserContext<Edge>& ctx) {
      auto src = e.first, dst = e.second;
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

  // Algorithm from the following reference: 
  //   Jonathan Cohen. Trusses: cohesive subgraphs for social netowrk analysis.
  //   National Security Agency Technical Report, page 16, 2008.
  void operator()(Graph& g, unsigned int k) {
    // reduce to k-1 core, e.g. nodes whose degree >= k-1 
    Galois::for_each_local(g, RemoveNodeLessDegreeJ{g, k-1}, Galois::loopname("reduce2JCore"));

    // remove edges whose support < k-2
    EdgeVec unsupported, w = ListAllEdges<AsyncAlgo>(g);
    Galois::do_all_local(w, 
      PickUnsupportedEdge{g, unsupported, k-2}, 
      Galois::do_all_steal<true>());
    Galois::for_each_local(unsupported, 
      RemoveEdgeSupportLessThanJ{g, k-2}, 
      Galois::loopname("removeUnsupportedEdges"), 
      Galois::needs_per_iter_alloc<>());

    // remove isolated nodes
    Galois::do_all_local(g, 
      [&g] (GNode n) { 
        if (0 == std::distance(g.edge_begin(n), g.edge_end(n))) { 
//          std::cout << "remove isolated node " << g.getData(n) << std::endl;
          g.removeNode(n); 
        } 
      },
      Galois::do_all_steal<true>()
    );

    reportKTruss(g, k, name());
  }
}; // end struct AsyncAlgo

struct BSPAlgo {
  // undirected graph with sorted neighbors
  // read only the lower-triangle or upper-triangle, or the edges are duplicated
  typedef Galois::Graph::FirstGraph<unsigned int, void, false>
    ::template with_sorted_neighbors<true>::type
    ::template with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;
  typedef std::pair<GNode, GNode> Edge;
  typedef Galois::InsertBag<Edge> EdgeVec;

  std::string name() { return "bsp"; }

  struct PickUnsupportedEdge {
    Graph& g;
    EdgeVec& r;
    EdgeVec& s;
    unsigned int j;
    PickUnsupportedEdge(Graph& g, EdgeVec& r, EdgeVec& s, unsigned int j): g(g), r(r), s(s), j(j) {}

    void operator() (Edge e) {
      auto unprotected = Galois::MethodFlag::UNPROTECTED;
      auto src = e.first, dst = e.second;
      auto ei = g.findEdge(src, dst, unprotected);
      if (ei == g.edge_end(src)) {
        return;
      }

      using Iter = typename Graph::edge_iterator;
      auto l = [=] (Iter i) { return (this->g).getEdgeDst(i); };
      counter<GNode> support;

      std::set_intersection(
        // Galois::NoDerefIterator lets dereference return the wrapped iterator itself
        // boost::make_transform_iterator gives an iterator, dereferenced to func(in_iter)
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_begin(src, unprotected)), l),
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_end(src, unprotected)), l),
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_begin(dst, unprotected)), l),
        boost::make_transform_iterator(Galois::NoDerefIterator<Iter>(g.edge_end(dst, unprotected)), l),
        support
      );

//      std::cout << "|intersection| = " << support.get() << std::endl;
      if (support.get() < j) {
        r.push_back(e);
//        std::cout << "edge " << g.getData(src) << " -> " << g.getData(dst) << ": " << g.getEdgeData(e.second) << " < " << j << std::endl;
      } else {
        s.push_back(e);
      }
    }
  };

  void operator()(Graph& g, unsigned int k) {
    if (0 == k-2) {
      return;
    }

    EdgeVec work[2], unsupported, *cur, *next;
    work[0] = ListAllEdges<BSPAlgo>(g);
    cur = &work[0];
    next = &work[1];
    
    while (true) {
      // pick out all edges in less than k-2 triangles
      Galois::do_all_local(*cur, 
        PickUnsupportedEdge{g, unsupported, *next, k-2}, 
        Galois::do_all_steal<true>()
      );

      if (!std::distance(unsupported.begin(), unsupported.end())) {
        break;
      }

      auto flag = Galois::MethodFlag::UNPROTECTED;
      for (auto e: unsupported) {
        g.removeEdge(e.first, g.findEdge(e.first, e.second, flag), flag);
      }
      unsupported.clear();
      cur->clear();
      std::swap(cur, next);
    }

    reportKTruss(g, k, name());
  }
}; // end struct BSPAlgo

template<typename Algo>
void run() {
  typedef typename Algo::Graph Graph;

  Algo algo;
  Graph g;

  Galois::Graph::readGraph(g, filename);
  std::cout << "Read " << g.size() << " nodes" << std::endl;

  initialize(g);
//  printGraph(g);

  std::cout << "Running " << algo.name() << " algorithm" << std::endl;

  Galois::StatTimer T;
  T.start();
  if (reportAllTheWay) {
    for (unsigned int i = fromK; i <= toK; ++i) {
      algo(g, i);
    }
  } else {
    algo(g, toK);
  }
  T.stop();
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if (fromK < 2 || toK < fromK) {
    std::cerr << "2 <= fromK <= toK" << std::endl;
    return -1;
  }

  Galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
  case bsp: 
    run<BSPAlgo>(); 
    break;
  case async: 
    run<AsyncAlgo>(); 
    break;
  default: 
    std::cerr << "Unknown algorithm\n"; 
    abort();
  }
  T.stop();

  return 0;
}
