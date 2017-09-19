/** Spanning-tree application -*- C++ -*-
 * @file
 *
 * A minimum spanning tree algorithm to demonstrate the Galois system.
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
#include "Galois/Timer.h"
#include "Galois/UnionFind.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"

#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/BulkSynchronousWork.h"
#endif

#include "Lonestar/BoilerPlate.h"

#include <atomic>
#include <utility>
#include <algorithm>
#include <iostream>

namespace cll = llvm::cl;

static const char* name = "Boruvka's Minimum Spanning Tree Algorithm";
static const char* desc = "Computes the minimum spanning forest of a graph";
static const char* url = "mst";

enum Algo {
  parallel,
  exp_parallel
};

static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Graph already symmetric"), cll::init(false));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(parallel, "Parallel"),
#ifdef GALOIS_USE_EXP
      clEnumVal(exp_parallel, "Parallel (exp)"),
#endif
      clEnumValEnd), cll::init(parallel));

typedef int EdgeData;

struct Node: public Galois::UnionFindNode<Node> {
  std::atomic<EdgeData*> lightest;
  Node(): Galois::UnionFindNode<Node>(const_cast<Node*>(this)) { }
};

typedef Galois::Graph::LC_CSR_Graph<Node,EdgeData>
  ::with_numa_alloc<true>::type
  ::with_no_lockable<true>::type Graph;

typedef Graph::GraphNode GNode;

Graph graph;

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "[id: " << &n << ", c: " << n.find() << "]";
  return os;
}

struct Edge {
  GNode src;
  GNode dst;
  const EdgeData* weight;
  Edge(const GNode& s, const GNode& d, const EdgeData* w): src(s), dst(d), weight(w) { }
};

Galois::InsertBag<Edge> mst;
EdgeData inf;
EdgeData heaviest;

/**
 * Boruvka's algorithm. Implemented bulk-synchronously in order to avoid the
 * need to merge edge lists.
 */
template<bool useExp>
struct ParallelAlgo {
  struct WorkItem {
    Edge edge;
    int cur;
    WorkItem(const GNode& s, const GNode& d, const EdgeData* w, int c): edge(s, d, w), cur(c) { }
  };

  typedef Galois::InsertBag<WorkItem> WL;

  WL wls[3];
  WL* current;
  WL* next;
  WL* pending;
  EdgeData limit;

  /**
   * Find lightest edge between components leaving a node and add it to the
   * worklist.
   */
  template<bool useLimit, typename Context, typename Pending>
  static void findLightest(ParallelAlgo* self,
      const GNode& src, int cur, Context& ctx, Pending& pending) {
    Node& sdata = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
    Graph::edge_iterator ii = graph.edge_begin(src, Galois::MethodFlag::UNPROTECTED);
    Graph::edge_iterator ei = graph.edge_end(src, Galois::MethodFlag::UNPROTECTED);

    std::advance(ii, cur);

    for (; ii != ei; ++ii, ++cur) {
      GNode dst = graph.getEdgeDst(ii);
      Node& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
      EdgeData& weight = graph.getEdgeData(ii);
      if (useLimit && weight > self->limit) {
        pending.push(WorkItem(src, dst, &weight, cur));
        return;
      }
      Node* rep;
      if ((rep = sdata.findAndCompress()) != ddata.findAndCompress()) {
        //const EdgeData& weight = graph.getEdgeData(ii);
        EdgeData* old;
        ctx.push(WorkItem(src, dst, &weight, cur));
        while (weight < *(old = rep->lightest)) {
          if (rep->lightest.compare_exchange_strong(old, &weight))
            break;
        }
        return;
      }
    }
  }

  /**
   * Merge step specialized for first round of the algorithm.
   */
  struct Initialize {
    ParallelAlgo* self;

    Initialize(ParallelAlgo* s): self(s) { }

    void operator()(const GNode& src) const {
      (*this)(src, *self->next, *self->pending);
    }

    template<typename Context>
    void operator()(const GNode& src, Context& ctx) const {
      (*this)(src, ctx, *self->pending);
    }

    template<typename Context,typename Pending>
    void operator()(const GNode& src, Context& ctx, Pending& pending) const {
      Node& sdata = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
      sdata.lightest = &inf;
      findLightest<false>(self, src, 0, ctx, pending);
    }
  };

  struct Merge {
    // NB: tells do_all_bs this operator implicitly calls ctx.push(x) for each
    // call to (*this)(x);
    typedef int tt_does_not_need_push;

    ParallelAlgo* self;

    Merge(ParallelAlgo* s): self(s) { }

    void operator()(const WorkItem& item) const {
      (*this)(item, *self->next, *self->pending);
    }

    template<typename Context>
    void operator()(const WorkItem& item, Context& ctx) const {
      (*this)(item, ctx, *self->pending);
    }

    template<typename Context, typename Pending>
    void operator()(const WorkItem& item, Context& ctx, Pending& pending) const {
      GNode src = item.edge.src;
      Node& sdata = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
      Node* rep = sdata.findAndCompress();
      int cur = item.cur;

      if (rep->lightest == item.edge.weight) {
        GNode dst = item.edge.dst;
        Node& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
        if ((rep = sdata.merge(&ddata))) {
          rep->lightest = &inf;
          mst.push(Edge(src, dst, item.edge.weight));
        }
        ++cur;
      }
    }
  };

  struct Find {
    ParallelAlgo* self;

    Find(ParallelAlgo* s): self(s) { }

    void operator()(const WorkItem& item) const {
      (*this)(item, *self->next, *self->pending);
    }

    template<typename Context>
    void operator()(const WorkItem& item, Context& ctx) const {
      (*this)(item, ctx, *self->pending);
    }

    template<typename Context, typename Pending>
    void operator()(const WorkItem& item, Context& ctx, Pending& pending) const {
      findLightest<true>(self, item.edge.src, item.cur, ctx, pending);
    }
  };

  void init() {
    current = &wls[0];
    next = &wls[1];
    pending = &wls[2];

    EdgeData delta = std::max(heaviest / 5, 1);
    limit = delta;
  }

  void process() {
    Galois::Statistic rounds("Rounds");

    init();

    Galois::do_all_local(graph, Initialize(this));
    while (true) {
      while (true) {
        rounds += 1;

        std::swap(current, next);
        Galois::do_all_local(*current, Merge(this));
        Galois::do_all_local(*current, Find(this));
        current->clear();

        if (next->empty())
          break;
      }

      if (pending->empty())
        break;

      std::swap(next, pending);

      limit *= 2;
    }
  }

#if defined(GALOIS_USE_EXP) && !defined(GALOIS_HAS_NO_BULKSYNCHRONOUS_EXECUTOR)
  void processExp() {
    typedef boost::fusion::vector<WorkItem,WorkItem> Items;

    init();

    Galois::do_all_bs_local<Items>(graph,
        boost::fusion::make_vector(Merge(this), Find(this)),
        Initialize(this));

    while (!pending->empty()) {
      std::swap(next, pending);

      Galois::do_all_bs_local<Items>(*next,
          boost::fusion::make_vector(Merge(this), Find(this)));

      next->clear();

      limit *= 2;
    }
  }
#else
  void processExp() { GALOIS_DIE("not supported"); }
#endif

  void operator()() {
    if (useExp) {
      processExp();
    } else {
      process();
    }
  }
};

struct is_bad_graph {
  bool operator()(const GNode& n) const {
    Node& me = graph.getData(n);
    for (auto ii : graph.edges(n)) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      if (me.findAndCompress() != data.findAndCompress()) {
        std::cerr << "not in same component: " << me << " and " << data << "\n";
        return true;
      }
    }
    return false;
  }
};

struct is_bad_mst {
  bool operator()(const Edge& e) const {
    return graph.getData(e.src).findAndCompress() != graph.getData(e.dst).findAndCompress();
  }
};

struct CheckAcyclic {
  struct Accum {
    Galois::GAccumulator<unsigned> roots;
  };

  Accum* accum;

  void operator()(const GNode& n) const {
    Node& data = graph.getData(n);
    if (data.isRep())
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

struct SortEdges {
  struct Accum {
    Galois::GReduceMax<EdgeData> heavy;
  };

  Accum* accum;

  void operator()(const GNode& src) const {
    //! [sortEdgeByEdgeData]
    graph.sortEdgesByEdgeData(src, std::less<EdgeData>(), Galois::MethodFlag::UNPROTECTED);
    //! [sortEdgeByEdgeData]

    Graph::edge_iterator ii = graph.edge_begin(src, Galois::MethodFlag::UNPROTECTED);
    Graph::edge_iterator ei = graph.edge_end(src, Galois::MethodFlag::UNPROTECTED);
    ptrdiff_t dist = std::distance(ii, ei);
    if (dist == 0)
      return;
    std::advance(ii, dist - 1);
    accum->heavy.update(graph.getEdgeData(ii));
  }

  EdgeData operator()() {
    Accum a;
    accum = &a;
    Galois::do_all_local(graph, *this);
    return a.heavy.reduce();
  }
};

struct get_weight {
  EdgeData operator()(const Edge& e) const { return *e.weight; }
};

template<typename Algo>
void run() {
  Algo algo;

  return algo();
}

bool verify() {
  if (Galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad_graph()) == graph.end()) {
    if (Galois::ParallelSTL::find_if(mst.begin(), mst.end(), is_bad_mst()) == mst.end()) {
      CheckAcyclic c;
      return c();
    }
  }
  return false;
}

void initializeGraph() {
  Galois::Graph::FileGraph origGraph;
  Galois::Graph::FileGraph symGraph;
  
  origGraph.fromFileInterleaved<EdgeData>(inputFilename);
  if (!symmetricGraph) 
    Galois::Graph::makeSymmetric<EdgeData>(origGraph, symGraph);
  else
    std::swap(symGraph, origGraph);

  Galois::Graph::readGraph(graph, symGraph);
  
  Galois::StatTimer Tsort("InitializeSortTime");
  Tsort.start();
  SortEdges sortEdges;
  heaviest = sortEdges();
  if (heaviest == std::numeric_limits<EdgeData>::max() || 
      heaviest == std::numeric_limits<EdgeData>::min()) {
    GALOIS_DIE("Edge weights of graph out of range");
  }
  inf = heaviest + 1;
  
  Tsort.stop();

  std::cout << "Nodes: " << graph.size()
    << " edges: " << graph.sizeEdges() 
    << " heaviest edge: " << heaviest 
    << "\n";
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();
  initializeGraph();
  Tinitial.stop();

  Galois::preAlloc(Galois::Runtime::numPagePoolAllocTotal() * 10);
  Galois::reportPageAlloc("MeminfoPre");
  Galois::StatTimer T;
  T.start();
  switch (algo) {
    case parallel: run<ParallelAlgo<false> >(); break;
    case exp_parallel: run<ParallelAlgo<true> >(); break;
    default: std::cerr << "Unknown algo: " << algo << "\n";
  }
  T.stop();
  Galois::reportPageAlloc("MeminfoPost");

  std::cout << "MST weight: "
    << Galois::ParallelSTL::map_reduce(mst.begin(), mst.end(),
        get_weight(), 0.0, std::plus<double>())
    << " ("
    << Galois::ParallelSTL::map_reduce(mst.begin(), mst.end(),
        get_weight(), 0UL, std::plus<size_t>())
    << ")\n";

  if (!skipVerify && !verify()) {
    GALOIS_DIE("verification failed");
  }

  return 0;
}

