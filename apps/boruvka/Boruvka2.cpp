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
#include "Galois/Statistic.h"
#include "Galois/UnionFind.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <algorithm>
#include <iostream>

namespace cll = llvm::cl;

const char* name = "Boruvka's Minimum Spanning Tree Algorithm";
const char* desc = "Compute the minimum spanning forest of a graph";
const char* url = NULL;

enum Algo {
  serial,
  parallel
};

static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Graph already symmetric"), cll::init(false));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(serial, "Serial"),
      clEnumVal(parallel, "Parallel"),
      clEnumValEnd), cll::init(parallel));

typedef int EdgeData;

struct Node: public Galois::UnionFindNode<Node> {
  EdgeData lightest;
};

#ifdef GALOIS_USE_NUMA
typedef Galois::Graph::LC_Numa_Graph<Node,EdgeData> Graph;
#else
typedef Galois::Graph::LC_CSR_Graph<Node,EdgeData> Graph;
#endif

typedef Graph::GraphNode GNode;

Graph graph;

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "[id: " << &n << ", c: " << n.find() << "]";
  return os;
}

struct Edge {
  GNode src;
  GNode dst;
  EdgeData weight;
  Edge(const GNode& s, const GNode& d, const EdgeData& w): src(s), dst(d), weight(w) { }
};

Galois::InsertBag<Edge> mst;

/**
 * Boruvka's algorithm. Implemented bulk-synchronously in order to avoid the
 * need to merge edge lists.
 */
struct ParallelAlgo {
  static const bool useEagerFiltering = false;

  struct WorkItem {
    Edge edge;
    int cur;
    WorkItem(const GNode& s, const GNode& d, const EdgeData& w, int c): edge(s, d, w), cur(c) { }
  };

  typedef Galois::InsertBag<WorkItem> WL;

  WL wls[2];
  WL* current;
  WL* next;
  Galois::Statistic filtered;

  ParallelAlgo(): filtered("FilteredUpdates") { }

  /**
   * Find lightest edge between components leaving a node and add it to the
   * worklist.
   */
  static bool findLightest(const GNode& src, int cur, WL& next, bool updateLightest) {
    Node& sdata = graph.getData(src, Galois::NONE);
    Graph::edge_iterator ii = graph.edge_begin(src, Galois::NONE);
    Graph::edge_iterator ei = graph.edge_end(src, Galois::NONE);

    std::advance(ii, cur);

    for (; ii != ei; ++ii, ++cur) {
      GNode dst = graph.getEdgeDst(ii);
      Node& ddata = graph.getData(dst, Galois::NONE);
      Node* rep;
      if ((rep = sdata.findAndCompress()) != ddata.findAndCompress()) {
        int weight = graph.getEdgeData(ii);
        next.push(WorkItem(src, dst, weight, cur));
        int old;
        while (updateLightest && weight < (old = rep->lightest)) {
          if (__sync_bool_compare_and_swap(&rep->lightest, old, weight)) {
            break;
          }
        }
        return true;
      }
    }

    return false;
  }

  /**
   * Merge step specialized for first round of the algorithm.
   */
  struct Initialize {
    ParallelAlgo* self;

    Initialize(ParallelAlgo* s): self(s) { }

    void operator()(const GNode& src) const {
      int cur = 0;
      Node& sdata = graph.getData(src, Galois::NONE);
      Graph::edge_iterator ii = graph.edge_begin(src, Galois::NONE);
      Graph::edge_iterator ei = graph.edge_end(src, Galois::NONE);

      sdata.lightest = std::numeric_limits<EdgeData>::max();

      if (ii == ei)
        return;

      GNode dst = graph.getEdgeDst(ii);
      int weight = graph.getEdgeData(ii);
      Node& ddata = graph.getData(dst, Galois::NONE);

      if (sdata.merge(&ddata)) {
        mst.push(Edge(src, dst, weight));
      }

      findLightest(src, cur + 1, *self->next, false);
    }
  };

  struct Merge {
    ParallelAlgo* self;

    Merge(ParallelAlgo* s): self(s) { }

    void operator()(const WorkItem& item) const {
      GNode src = item.edge.src;
      Node& sdata = graph.getData(src, Galois::NONE);
      Node* rep = sdata.findAndCompress();
      int cur = item.cur;

      if (rep->lightest == item.edge.weight) {
        GNode dst = item.edge.dst;
        Node& ddata = graph.getData(dst, Galois::NONE);
        if ((rep = sdata.merge(&ddata))) {
          rep->lightest = std::numeric_limits<EdgeData>::max();
          mst.push(Edge(src, dst, item.edge.weight));
        }
        ++cur;
      }

      if (useEagerFiltering && !findLightest(src, cur, *self->next, false))
        self->filtered += 1;
    }
  };

  struct Find {
    ParallelAlgo* self;

    Find(ParallelAlgo* s): self(s) { }

    void operator()(const WorkItem& item) const {
      // Find true lightest out of each component
      findLightest(item.edge.src, item.cur, *self->next, true);
    }
  };

  void operator()() {
    Galois::Statistic rounds("Rounds");

    current = &wls[0];
    next = &wls[1];

    // TODO switch to serial appropriately
    Galois::do_all_local(graph, Initialize(this));
    while (true) {
      rounds += 1;

      std::swap(current, next);
      Galois::do_all_local(*current, Merge(this));

      if (useEagerFiltering) {
        current->clear();
        std::swap(current, next);
      }

      Galois::do_all_local(*current, Find(this));
      current->clear();

      if (next->empty())
        break;
    }
  }
};

struct SerialAlgo {
  void operator()() {
    // TODO
  }
};

struct is_bad_graph {
  bool operator()(const GNode& n) const {
    Node& me = graph.getData(n);
    for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n); ii != ei; ++ii) {
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

  void operator()(const GNode& n) {
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
  void operator()(const GNode& src) {
    graph.sortEdges(src, std::less<EdgeData>(), Galois::NONE);
  }

  void operator()() {
    Galois::do_all_local(graph, *this);
  }
};

struct get_weight {
  EdgeData operator()(const Edge& e) const { return e.weight; }
};

template<typename Algo>
void run() {
  Algo algo;

  algo();
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
  
  origGraph.structureFromFile(inputFilename.c_str());
  if (!symmetricGraph) 
    Galois::Graph::makeSymmetric<EdgeData>(origGraph, symGraph);
  else
    symGraph.swap(origGraph);

  graph.structureFromGraph(symGraph);
  
  Galois::StatTimer Tsort("InitializeSortTime");
  Tsort.start();
  run<SortEdges>();
  Tsort.stop();

  std::cout << "Nodes: " << graph.size() << ", edges: " << graph.sizeEdges() << "\n";
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();
  initializeGraph();
  Tinitial.stop();

  Galois::preAlloc(GaloisRuntime::MM::pageAllocInfo()* 4);
  Galois::Statistic("MeminfoPre", GaloisRuntime::MM::pageAllocInfo());
  Galois::StatTimer T;
  T.start();
  switch (algo) {
    case serial: run<SerialAlgo>(); break;
    case parallel: run<ParallelAlgo>(); break;
    default: std::cerr << "Unknown algo: " << algo << "\n";
  }
  T.stop();
  Galois::Statistic("MeminfoPost", GaloisRuntime::MM::pageAllocInfo());

  size_t weight = Galois::ParallelSTL::map_reduce(mst.begin(), mst.end(), get_weight(), 0, std::plus<size_t>());
  std::cout << "MST weight: " << weight << "\n";

  if (!skipVerify && !verify()) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  return 0;
}

