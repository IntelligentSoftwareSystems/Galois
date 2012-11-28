/** Connected components -*- C++ -*-
 * @file
 *
 * A simple spanning tree algorithm to demostrate the Galois system.
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
#include "Galois/Graphs/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "boost/optional.hpp"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

const char* name = "Connected Components";
const char* desc = "Compute connected components of a graph";
const char* url = 0;

enum Algo {
  serial,
  asynchronous,
  synchronous
};

enum WriteType {
  none,
  largest
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional, cll::desc("[output file]"));
static cll::opt<WriteType> writeType(cll::desc("Output type:"),
    cll::values(
      clEnumVal(none, "None"),
      clEnumVal(largest, "Write largest component"),
      clEnumValEnd), cll::init(none));
static cll::opt<Algo> algo(cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(serial, "Serial"),
      clEnumVal(asynchronous, "Asynchronous"),
      clEnumVal(synchronous, "Synchronous"),
      clEnumValEnd), cll::init(synchronous));

struct Node {
  unsigned int id;
  Node* component;
};

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "[id: " << n.id << ", c: " << n.component->id << "]";
  return os;
}


#ifdef GALOIS_USE_NUMA
typedef Galois::Graph::LC_Numa_Graph<Node,void> Graph;
#else
typedef Galois::Graph::LC_CSR_Graph<Node,void> Graph;
#endif

typedef Graph::GraphNode GNode;

Graph graph;

/** 
 * Serial connected component algorithm. Just use union-find.
 */
struct SerialAlgo {
  // Not exactly union-by-rank because this ``rank'' is a lower-bound on rank
  static std::pair<int,Node*> find(Node* x) {
    int rank = 0;
    Node* rep;
    for (rep = x; rep->component != rep; rep = rep->component, ++rank)
      ;
    while (x != rep) {
      Node* next = x->component;
      x->component = rep;
      x = next;
    }
    return std::make_pair(rank, rep);
  }

  static void merge(Node* a, Node* b) {
    std::pair<int,Node*> aa = find(a);
    std::pair<int,Node*> bb = find(b);
    if (aa.second != bb.second) {
      if (aa.first > bb.first) {
        boost::swap(aa, bb);
      }
      aa.second->component = bb.second;
    }
  }

  struct Merge {
    void operator()(const GNode& src) const {
      Node& sdata = graph.getData(src, Galois::NONE);
      
      for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::NONE),
          ei = graph.edge_end(src, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, Galois::NONE);
        merge(&sdata, &ddata);
      }
    }
  };

  //! Normalize component by doing find with path compression
  struct Normalize {
    void operator()(const GNode& src) const {
      Node& sdata = graph.getData(src, Galois::NONE);
      find(&sdata);
    }
  };

  void initialize() { }

  void operator()() {
    Galois::do_all_local(graph, Merge());
    Galois::do_all_local(graph, Normalize());
  }
};

/**
 * Synchronous connected component algorithm.  Initially all nodes are in
 * their own component. Then, we merge endpoints of edges to form the spanning
 * tree. Merging is done in two phases to simplify concurrent updates: (1)
 * find components and (2) union components.  Since the merge phase does not
 * do any finds, we only process a fraction of edges at a time; otherwise,
 * the union phase may unnecessarily merge two endpoints in the same
 * component.
 */
struct SynchronousAlgo {
  struct Edge {
    GNode src;
    Node* ddata;
    int count;
    Edge(GNode src, Node* ddata, int count): src(src), ddata(ddata), count(count) { }
  };

  Galois::InsertBag<Edge> wls[2];
  Galois::InsertBag<Edge>* next;
  Galois::InsertBag<Edge>* cur;

  static Node* find(Node* x, bool compress) {
    Node* rep;
    for (rep = x; rep->component != rep; rep = rep->component)
      ;
    if (compress) { 
      while (x != rep) {
        Node* next = x->component;
        x->component = rep;
        x = next;
      }
    }
    return rep;
  }

  //! Lock-free merge. Returns if merge was done.
  static bool merge(Node* a, Node* b) {
    while (true) {
      a = find(a, false);
      b = find(b, false);
      if (a == b)
        return false;
      // Avoid cycles by directing edges consistently
      if (a->id > b->id)
        boost::swap(a, b);
      if (__sync_bool_compare_and_swap(&a->component, a, b))
        return true;
    }
  }

  struct Initialize {
    Galois::InsertBag<Edge>& next;
    Initialize(Galois::InsertBag<Edge>& next): next(next) { }

    //! Add the first edge between components to the worklist
    void operator()(const GNode& src) const {
      for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::NONE),
          ei = graph.edge_end(src, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, Galois::NONE);
        next.push(Edge(src, &ddata, 0));
        break;
      }
    }
  };

  struct Merge {
    Galois::Statistic& emptyMerges;
    Merge(Galois::Statistic& e): emptyMerges(e) { }

    void operator()(const Edge& edge) const {
      Node& sdata = graph.getData(edge.src, Galois::NONE);
      if (!merge(&sdata, edge.ddata))
        emptyMerges += 1;
    }
  };

  struct Find {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_parallel_push;
    typedef int tt_does_not_need_stats;

    Galois::InsertBag<Edge>& next;
    Find(Galois::InsertBag<Edge>& next): next(next) { }

    //! Add the next edge between components to the worklist
    void operator()(const Edge& edge, Galois::UserContext<Edge>&) const {
      (*this)(edge);
    }

    void operator()(const Edge& edge) const {
      GNode src = edge.src;
      Node& sdata = graph.getData(src, Galois::NONE);
      Node* scomponent = find(&sdata, true);
      Graph::edge_iterator ii = graph.edge_begin(src, Galois::NONE);
      Graph::edge_iterator ei = graph.edge_end(src, Galois::NONE);
      int count = edge.count + 1;
      std::advance(ii, count);
      for (; ii != ei; ++ii, ++count) {
        GNode dst = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, Galois::NONE);
        Node* dcomponent = find(&ddata, true);
        if (scomponent != dcomponent) {
          next.push(Edge(src, dcomponent, count));
          break;
        }
      }
    }
  };

  //! Normalize component by doing find with path compression
  struct Normalize {
    void operator()(const GNode& src) const {
      Node& sdata = graph.getData(src, Galois::NONE);
      sdata.component = find(&sdata, true);
    }
  };

  void initialize() { 
    cur = &wls[0];
    next = &wls[1];
    Galois::do_all_local(graph, Initialize(*cur));
  }

  void operator()() {
    Galois::Statistic rounds("Rounds");
    Galois::Statistic emptyMerges("EmptyMerges");

    while (!cur->empty()) {
      Galois::do_all_local(*cur, Merge(emptyMerges));
      Galois::for_each_local(*cur, Find(*next));
      cur->clear();
      boost::swap(cur, next);
      rounds += 1;
    }
    Galois::do_all_local(graph, Normalize());
  }
};

/**
 * Like synchronous algorithm, but if we restrict path compression, we
 * can perform unions and finds concurrently.
 */
struct AsynchronousAlgo {
  static Node* find(Node* x, bool compress) {
    // Basic outline of race in synchronous path compression is that two path
    // compressions along two different paths to the root can create a cycle
    // in the union-find tree. Prevent that from happening by compressing
    // incrementally.
    Node* rep = x;
    Node* prev = 0;
    int rank = 0;
    while (rep->component != rep) {
      Node* next = rep->component;

      if (compress) {
        if (prev && prev->component == rep)
          prev->component = next;
        prev = rep;
      }

      rep = next;
      ++rank;
    }
    return rep;
  }

  //! Lock-free merge. Returns if merge was done.
  static bool merge(Node* a, Node* b) {
    while (true) {
      a = find(a, true);
      b = find(b, true);
      if (a == b)
        return false;
      // Avoid cycles by directing edges consistently
      if (a->id > b->id)
        boost::swap(a, b);
      if (__sync_bool_compare_and_swap(&a->component, a, b))
        return true;
    }
  }

  struct Merge {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_parallel_push;
    typedef int tt_does_not_need_stats;

    Galois::Statistic& emptyMerges;
    Merge(Galois::Statistic& e): emptyMerges(e) { }

    //! Add the next edge between components to the worklist
    void operator()(const GNode& src, Galois::UserContext<GNode>&) const {
      (*this)(src);
    }

    void operator()(const GNode& src) const {
      Node& sdata = graph.getData(src, Galois::NONE);

      for (Graph::edge_iterator ii = graph.edge_begin(src, Galois::NONE),
          ei = graph.edge_end(src, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, Galois::NONE);
        if (!merge(&sdata, &ddata))
          emptyMerges += 1;
      }
    }
  };

  //! Normalize component by doing find with path compression
  struct Normalize {
    void operator()(const GNode& src) const {
      Node& sdata = graph.getData(src, Galois::NONE);
      sdata.component = find(&sdata, true);
    }
  };

  void initialize() { }

  void operator()() {
    Galois::Statistic emptyMerges("EmptyMerges");
    Galois::for_each_local(graph, Merge(emptyMerges));
    Galois::do_all_local(graph, Normalize());
  }
};

struct is_bad {
  bool operator()(const GNode& n) const {
    Node& me = graph.getData(n);
    for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      if (data.component != me.component) {
        std::cerr << "not in same component: " << me << " and " << data << "\n";
        return true;
      }
    }
    return false;
  }
};

bool verify() {
  return Galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad()) == graph.end();
}

void writeComponent(Node* component) {
  // id == 1 if node is in component
  size_t numEdges = 0;
  size_t numNodes = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    Node& data = graph.getData(*ii);
    data.id = data.component == component ? 1 : 0;
    if (data.id) {
      size_t degree = 
        std::distance(graph.edge_begin(*ii, Galois::NONE), graph.edge_end(*ii, Galois::NONE));
      numEdges += degree;
      numNodes += 1;
    }
  }

  typedef Galois::Graph::FileGraphParser Parser;
  Parser p;
  p.setNumNodes(numNodes);
  p.setNumEdges(numEdges);

  p.phase1();
  // partial sums of ids: id == new_index + 1
  Node* prev = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    Node& data = graph.getData(*ii);
    if (prev)
      data.id = prev->id + data.id;
    if (data.component == component) {
      size_t degree = 
        std::distance(graph.edge_begin(*ii, Galois::NONE), graph.edge_end(*ii, Galois::NONE));
      size_t sid = data.id - 1;
      assert(sid < numNodes);
      p.incrementDegree(sid, degree);
    }
    
    prev = &data;
  }

  assert(!prev || prev->id == numNodes);

  p.phase2();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    Node& data = graph.getData(*ii);
    if (data.component != component)
      continue;

    size_t sid = data.id - 1;

    for (Graph::edge_iterator jj = graph.edge_begin(*ii, Galois::NONE),
        ej = graph.edge_end(*ii, Galois::NONE); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      Node& ddata = graph.getData(dst, Galois::NONE);
      size_t did = ddata.id - 1;

      assert(ddata.component == component);
      assert(sid < numNodes && did < numNodes);
      p.addNeighbor(sid, did);
    }
  }

  p.finish(0);

  std::cout << "Writing largest component to " << outputFilename
    << " (nodes: " << numNodes << " edges: " << numEdges << ")\n";

  p.structureToFile(outputFilename.c_str());
}

struct CountLargest {
  typedef std::map<Node*,int> Map;

  struct Accums {
    Galois::GMapElementAccumulator<Map> map;
    Galois::GAccumulator<size_t> trivial;
  };

  Accums& accums;
  
  CountLargest(Accums& accums): accums(accums) { }
  
  void operator()(const GNode& x) {
    Node& n = graph.getData(x, Galois::NONE);
    // Ignore trivial components
    if (&n == n.component) {
      accums.trivial += 1;
      return;
    }

    accums.map.update(n.component, 1);
  }
};

struct ComponentSizePair {
  Node* component;
  int size;

  struct Max {
    ComponentSizePair operator()(const ComponentSizePair& a, const ComponentSizePair& b) const {
      if (a.size > b.size)
        return a;
      return b;
    }
  };

  ComponentSizePair(): component(0), size(0) { }
  ComponentSizePair(Node* c, int s): component(c), size(s) { }
};

struct ReduceMax {
  typedef Galois::GSimpleReducible<ComponentSizePair,ComponentSizePair::Max> Accum;

  Accum& accum;

  ReduceMax(Accum& accum): accum(accum) { }

  void operator()(const std::pair<Node*,int>& x) {
    accum.update(ComponentSizePair(x.first, x.second));
  }
};

// XXX: Good example of need to fix current system for reductions
Node* findLargest() {
  CountLargest::Accums accums;
  Galois::do_all(graph.begin(), graph.end(), CountLargest(accums));
  CountLargest::Map& map = accums.map.reduce();
  size_t trivialComponents = accums.trivial.reduce();

  ReduceMax::Accum accumMax;
  Galois::do_all(map.begin(), map.end(), ReduceMax(accumMax));
  ComponentSizePair& largest = accumMax.reduce();

  // Componsate for dropping trivial entries of components
  double ratio = graph.size() - trivialComponents + map.size();
  size_t largestSize = largest.size + 1;
  if (ratio)
    ratio = largestSize / ratio;

  std::cout << "Number of components: " << map.size() 
    << " (largest: " << largestSize << " [" << ratio << "])\n";

  return largest.component;
}

template<typename Algo>
void run() {
  Algo algo;

  Galois::StatTimer Tinitial("AlgoInitializeTime");
  Tinitial.start();
  algo.initialize();
  Tinitial.stop();

  Galois::StatTimer T;
  T.start();
  algo();
  T.stop();

  if (!skipVerify || writeType == largest) {
    Node* component = findLargest();
    if (!verify()) {
      std::cerr << "verification failed\n";
      assert(0 && "verification failed");
      abort();
    }
    if (writeType == largest && component) {
      writeComponent(component);
    }
  }
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();
  graph.structureFromFile(inputFilename);

  unsigned int id = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii, ++id) {
    Node& n = graph.getData(*ii);
    n.id = id;
    n.component = &n;
  }
  Tinitial.stop();
  
  // XXX Test if preallocation matters
  //Galois::preAlloc(numThreads);
  Galois::Statistic("MeminfoPre", GaloisRuntime::MM::pageAllocInfo());
  switch (algo) {
    case serial: run<SerialAlgo>(); break;
    case synchronous: run<SynchronousAlgo>(); break;
    case asynchronous: run<AsynchronousAlgo>(); break;
    default: std::cerr << "Unknown algo: " << algo << "\n";
  }
  Galois::Statistic("MeminfoPost", GaloisRuntime::MM::pageAllocInfo());

  return 0;
}
