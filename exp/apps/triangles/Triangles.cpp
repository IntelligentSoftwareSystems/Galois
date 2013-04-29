/** Count triangles -*- C++ -*-
 * @file
 *
 * Count the number of triangles in a graph.
 *
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Graphs/Graph3.h"
#include "Galois/Runtime/DistSupport.h"

#include <boost/iterator/transform_iterator.hpp>
#include <Eigen/Dense>
#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

const char* name = "Triangles";
const char* desc = "Count triangles in a graph";
const char* url = 0;

enum Algo {
  nodeiterator,
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::nodeiterator, "nodeiterator", "Node Iterator (default)"),
      clEnumValEnd), cll::init(Algo::nodeiterator));

typedef Galois::Graph::LC_Numa_Graph<uint32_t,void> Graph;
typedef Graph::GraphNode GNode;
Graph graph;

// DistGraph nodes
typedef Galois::Graph::ThirdGraph<uint32_t,void,Galois::Graph::EdgeDirection::Un> DGraph;
typedef DGraph::NodeHandle DGNode;
typedef typename DGraph::pointer Graphp;

std::unordered_map<GNode,DGNode> mapping;

struct element: public Galois::Runtime::Lockable {
  GNode g;
  unsigned v;
  element() { }
  element(GNode _g,unsigned _v): g(_g), v(_v) { }
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,g,v);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,g,v);
  }
};

/**
 * Like std::lower_bound but doesn't dereference iterators. Returns the first element
 * for which comp is not true. 
 */
template<typename Iterator, typename Compare>
Iterator lowerBound(Iterator first, Iterator last, Compare comp) {
  Iterator it;
  typename std::iterator_traits<Iterator>::difference_type count, half;
  count = std::distance(first, last);
  while (count > 0) {
    it = first; half = count / 2; std::advance(it, half);
    if (comp(it)) {
      first = ++it;
      count -= half + 1;
    } else {
      count = half;
    }
  }
  return first;
}

/**
 * std::set_intersection over edge_iterators.
 */
template<typename G>
struct LessThan {
  G& g;
  typename G::GraphNode n;
  LessThan(G& g, typename G::GraphNode n): g(g), n(n) { }
  bool operator()(typename G::edge_iterator it) {
    return g.getEdgeDst(it) < n;
  }
};

template<typename G>
struct GreaterThanOrEqual {
  G& g;
  typename G::GraphNode n;
  GreaterThanOrEqual(G& g, typename G::GraphNode n): g(g), n(n) { }
  bool operator()(typename G::edge_iterator it) {
    return !(n < g.getEdgeDst(it));
  }
};

/**
 * Node Iterator algorithm for counting triangles.
 * <code>
 * for (v in G) 
 *   for (all pairs of neighbors (a, b) of v)
 *     if ((a,b) in G and a < v < b)
 *       triangle += 1
 * </code>
 *
 * Thomas Schank. Algorithmic Aspects of Triangle-Based Network Analysis. PhD
 * Thesis. Universitat Karlsruhe. 2007.
 */
struct NodeIteratorAlgo {
  Galois::GAccumulator<size_t> numTriangles;
  
  struct Process {
    NodeIteratorAlgo* self;
    Process(NodeIteratorAlgo* s): self(s) { }

    void operator()(const GNode& n, Galois::UserContext<GNode>&) { (*this)(n); }
    void operator()(const GNode& n) {
      // Partition neighbors
      // [first, ea) [n] [bb, last)
      Graph::edge_iterator first = graph.edge_begin(n, Galois::MethodFlag::NONE);
      Graph::edge_iterator last = graph.edge_end(n, Galois::MethodFlag::NONE);
      Graph::edge_iterator ea = lowerBound(first, last, LessThan<Graph>(graph, n));
      Graph::edge_iterator bb = lowerBound(first, last, GreaterThanOrEqual<Graph>(graph, n));

      for (; bb != last; ++bb) {
        GNode B = graph.getEdgeDst(bb);
        for (auto aa = first; aa != ea; ++aa) {
          GNode A = graph.getEdgeDst(aa);
          Graph::edge_iterator vv = graph.edge_begin(A, Galois::MethodFlag::NONE);
          Graph::edge_iterator ev = graph.edge_end(A, Galois::MethodFlag::NONE);
          Graph::edge_iterator it = lowerBound(vv, ev, LessThan<Graph>(graph, B));
          if (it != ev && graph.getEdgeDst(it) == B) {
            self->numTriangles += 1;
          }
        }
      }
    }
  };

  void operator()() { 
    Galois::do_all_local(&graph, Process(this));
    std::cout << "NumTriangles: " << numTriangles.reduce() << "\n";
  }
};

template<typename Algo>
void run() {
  Algo algo;

  Galois::StatTimer T;
  T.start();
  algo();
  T.stop();
}

using namespace Galois::Runtime;

struct create_nodes : public Galois::Runtime::Lockable {
  Graphp g;
  create_nodes() {}
  create_nodes(Graphp _g): g(_g) {}

  template<typename Context>
  void operator()(element& item, const Context& cnx) {
    DGNode n = g->createNode(item.v);
    g->addNode(n);
    mapping[item.g] = n;
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,g);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,g);
  }
};

static void readInputGraph_landing_pad(Distributed::RecvBuffer& buf) {
  std::string triangleFilename;
  printf("host: %u and thread id: %d\n", Distributed::networkHostID, LL::getTID());
abort();
  Distributed::gDeserialize(buf, triangleFilename);
  //graph.structureFromFile(triangleFilename);
}

void readInputGraph(std::string triangleFilename) {
  std::vector<element> e;
  Graphp dgraph = DGraph::allocate();;
  if (Distributed::networkHostNum > 1) {
    Distributed::SendBuffer b;
    Distributed::gSerialize(b, triangleFilename);
sleep(2);
printf("Before Broadcast\n");
    Distributed::getSystemNetworkInterface().broadcast(readInputGraph_landing_pad, b);
    getSystemLocalDirectory().makeProgress();
    getSystemRemoteDirectory().makeProgress();
    Distributed::getSystemNetworkInterface().handleReceives();
printf("After Broadcast\n");
    getSystemLocalDirectory().makeProgress();
    getSystemRemoteDirectory().makeProgress();
    Distributed::getSystemNetworkInterface().handleReceives();
  }

  graph.structureFromFile(triangleFilename);
  for (auto ii = graph.begin(); ii != graph.end(); ++ii) {
    unsigned val = graph.getData(*ii,Galois::MethodFlag::NONE);
    e.push_back(element(*ii,val));
  }

printf("adding the vector elements to dist graph\n");
  Galois::for_each<>(e.begin(), e.end(), create_nodes(dgraph));
printf("done with adding the vector elements to dist graph\n");
}

void readGraph() {
  if (inputFilename.find(".gr.triangles") != inputFilename.size() - strlen(".gr.triangles")) {
    // Not directly passed .gr.triangles file
    std::string triangleFilename = inputFilename + ".triangles";
    std::ifstream triangleFile(triangleFilename.c_str());
    if (!triangleFile.good()) {
      // triangles doesn't already exist, create it
      //makeGraph(triangleFilename);
      abort();
    } else {
      // triangles does exist, load it
      readInputGraph(triangleFilename);
    }
  } else {
    //graph.structureFromFile(inputFilename);
    printf("No triangles file!\n");
    abort();
  }

  size_t index = 0;
  for (GNode n : graph) {
    graph.getData(n) = index++;
  }
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  // check the host id and initialise the network
  Galois::Runtime::Distributed::networkStart();

  Galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();
  readGraph();
  Tinitial.stop();

/*
  // XXX Test if preallocation matters
  Galois::Statistic("MeminfoPre", Galois::Runtime::MM::pageAllocInfo());
  Galois::preAlloc(numThreads + 8 * Galois::Runtime::MM::pageAllocInfo());
  Galois::Statistic("MeminfoMid", Galois::Runtime::MM::pageAllocInfo());
 */
  switch (algo) {
    case nodeiterator: run<NodeIteratorAlgo>(); break;
    default: std::cerr << "Unknown algo: " << algo << "\n";
  }
  Galois::Statistic("MeminfoPost", Galois::Runtime::MM::pageAllocInfo());

  // TODO Print num triangles

  // master_terminate();
  Galois::Runtime::Distributed::networkTerminate();

  return 0;
}
