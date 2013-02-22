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

#include "boost/optional.hpp"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

const char* name = "Triangles";
const char* desc = "Count triangles in a graph";
const char* url = 0;

enum Algo {
  nodeiterator,
  edgeiterator,
  approx
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::nodeiterator, "nodeiterator", "Node Iterator (default)"),
      clEnumValN(Algo::edgeiterator, "edgeiterator", "Edge Iterator"),
      clEnumValN(Algo::approx, "approx", "Approximate"),
      clEnumValEnd), cll::init(Algo::nodeiterator));

//typedef Galois::Graph::LC_CSR_Graph<void,void> Graph;
typedef Galois::Graph::LC_Linear_Graph<void,void> Graph;

typedef Graph::GraphNode GNode;

Graph graph;

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
size_t countEqual(Graph::edge_iterator aa, Graph::edge_iterator ea, Graph::edge_iterator bb, Graph::edge_iterator eb) {
  size_t retval = 0;
  while (aa != ea && bb != eb) {
    GNode a = graph.getEdgeDst(aa);
    GNode b = graph.getEdgeDst(bb);
    if (a < b) {
      ++aa;
    } else if (b < a) {
      ++bb;
    } else {
      retval += 1;
      ++aa; ++bb;
    }
  }
  return retval;
}

struct LessThan {
  GNode n;
  LessThan(GNode n): n(n) { }
  bool operator()(Graph::edge_iterator it) {
    return graph.getEdgeDst(it) < n;
  }
};

struct GreaterThanOrEqual {
  GNode n;
  GreaterThanOrEqual(GNode n): n(n) { }
  bool operator()(Graph::edge_iterator it) {
    return !(n < graph.getEdgeDst(it));
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
  // Break up work into sub-node pieces to ease load balancing
  struct WorkItem {
    GNode node;
    int count;
    WorkItem(const GNode& n, int c): node(n), count(c) { }
  };

  Galois::InsertBag<WorkItem> items;
  Galois::GAccumulator<size_t> numTriangles;

  struct Initialize {
    NodeIteratorAlgo* self;
    Initialize(NodeIteratorAlgo* s): self(s) { }

    void operator()(GNode n) {
      Graph::edge_iterator first = graph.edge_begin(n, Galois::MethodFlag::NONE);
      Graph::edge_iterator last = graph.edge_end(n, Galois::MethodFlag::NONE);

      std::iterator_traits<Graph::edge_iterator>::difference_type d = std::distance(first, last);
      if (d < 128) {
        self->items.push(WorkItem(n, -1));
        return;
      }

      Graph::edge_iterator bb = lowerBound(first, last, GreaterThanOrEqual(n));
      for (int count = 0; bb != last; ++bb, ++count)
        self->items.push(WorkItem(n, count));
    }
  };

  struct Process {
    NodeIteratorAlgo* self;
    Process(NodeIteratorAlgo* s): self(s) { }

    void operator()(const WorkItem& w, Galois::UserContext<WorkItem>&) { (*this)(w); }
    void operator()(const WorkItem& w) {
      // Partition neighbors
      // [first, ea) [n] [bb, last)
      Graph::edge_iterator first = graph.edge_begin(w.node, Galois::MethodFlag::NONE);
      Graph::edge_iterator last = graph.edge_end(w.node, Galois::MethodFlag::NONE);
      Graph::edge_iterator ea = lowerBound(first, last, LessThan(w.node));
      Graph::edge_iterator bb = lowerBound(first, last, GreaterThanOrEqual(w.node));

      if (w.count >= 0) {
        std::advance(bb, w.count);
        last = bb;
        std::advance(last, 1);
      }
      for (; bb != last; ++bb) {
        GNode B = graph.getEdgeDst(bb);
        for (auto aa = first; aa != ea; ++aa) {
          GNode A = graph.getEdgeDst(aa);
          Graph::edge_iterator vv = graph.edge_begin(A, Galois::MethodFlag::NONE);
          Graph::edge_iterator ev = graph.edge_end(A, Galois::MethodFlag::NONE);
          Graph::edge_iterator it = lowerBound(vv, ev, LessThan(B));
          if (it != ev && graph.getEdgeDst(it) == B) {
            self->numTriangles += 1;
          }
        }
      }
    }
  };

  void operator()() { 
    Galois::do_all(graph.begin(), graph.end(), Initialize(this));
    Galois::for_each_local(items, Process(this));
    std::cout << "NumTriangles: " << numTriangles.reduce() << "\n";
  }
};

/**
 * Edge Iterator algorithm for counting triangles.
 * <code>
 * for ((a, b) in E) 
 *   if (a < b)
 *     for (v in intersect(neighbors(a), neighbors(b)))
 *       if (a < v < b)
 *         triangle += 1
 * </code>
 *
 * Thomas Schank. Algorithmic Aspects of Triangle-Based Network Analysis. PhD
 * Thesis. Universitat Karlsruhe. 2007.
 */
struct EdgeIteratorAlgo {
  struct WorkItem {
    GNode src;
    GNode dst;
    WorkItem(const GNode& a1, const GNode& a2): src(a1), dst(a2) { }
  };

  Galois::InsertBag<WorkItem> items;
  Galois::GAccumulator<size_t> numTriangles;

  struct Initialize {
    EdgeIteratorAlgo* self;
    Initialize(EdgeIteratorAlgo* s): self(s) { }

    void operator()(GNode n) {
      for (Graph::edge_iterator edge : graph.out_edges(n, Galois::MethodFlag::NONE)) {
        GNode dst = graph.getEdgeDst(edge);
        if (n < dst)
          self->items.push(WorkItem(n, dst));
      }
    }
  };

  struct Process {
    EdgeIteratorAlgo* self;
    Process(EdgeIteratorAlgo* s): self(s) { }

    void operator()(const WorkItem& w, Galois::UserContext<WorkItem>&) { (*this)(w); }
    void operator()(const WorkItem& w) {
      // Compute intersection of range (w.src, w.dst) in neighbors of w.src and w.dst
      Graph::edge_iterator abegin = graph.edge_begin(w.src, Galois::MethodFlag::NONE);
      Graph::edge_iterator aend = graph.edge_end(w.src, Galois::MethodFlag::NONE);
      Graph::edge_iterator bbegin = graph.edge_begin(w.dst, Galois::MethodFlag::NONE);
      Graph::edge_iterator bend = graph.edge_end(w.dst, Galois::MethodFlag::NONE);

      Graph::edge_iterator aa = lowerBound(abegin, aend, GreaterThanOrEqual(w.src));
      Graph::edge_iterator ea = lowerBound(abegin, aend, LessThan(w.dst));
      Graph::edge_iterator bb = lowerBound(bbegin, bend, GreaterThanOrEqual(w.src));
      Graph::edge_iterator eb = lowerBound(bbegin, bend, LessThan(w.dst));

      self->numTriangles += countEqual(aa, ea, bb, eb);
    }
  };

  void operator()() { 
    Galois::do_all(graph.begin(), graph.end(), Initialize(this));
    Galois::for_each_local(items, Process(this));
    std::cout << "NumTriangles: " << numTriangles.reduce() << "\n";
  }
};

struct ApproxAlgo {
  void operator()() { abort(); }
};

template<typename Algo>
void run() {
  Algo algo;

  Galois::StatTimer T;
  T.start();
  algo();
  T.stop();
}

template<typename G>
struct DegreeLess {
  typedef typename G::GraphNode N;
  G& g;
  DegreeLess(G& g): g(g) { }

  bool operator()(const N& n1, const N& n2) const {
    return std::distance(g.edge_begin(n1), g.edge_end(n1)) < std::distance(g.edge_begin(n2), g.edge_end(n2));
  }
};

template<typename EdgeTy>
struct IdLess {
  bool operator()(const Galois::Graph::EdgeSortValue<EdgeTy>& e1, const Galois::Graph::EdgeSortValue<EdgeTy>& e2) const {
    return e1.dst < e2.dst;
  }
};

void makeGraph(const std::string& triangleFilename) {
  typedef Galois::Graph::FileGraph G;
  typedef G::GraphNode N;

  G initial, permuted;

  initial.structureFromFile(inputFilename);
  
  // Getting around lack of resize for deque
  std::deque<N> nodes;
  std::copy(initial.begin(), initial.end(), std::back_inserter(nodes));
  // Sort by degree
  Galois::ParallelSTL::sort(nodes.begin(), nodes.end(), DegreeLess<G>(initial));
  
  std::deque<N> p;
  std::copy(nodes.begin(), nodes.end(), std::back_inserter(p));
  // Transpose
  size_t idx = 0;
  for (N n : nodes) {
    p[n] = idx++;
  }

  Galois::Graph::permute<void>(initial, p, permuted);
  Galois::do_all(permuted.begin(), permuted.end(), [&](N x) { permuted.sortEdges<void>(x, IdLess<void>()); });

  std::cout << "Writing new input file: " << triangleFilename << "\n";
  permuted.structureToFile(triangleFilename);
  graph.structureFromGraph(permuted);
}

void readGraph() {
  if (inputFilename.find(".gr.triangles") != inputFilename.size() - strlen(".gr.triangles")) {
    // Not directly passed .gr.triangles file
    std::string triangleFilename = inputFilename + ".triangles";
    std::ifstream triangleFile(triangleFilename.c_str());
    if (!triangleFile.good()) {
      // triangles doesn't already exist, create it
      makeGraph(triangleFilename);
    } else {
      // triangles does exist, load it
      graph.structureFromFile(triangleFilename);
    }
  } else {
    graph.structureFromFile(inputFilename);
  }
}

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();
  readGraph();
  Tinitial.stop();

  // XXX Test if preallocation matters
  //Galois::preAlloc(numThreads);
  Galois::Statistic("MeminfoPre", Galois::Runtime::MM::pageAllocInfo());
  switch (algo) {
    case nodeiterator: run<NodeIteratorAlgo>(); break;
    case edgeiterator: run<EdgeIteratorAlgo>(); break;
    case approx: run<ApproxAlgo>(); break;
    default: std::cerr << "Unknown algo: " << algo << "\n";
  }
  Galois::Statistic("MeminfoPost", Galois::Runtime::MM::pageAllocInfo());

  // TODO Print num triangles

  return 0;
}
