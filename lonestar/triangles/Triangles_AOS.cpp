/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/gstl.h"

#include "galois/runtime/Profile.h"

#include <boost/iterator/transform_iterator.hpp>

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

const char* name = "Triangles";
const char* desc = "Counts the triangles in a graph";
const char* url  = 0;

enum Algo {
  nodeiteratorpre,
};

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumValN(Algo::nodeiteratorpre, "nodeiteratorpre", "Node Iterator (default)"),
                clEnumValEnd),
    cll::init(Algo::nodeiteratorpre));

struct NodeData{
  uint64_t ea;
  uint64_t bb;
};


typedef galois::graphs::LC_CSR_Graph<NodeData, void>::with_numa_alloc<
true>::type ::with_no_lockable<true>::type Graph;
// typedef galois::graphs::LC_CSR_Graph<uint32_t,void> Graph;
// typedef galois::graphs::LC_Linear_Graph<uint32_t,void> Graph;

typedef Graph::GraphNode GNode;

/**
 * Like std::lower_bound but doesn't dereference iterators. Returns the first
 * element for which comp is not true.
 */
template <typename Iterator, typename Compare>
Iterator lowerBound(Iterator first, Iterator last, Compare comp) {
  Iterator it;
  typename std::iterator_traits<Iterator>::difference_type count, half;
  count = std::distance(first, last);
  while (count > 0) {
    it   = first;
    half = count / 2;
    std::advance(it, half);
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
template <typename G>
size_t countEqual(G& g, typename G::edge_iterator aa,
                  typename G::edge_iterator ea, typename G::edge_iterator bb,
                  typename G::edge_iterator eb) {
  size_t retval = 0;
  while (aa != ea && bb != eb) {
    typename G::GraphNode a = g.getEdgeDst(aa);
    typename G::GraphNode b = g.getEdgeDst(bb);
    if (a < b) {
      ++aa;
    } else if (b < a) {
      ++bb;
    } else {
      retval += 1;
      ++aa;
      ++bb;
    }
  }
  return retval;
}

template <typename G>
struct LessThan {
  G& g;
  typename G::GraphNode n;
  LessThan(G& g, typename G::GraphNode n) : g(g), n(n) {}
  bool operator()(typename G::edge_iterator it) { return g.getEdgeDst(it) < n; }
};

template <typename G>
struct GreaterThanOrEqual {
  G& g;
  typename G::GraphNode n;
  GreaterThanOrEqual(G& g, typename G::GraphNode n) : g(g), n(n) {}
  bool operator()(typename G::edge_iterator it) {
    return !(n < g.getEdgeDst(it));
  }
};

template <typename G>
struct DegreeLess : public std::binary_function<typename G::GraphNode,
                                                typename G::GraphNode, bool> {
  typedef typename G::GraphNode N;
  G* g;
  DegreeLess(G& g) : g(&g) {}

  bool operator()(const N& n1, const N& n2) const {
    return std::distance(g->edge_begin(n1), g->edge_end(n1)) <
           std::distance(g->edge_begin(n2), g->edge_end(n2));
  }
};

template <typename G>
struct GetDegree
    : public std::unary_function<typename G::GraphNode, ptrdiff_t> {
  typedef typename G::GraphNode N;
  G* g;
  GetDegree(G& g) : g(&g) {}

  ptrdiff_t operator()(const N& n) const {
    return std::distance(g->edge_begin(n), g->edge_end(n));
  }
};

template <typename GraphNode, typename EdgeTy>
struct IdLess {
  bool
  operator()(const galois::graphs::EdgeSortValue<GraphNode, EdgeTy>& e1,
             const galois::graphs::EdgeSortValue<GraphNode, EdgeTy>& e2) const {
    return e1.dst < e2.dst;
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
typedef galois::gstl::Vector<GNode> VecTy;
void nodeIteratingAlgoPre(Graph& graph) {

  galois::GAccumulator<size_t> numTriangles;
  //galois::GAccumulator<size_t> numItems;
  //numItems.reset();
  //3-motif
  std::map<uint32_t, VecTy> kmap;

    galois::do_all(galois::iterate(graph),//galois::iterate(graph.begin() + i, graph.begin() + i + delta ),
                 [&](GNode n) {

                // Partition neighbors
                // [first, ea) [n] [bb, last)
                auto& ndata = graph.getData(n);
                Graph::edge_iterator first =
                    graph.edge_begin(n, galois::MethodFlag::UNPROTECTED);
                Graph::edge_iterator last =
                    graph.edge_end(n, galois::MethodFlag::UNPROTECTED);
                Graph::edge_iterator ea =
                    lowerBound(first, last, LessThan<Graph>(graph, n));
                Graph::edge_iterator bb =
                    lowerBound(first, last, GreaterThanOrEqual<Graph>(graph, n));

                  ndata.ea = std::abs(std::distance(first, ea));
                  ndata.bb = std::abs(std::distance(first, bb));

                 },
                 galois::loopname("Initialize"));

    //std::cout << "numItems: " << numItems.reduce() << "\n";
    std::cout << "Done Initializing\n";
       galois::do_all(
            galois::iterate(graph),
            [&](const GNode& n) {
              auto& ndata = graph.getData(n);
              Graph::edge_iterator first =
                  graph.edge_begin(n, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator last =
                  graph.edge_end(n, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator ea = first + ndata.ea; // std::advance(first, ndata.ea);
              Graph::edge_iterator bb = first + ndata.bb; //std::advance(first, ndata.bb);

              for (; bb != last; ++bb) {
                GNode B = graph.getEdgeDst(bb);
                for (auto aa = first; aa != ea; ++aa) {
                  GNode A = graph.getEdgeDst(aa);
                  Graph::edge_iterator vv =
                      graph.edge_begin(A, galois::MethodFlag::UNPROTECTED);
                  Graph::edge_iterator ev =
                      graph.edge_end(A, galois::MethodFlag::UNPROTECTED);
                  Graph::edge_iterator it =
                      lowerBound(vv, ev, LessThan<Graph>(graph, B));
                  if (it != ev && graph.getEdgeDst(it) == B) {
                    numTriangles += 1;
                  }
                }
              }
            },
            galois::chunk_size<512>(), galois::steal(),
            galois::loopname("nodeIteratingAlgoPre"));

  std::cout << "NumTriangles: " << numTriangles.reduce() << "\n";
}

void makeGraph(Graph& graph, const std::string& triangleFilename) {
  typedef galois::graphs::FileGraph G;
  typedef G::GraphNode N;

  G initial, permuted;

  initial.fromFileInterleaved<void>(inputFilename);

  // Getting around lack of resize for deque
  std::deque<N> nodes;
  std::copy(initial.begin(), initial.end(), std::back_inserter(nodes));
  // Sort by degree
  galois::ParallelSTL::sort(nodes.begin(), nodes.end(), DegreeLess<G>(initial));

  std::deque<N> p;
  std::copy(nodes.begin(), nodes.end(), std::back_inserter(p));
  // Transpose
  size_t idx = 0;
  for (N n : nodes) {
    p[n] = idx++;
  }

  galois::graphs::permute<void>(initial, p, permuted);
  galois::do_all(galois::iterate(permuted),
                 [&](N x) { permuted.sortEdges<void>(x, IdLess<N, void>()); });

  std::cout << "Writing new input file: " << triangleFilename << "\n";
  permuted.toFile(triangleFilename);
  galois::gPrint("loading file after creating triangleFilename\n");
  galois::graphs::readGraph(graph, permuted);
  //graph.allocateAndLoadGraph(triangleFilename);
}

void readGraph(Graph& graph) {
  if (inputFilename.find(".gr.triangles") !=
      inputFilename.size() - strlen(".gr.triangles")) {
    // Not directly passed .gr.triangles file
    std::string triangleFilename = inputFilename + ".triangles";
    std::ifstream triangleFile(triangleFilename.c_str());
    if (!triangleFile.good()) {
      // triangles doesn't already exist, create it
      galois::gPrint("Start makeGraph\n");
      makeGraph(graph, triangleFilename);
      galois::gPrint("Done makeGraph\n");
    } else {
      // triangles does exist, load it
      galois::gPrint("Start loading", triangleFilename, "\n");
      galois::graphs::readGraph(graph, triangleFilename);
      //graph.allocateAndLoadGraph(triangleFilename);
      galois::gPrint("Done loading", triangleFilename, "\n");
    }
  } else {
    galois::gPrint("Start loading", inputFilename, "\n");
    galois::graphs::readGraph(graph, inputFilename);
    //graph.allocateAndLoadGraph(inputFilename);
    galois::gPrint("Done loading", inputFilename, "\n");
  }

  size_t index = 0;
  for (GNode n : graph) {
    //graph.getData(n) = index++;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph;

  galois::StatTimer Tinitial("GraphReadingTime");
  galois::gPrint("Start readGraph\n");
  Tinitial.start();
  readGraph(graph);
  Tinitial.stop();
  galois::gPrint("Done readGraph\n");

  //galois::preAlloc(500);
  //galois::preAlloc(numThreads + 16 * (graph.size() + graph.sizeEdges()) /
                                    //galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  T.start();
  // case by case preAlloc to avoid allocating unnecessarily
  switch (algo) {
  case nodeiteratorpre:
    nodeIteratingAlgoPre(graph);
    break;

  default:
    std::cerr << "Unknown algo: " << algo << "\n";
  }
  T.stop();

  galois::reportPageAlloc("MeminfoPost");
  return 0;
}
