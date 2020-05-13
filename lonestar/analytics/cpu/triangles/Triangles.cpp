/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

constexpr static const unsigned CHUNK_SIZE = 64u;
enum Algo { nodeiterator, edgeiterator, orderedCount };

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumValN(Algo::nodeiterator, "nodeiterator", "Node Iterator"),
                clEnumValN(Algo::edgeiterator, "edgeiterator", "Edge Iterator"),
                clEnumValN(Algo::orderedCount, "orderedCount",
                           "Ordered Simple Count (default)")),
    cll::init(Algo::orderedCount));

static cll::opt<bool>
    relabel("relabel",
            cll::desc("Relabel nodes of the graph (default value true)"),
            cll::init(true));

static cll::opt<bool> storeRelabeledGraph(
    "storeRelabeledGraph",
    cll::desc("Write the relabeled graph to disk for future use with "
              ".gr.triangles extension (default value true)"),
    cll::init(true));

typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<
    true>::type ::with_no_lockable<true>::type Graph;

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
struct DegreeGreater
    : public std::binary_function<typename G::GraphNode, typename G::GraphNode,
                                  bool> {
  typedef typename G::GraphNode N;
  G* g;
  DegreeGreater(G& g) : g(&g) {}

  bool operator()(const N& n1, const N& n2) const {
    return std::distance(g->edge_begin(n1), g->edge_end(n1)) >
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
void nodeIteratingAlgo(Graph& graph) {

  galois::GAccumulator<size_t> numTriangles;

  //! [profile w/ vtune]
  galois::runtime::profileVtune(
      [&]() {
        galois::do_all(
            galois::iterate(graph),
            [&](const GNode& n) {
              // Partition neighbors
              // [first, ea) [n] [bb, last)
              Graph::edge_iterator first =
                  graph.edge_begin(n, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator last =
                  graph.edge_end(n, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator ea =
                  lowerBound(first, last, LessThan<Graph>(graph, n));
              Graph::edge_iterator bb =
                  lowerBound(first, last, GreaterThanOrEqual<Graph>(graph, n));

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
            galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
            galois::loopname("nodeIteratingAlgo"));
      },
      "nodeIteratorAlgo");
  //! [profile w/ vtune]

  std::cout << "Num Triangles: " << numTriangles.reduce() << "\n";
}

/**
 * Lambda function to count triangles
 */
void orderedCountFunc(Graph& graph, GNode n,
                      galois::GAccumulator<size_t>& numTriangles) {
  size_t numTriangles_local = 0;
  for (auto it_v : graph.edges(n)) {
    auto v = graph.getEdgeDst(it_v);
    if (v > n)
      break;
    Graph::edge_iterator it_n =
        graph.edge_begin(n, galois::MethodFlag::UNPROTECTED);

    for (auto it_vv : graph.edges(v)) {
      auto vv = graph.getEdgeDst(it_vv);
      if (vv > v)
        break;
      while (graph.getEdgeDst(it_n) < vv)
        it_n++;
      if (vv == graph.getEdgeDst(it_n)) {
        numTriangles_local += 1;
      }
    }
  }
  numTriangles += numTriangles_local;
}

/*
 * Simple counting loop, instead of binary searching.
 */
void orderedCountAlgo(Graph& graph) {
  galois::GAccumulator<size_t> numTriangles;
  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& n) { orderedCountFunc(graph, n, numTriangles); },
      galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
      galois::loopname("orderedCountAlgo"));

  galois::gPrint("Num Triangles: ", numTriangles.reduce(), "\n");
}

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
void edgeIteratingAlgo(Graph& graph) {

  struct WorkItem {
    GNode src;
    GNode dst;
    WorkItem(const GNode& a1, const GNode& a2) : src(a1), dst(a2) {}
  };

  galois::InsertBag<WorkItem> items;
  galois::GAccumulator<size_t> numTriangles;

  galois::do_all(
      galois::iterate(graph),
      [&](GNode n) {
        for (Graph::edge_iterator edge :
             graph.out_edges(n, galois::MethodFlag::UNPROTECTED)) {
          GNode dst = graph.getEdgeDst(edge);
          if (n < dst)
            items.push(WorkItem(n, dst));
        }
      },
      galois::loopname("Initialize"));

  //  galois::runtime::profileVtune(
  //! [profile w/ papi]
  galois::runtime::profilePapi(
      [&]() {
        galois::do_all(
            galois::iterate(items),
            [&](const WorkItem& w) {
              // Compute intersection of range (w.src, w.dst) in neighbors of
              // w.src and w.dst
              Graph::edge_iterator abegin =
                  graph.edge_begin(w.src, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator aend =
                  graph.edge_end(w.src, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator bbegin =
                  graph.edge_begin(w.dst, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator bend =
                  graph.edge_end(w.dst, galois::MethodFlag::UNPROTECTED);

              Graph::edge_iterator aa = lowerBound(
                  abegin, aend, GreaterThanOrEqual<Graph>(graph, w.src));
              Graph::edge_iterator ea =
                  lowerBound(abegin, aend, LessThan<Graph>(graph, w.dst));
              Graph::edge_iterator bb = lowerBound(
                  bbegin, bend, GreaterThanOrEqual<Graph>(graph, w.src));
              Graph::edge_iterator eb =
                  lowerBound(bbegin, bend, LessThan<Graph>(graph, w.dst));

              numTriangles += countEqual(graph, aa, ea, bb, eb);
            },
            galois::loopname("edgeIteratingAlgo"),
            galois::chunk_size<CHUNK_SIZE>(), galois::steal());
      },
      "edgeIteratorAlgo");
  //! [profile w/ papi]

  std::cout << "NumTriangles: " << numTriangles.reduce() << "\n";
}

//! Sorts read graph by degree (high degree nodes are reindexed to beginning)
void makeSortedGraph(Graph& graph) {
  // read original graph
  galois::graphs::FileGraph initial;
  initial.fromFileInterleaved<void>(inputFilename);

  size_t numGraphNodes = initial.size();
  // create node -> degree pairs
  using DegreeNodePair = std::pair<uint64_t, uint32_t>;
  std::vector<DegreeNodePair> dnPairs(numGraphNodes);
  galois::do_all(
      galois::iterate((size_t)0, numGraphNodes),
      [&](size_t nodeID) {
        size_t nodeDegree =
            std::distance(initial.edge_begin(nodeID), initial.edge_end(nodeID));
        dnPairs[nodeID] = DegreeNodePair(nodeDegree, nodeID);
      },
      galois::loopname("CreateDegreeNodeVector"));

  // sort by degree (first item)
  galois::ParallelSTL::sort(dnPairs.begin(), dnPairs.end(),
                            std::greater<DegreeNodePair>());

  // create mapping, get degrees out to another vector to get prefix sum
  std::vector<uint32_t> oldToNewMapping(numGraphNodes);
  std::vector<uint64_t> newPrefixSum(numGraphNodes);
  galois::do_all(
      galois::iterate((size_t)0, numGraphNodes),
      [&](size_t index) {
        // save degree, which is pair.first
        newPrefixSum[index] = dnPairs[index].first;
        // save mapping; original index is in .second, map it to current index
        oldToNewMapping[dnPairs[index].second] = index;
      },
      galois::loopname("CreateRemappingGetPrefixSum"));

  // get prefix sum
  for (size_t i = 1; i < numGraphNodes; i++) {
    newPrefixSum[i] += newPrefixSum[i - 1];
  }

  // allocate graph
  graph.allocateFrom(numGraphNodes, initial.sizeEdges());
  // construct nodes
  graph.constructNodes();
  // set edge endpoints using prefix sum
  galois::do_all(
      galois::iterate((size_t)0, numGraphNodes),
      [&](size_t nodeIndex) {
        graph.fixEndEdge(nodeIndex, newPrefixSum[nodeIndex]);
      },
      galois::loopname("SetEdgeEndpoints"));

  // construct edges by looping through filegraph and saving to correct
  // locations
  galois::do_all(
      galois::iterate(initial.begin(), initial.end()),
      [&](uint32_t oldNodeID) {
        uint32_t newIndex = oldToNewMapping[oldNodeID];

        // get the start location of this reindex'd nodes edges
        uint64_t currentEdgeIndex;
        if (newIndex != 0) {
          currentEdgeIndex = newPrefixSum[newIndex - 1];
        } else {
          currentEdgeIndex = 0;
        }

        // construct the graph, reindexing as it goes along
        for (auto e = initial.edge_begin(oldNodeID);
             e < initial.edge_end(oldNodeID); e++) {
          // get destination, reindex
          uint32_t oldEdgeDst       = initial.getEdgeDst(e);
          uint32_t reindexedEdgeDst = oldToNewMapping[oldEdgeDst];

          // construct edge
          graph.constructEdge(currentEdgeIndex, reindexedEdgeDst);
          currentEdgeIndex++;
        }
        // this assert makes sure reindex was correct + makes sure all edges
        // are accounted for
        assert(currentEdgeIndex = newPrefixSum[newIndex]);
      },
      galois::steal(), galois::loopname("ReindexingGraph"));

  // sort by destinations
  graph.sortAllEdgesByDst();
  // initialize local ranges
  graph.initializeLocalRanges();
}

void readGraph(Graph& graph) {
  if (relabel) {
    galois::gInfo("Relabeling and sorting graph...");
    galois::StatTimer Trelabel("GraphRelabelTimer");
    Trelabel.start();
    makeSortedGraph(graph);
    Trelabel.stop();
  } else {
    galois::graphs::readGraph(graph, inputFilename);
    // algorithm correctness requires sorting edges by destination
    graph.sortAllEdgesByDst();
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph;

  galois::StatTimer Tinitial("GraphReadingTime");
  Tinitial.start();
  readGraph(graph);
  Tinitial.stop();

  galois::preAlloc(numThreads + 16 * (graph.size() + graph.sizeEdges()) /
                                    galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::gInfo("Starting triangle counting...");
  galois::StatTimer T;
  T.start();
  // case by case preAlloc to avoid allocating unnecessarily
  switch (algo) {
  case nodeiterator:
    nodeIteratingAlgo(graph);
    break;

  case edgeiterator:
    edgeIteratingAlgo(graph);
    break;

  case orderedCount:
    orderedCountAlgo(graph);
    break;

  default:
    std::cerr << "Unknown algo: " << algo << "\n";
  }
  T.stop();

  galois::reportPageAlloc("MeminfoPost");
  return 0;
}
