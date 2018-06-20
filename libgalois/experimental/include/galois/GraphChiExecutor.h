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

#ifndef GALOIS_GRAPHCHIEXECUTOR_H
#define GALOIS_GRAPHCHIEXECUTOR_H

#include "galois/graphs/OCGraph.h"
#include "galois/graphs/GraphNodeBag.h"

#include <boost/iterator/filter_iterator.hpp>
#include <boost/utility.hpp>

namespace galois {
//! Implementation of GraphChi DSL in Galois
namespace graphChi {

namespace internal {

template <bool PassWrappedGraph>
struct DispatchOperator {
  template <typename O, typename G, typename N>
  static void run(O&& o, G&& g, N&& n) {
    std::forward<O>(o)(std::forward<G>(g), std::forward<N>(n));
  }
};

template <>
struct DispatchOperator<false> {
  template <typename O, typename G, typename N>
  static void run(O&& o, G&& g, N&& n) {
    std::forward<O>(o)(std::forward<N>(n));
  }
};

template <typename Graph, typename Bag>
struct contains_node {
  Graph* graph;
  Bag* bag;
  contains_node(Graph* g, Bag* b) : graph(g), bag(b) {}
  bool operator()(typename Graph::GraphNode n) {
    return bag->contains(graph->idFromNode(n));
  }
};

template <typename EdgeTy>
struct sizeof_edge {
  static const unsigned int value = sizeof(EdgeTy);
};

template <>
struct sizeof_edge<void> {
  static const unsigned int value = 0;
};

struct logical_or {
  bool operator()(bool a, bool b) const { return a || b; }
};

template <typename Graph, typename Seg, typename Bag>
bool any_in_range(Graph& graph, const Seg& cur, Bag* input) {
  return std::find_if(graph.begin(cur), graph.end(cur),
                      contains_node<Graph, Bag>(&graph, input)) !=
         graph.end(cur);
  // TODO(ddn): Figure out the memory leak in ParallelSTL::find_if
  // return galois::ParallelSTL::find_if(graph.begin(cur), graph.end(cur),
  // contains_node<Graph>(&graph, input)) != graph.end(cur); return
  // galois::ParallelSTL::map_reduce(graph.begin(cur), graph.end(cur),
  // contains_node<Graph,Bag>(&graph, input), false, logical_or());
}

template <typename Graph>
size_t computeEdgeLimit(Graph& graph, size_t memoryLimit) {
  // Convert memoryLimit which is in MB into edges
  size_t bytes = memoryLimit;
  bytes *= 1024 * 1024;
  size_t sizeNodes = graph.size() * sizeof(uint64_t);
  if (bytes < sizeNodes) {
    GALOIS_DIE("Cannot limit graph in memory allotted");
  }
  bytes -= sizeNodes;
  // double-buffering (2), in and out edges (2)
  size_t edgeBytes =
      2 * 2 *
      (sizeof(uint64_t) + sizeof_edge<typename Graph::edge_data_type>::value);
  size_t edges = bytes / edgeBytes;

  return edges;
}

template <typename Graph>
bool fitsInMemory(Graph& graph, size_t memoryLimit) {
  size_t bytes = memoryLimit;
  bytes *= 1024 * 1024;
  size_t nodeBytes = graph.size() * sizeof(uint64_t);
  size_t edgeBytes =
      graph.sizeEdges() * 2 *
      (sizeof(uint64_t) + sizeof_edge<typename Graph::edge_data_type>::value);

  return nodeBytes + edgeBytes < bytes;
}

template <bool CheckInput, bool PassWrappedGraph, typename Graph,
          typename WrappedGraph, typename VertexOperator, typename Bag>
void vertexMap(Graph& graph, WrappedGraph& wgraph, VertexOperator op,
               Bag* input, size_t memoryLimit) {
  typedef typename Graph::segment_type segment_type;
  galois::GAccumulator<size_t> rounds;

  size_t edges = computeEdgeLimit(graph, memoryLimit);
  segment_type prev;
  segment_type cur = graph.nextSegment(edges);

  bool useDense;
  if (!CheckInput) {
    useDense = true;
  } else {
    // TODO improve this heuristic
    bool useSparse = (cur.size() > graph.size() / 2) &&
                     (input->getSize() < graph.size() / 4);
    useDense = !useSparse;
  }

  if (useDense && CheckInput) {
    input->densify();
  }

  while (cur) {
    if (!CheckInput || !useDense || any_in_range(graph, cur, input)) {
      if (!cur.loaded()) {
        graph.load(cur);
      }

      segment_type next = graph.nextSegment(cur, edges);

      int first    = 0;
      bool updated = false;
      wgraph.setSegment(cur);

      if (useDense) {

        galois::do_all(
            galois::iterate(graph.begin(cur), graph.end(cur)),
            [&](typename Graph::GraphNode n) {
              if (!updated) {
                if (first == 0 && __sync_bool_compare_and_swap(&first, 0, 1)) {
                  if (prev.loaded()) {
                    graph.unload(prev);
                  }
                  if (next) {
                    graph.load(next);
                  }
                }
                updated = true;
              }
              if (CheckInput && !input->contains(graph.idFromNode(n)))
                return;

              DispatchOperator<PassWrappedGraph>::run(op, wgraph, n);
            },
            galois::loopname("DenseVertexMap"));

      } else {
        galois::do_all(
            galois::iterate(*input),
            [&](size_t n) {
              if (!updated) {
                if (first == 0 && __sync_bool_compare_and_swap(&first, 0, 1)) {
                  if (prev.loaded()) {
                    graph.unload(prev);
                  }
                  if (next) {
                    graph.load(next);
                  }
                }
                updated = true;
              }
              // Check if range
              if (!cur.containsNode(n)) {
                return;
              }

              DispatchOperator<PassWrappedGraph>::run(op, wgraph,
                                                      graph.nodeFromId(n));
            },
            galois::loopname("SparseVertexMap"));
      }

      // XXX Shouldn't be necessary
      if (prev.loaded()) {
        abort();
        graph.unload(prev);
      }

      rounds += 1;

      prev = cur;
      cur  = next;
    } else {
      segment_type next = graph.nextSegment(cur, edges);
      if (prev.loaded())
        graph.unload(prev);
      if (cur.loaded())
        graph.unload(cur);
      cur = next;
    }
  }

  if (prev.loaded())
    graph.unload(prev);

  galois::runtime::reportStat_Single("GraphChiExec", "Rounds", rounds.reduce());
}
} // namespace internal

template <typename Graph, typename VertexOperator>
void vertexMap(Graph& graph, VertexOperator op, size_t size) {
  galois::graphs::BindSegmentGraph<Graph> wgraph(graph);

  internal::vertexMap<false, true>(graph, wgraph, op,
                                   static_cast<GraphNodeBag<>*>(0), size);
}

template <typename Graph, typename VertexOperator, typename Bag>
void vertexMap(Graph& graph, VertexOperator op, Bag& input, size_t size) {
  galois::graphs::BindSegmentGraph<Graph> wgraph(graph);

  internal::vertexMap<true, true>(graph, wgraph, op, &input, size);
}

} // namespace graphChi
} // namespace galois

#endif
