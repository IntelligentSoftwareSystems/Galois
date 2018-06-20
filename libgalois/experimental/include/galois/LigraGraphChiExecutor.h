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

#ifndef GALOIS_LIGRAGRAPHCHIEXECUTOR_H
#define GALOIS_LIGRAGRAPHCHIEXECUTOR_H

#include "LigraExecutor.h"
#include "GraphChiExecutor.h"

namespace galois {
//! Implementation of combination of Ligra and GraphChi DSL in Galois
namespace ligraGraphChi {

template <bool Forward, typename Graph, typename EdgeOperator, typename Bag>
void edgeMap(size_t size, Graph& graph, EdgeOperator op, Bag& output) {
  typedef galois::graphs::BindSegmentGraph<Graph> WrappedGraph;
  WrappedGraph wgraph(graph);

  output.densify();
  galois::graphChi::internal::vertexMap<false, false>(
      graph, wgraph,
      galois::Ligra::internal::DenseForwardOperator<
          WrappedGraph, Bag, EdgeOperator, Forward, true>(wgraph, output,
                                                          output, op),
      static_cast<Bag*>(0), size);
}

template <bool Forward, typename Graph, typename EdgeOperator, typename Bag>
void edgeMap(size_t size, Graph& graph, EdgeOperator op, Bag& input,
             Bag& output, bool denseForward) {
  typedef galois::graphs::BindSegmentGraph<Graph> WrappedGraph;
  WrappedGraph wgraph(graph);

  size_t count = input.getCount();

  if (!denseForward && count > graph.sizeEdges() / 20) {
    input.densify();
    if (denseForward) {
      abort(); // Never used now
      output.densify();
      galois::graphChi::internal::vertexMap<false, false>(
          graph, wgraph,
          galois::Ligra::internal::DenseForwardOperator<
              WrappedGraph, Bag, EdgeOperator, Forward, false>(wgraph, input,
                                                               output, op),
          static_cast<Bag*>(0), size);
    } else {
      galois::graphChi::internal::vertexMap<false, false>(
          graph, wgraph,
          galois::Ligra::internal::DenseOperator<WrappedGraph, Bag,
                                                 EdgeOperator, Forward>(
              wgraph, input, output, op),
          static_cast<Bag*>(0), size);
    }
  } else {
    galois::graphChi::internal::vertexMap<true, false>(
        graph, wgraph,
        galois::Ligra::internal::SparseOperator<WrappedGraph, Bag, EdgeOperator,
                                                Forward>(wgraph, output, op),
        &input, size);
  }
}

template <bool Forward, typename Graph, typename EdgeOperator, typename Bag>
void edgeMap(size_t size, Graph& graph, EdgeOperator op,
             typename Graph::GraphNode single, Bag& output) {
  Bag input(graph.size());
  input.push(graph.idFromNode(single), 1);
  edgeMap<Forward>(size, graph, op, input, output, false);
}

template <typename... Args>
void outEdgeMap(Args&&... args) {
  edgeMap<true>(std::forward<Args>(args)...);
}

template <typename... Args>
void inEdgeMap(Args&&... args) {
  edgeMap<false>(std::forward<Args>(args)...);
}

template <bool UseGraphChi>
struct ChooseExecutor {
  template <typename... Args>
  void inEdgeMap(size_t size, Args&&... args) {
    edgeMap<false>(size, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void outEdgeMap(size_t size, Args&&... args) {
    edgeMap<true>(size, std::forward<Args>(args)...);
  }

  template <typename Graph>
  void checkIfInMemoryGraph(Graph& g, size_t size) {
    if (galois::graphChi::internal::fitsInMemory(g, size)) {
      g.keepInMemory();
    }
  }
};

template <>
struct ChooseExecutor<false> {
  template <typename... Args>
  void inEdgeMap(size_t size, Args&&... args) {
    galois::Ligra::edgeMap<false>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void outEdgeMap(size_t size, Args&&... args) {
    galois::Ligra::edgeMap<true>(std::forward<Args>(args)...);
  }

  template <typename Graph>
  void checkIfInMemoryGraph(Graph& g, size_t size) {}
};

} // namespace ligraGraphChi
} // namespace galois
#endif
