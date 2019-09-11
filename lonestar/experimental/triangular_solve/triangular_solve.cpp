/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2019, The University of Texas at Austin. All rights reserved.
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

#include <algorithm>
#include <atomic>
#include <cmath>

// Silence erroneous warnings from within Boost headers
// that show up with gcc 8.1.
#pragma GCC diagnostic ignored "-Wparentheses"

#include <galois/Galois.h>
#include <galois/graphs/FileGraph.h>
#include <galois/graphs/Graph.h>
#include <galois/graphs/LCGraph.h>

// Vendored from an old version of LLVM for Lonestar app command line handling.
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

static char const* name = "Triangular Solve";
// NOTE: left solve works how you'd want for CSR
// Right solve works how you'd want for CSC.
// You need a push style operator, otherwise the dependency tracking
// requires a graph transpose, and those are the only two "triangular solve"
// variants that permit that.
// In particular, right solve with CSR and left solve with CSC require a transpose.
// Upper/lower triangular will need slightly different loop bounds,
// but will have similar dependency tracking needs.
// Note: To use ILU efficiently, this will eventually need an option for
// a triangular solve with ones along the diagonal.
static char const* desc =
    "Back substitution to solve x.T A = b.T for sparse "
    "upper triangular matrix A and vector b.";
static char const* url = "triangular_solve";

static llvm::cl::opt<unsigned long long> n{
    "n", llvm::cl::desc("number of rows of the generated square matrix.")};

using graph_t = typename galois::graphs::LC_CSR_Graph<std::atomic<std::size_t>, double>::with_no_lockable<true>::type;

// Autogenerate a very simple lower triangular sparse matrix.
auto generate_matrix(graph_t& built_graph, std::size_t n) noexcept {

  galois::graphs::FileGraphWriter temp_graph;
  // Approximate the dependency pattern resulting from
  // an ILU(0) factorization of a system resulting from
  // a 2d regular grid (similar to what's done for autogenerating
  // the matrix used in the ILU(0) app).
  std::size_t offset = std::sqrt(n);
  temp_graph.setNumNodes(n);
  std::size_t num_edges = 3 * n - 1 - offset;
  temp_graph.setNumEdges(num_edges);
  temp_graph.setSizeofEdgeData(sizeof(double));
  galois::LargeArray<graph_t::edge_data_type> edge_data;
  edge_data.create(num_edges);
  temp_graph.phase1();
  for (std::size_t i = 0; i < n; i++) {
    temp_graph.incrementDegree(i);
    if (i + 1 < n) {
      temp_graph.incrementDegree(i);
    }
    if (i < n - offset) {
      temp_graph.incrementDegree(i);
    }
  }
  temp_graph.phase2();

  for (std::size_t i = 0; i < n; i++) {
    edge_data.set(temp_graph.addNeighbor(i, i), 4.);
    if (i + 1 < n) {
      edge_data.set(temp_graph.addNeighbor(i, i + 1), -1.);
    }
    if (i < n - offset) {
      edge_data.set(temp_graph.addNeighbor(i, i + offset), -1.);
    }
  }

  // TODO: is it possible to set the edge data
  // during construction without copying here?
  auto* rawEdgeData = temp_graph.finish<graph_t::edge_data_type>();
  std::uninitialized_copy(std::make_move_iterator(edge_data.begin()),
                          std::make_move_iterator(edge_data.end()),
                          rawEdgeData);

  galois::graphs::readGraph(built_graph, temp_graph);
}

int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url);

  graph_t graph;

  generate_matrix(graph, n);

  // Initialize the right hand side with filler data for now.
  galois::LargeArray<double> rhs;
  rhs.create(n);
  std::fill(rhs.begin(), rhs.end(), 1.);

  // Initialize counters for dependency tracking.
  // TODO: optimize the initialization phase at some point.
  // First set them all to 0.
  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&](auto node) {
      graph.getData(node, galois::MethodFlag::UNPROTECTED).store(0, std::memory_order_relaxed);
    },
    galois::loopname("zero_counters"));
  // Initialize the atomic counters.
  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&](auto node) {
      for (auto edge : graph.edges(node, galois::MethodFlag::UNPROTECTED)) {
        auto neighbor = graph.getEdgeDst(edge);
        if (neighbor == node) continue;
        graph.getData(neighbor, galois::MethodFlag::UNPROTECTED).fetch_add(1, std::memory_order_relaxed);
      }
    },
    galois::loopname("initialize_counters"));

  // Now find the ones that start as ready.
  // This could potentially be done as a part of the iterator for the starting set instead.
  galois::InsertBag<graph_t::GraphNode> starting_nodes;

  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&](auto node) {
      if (!graph.getData(node, galois::MethodFlag::UNPROTECTED).load(std::memory_order_relaxed)) {
        starting_nodes.emplace(node);
      }
    },
    galois::loopname("find_starts"));

  // Now actually do the triangular solve.
  galois::for_each(
    galois::iterate(starting_nodes.begin(), starting_nodes.end()),
    [&](auto node, auto &context) noexcept {
      auto counter_check_val = graph.getData(node, galois::MethodFlag::UNPROTECTED).load(std::memory_order_acquire);
      if (counter_check_val) {
        GALOIS_DIE("Work item asked to run before it was actually ready.");
      }
      auto edge_iterator = graph.edge_begin(node, galois::MethodFlag::UNPROTECTED);
      auto edge_end = graph.edge_end(node, galois::MethodFlag::UNPROTECTED);
      assert(graph.getEdgeDst(*edge_iterator) == node);
      rhs[node] /= graph.getEdgeData(*edge_iterator);
      ++edge_iterator;
      while (edge_iterator < edge_end) {
        auto neighbor = graph.getEdgeDst(*edge_iterator);
        assert(neighbor > node);
        rhs[neighbor] -= rhs[node] * graph.getEdgeData(*edge_iterator);
        auto &other_counter = graph.getData(neighbor, galois::MethodFlag::UNPROTECTED);
        if (!(other_counter.fetch_sub(1, std::memory_order_release) - 1)) {
          context.push(neighbor);
        }
        ++edge_iterator;
      }
    },
    galois::loopname("back_substitution"),
    galois::no_conflicts(),
    galois::wl<galois::worklists::PerSocketChunkFIFO<128>>());
}
