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

#include <atomic>
#include <cmath>
#include <limits>

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

static char const* name = "ILU(0) Preconditioner";
static char const* desc =
    "Incomplete LU factorization";
static char const* url = "ilu";

static llvm::cl::opt<unsigned long long> n{
    "n", llvm::cl::desc("number of rows of the generated square matrix.")};

using graph_t = typename galois::graphs::LC_CSR_Graph<std::atomic<std::size_t>, double>::with_no_lockable<true>::type;

// Autogenerate a very simple sparse matrix.
auto generate_matrix(graph_t& built_graph, std::size_t n) noexcept {

  galois::graphs::FileGraphWriter temp_graph;
  // Couple every node with the node +-1 from it and +- offset from it
  // (if such nodes exist).
  // This gives a very rough approximation to the dependency pattern in
  // a 2d regular grid.
  std::size_t offset = std::round(std::sqrt(n));
  temp_graph.setNumNodes(n);
  std::size_t num_edges = 3 * n - 2 + 2 * (n - offset);
  temp_graph.setNumEdges(num_edges);
  temp_graph.setSizeofEdgeData(sizeof(double));
  galois::LargeArray<graph_t::edge_data_type> edge_data;
  edge_data.create(num_edges);
  temp_graph.phase1();
  for (std::size_t i = 0; i < n; i++) {
    if (i >= offset) {
      temp_graph.incrementDegree(i);
    }
    if (i >= offset && i % offset) {
      temp_graph.incrementDegree(i);
    }
    temp_graph.incrementDegree(i);
    if (i + 1 < n && (i + 1) % offset) {
      temp_graph.incrementDegree(i);
    }
    if (i < n - offset) {
      temp_graph.incrementDegree(i);
    }
  }
  temp_graph.phase2();

  for (std::size_t i = 0; i < n; i++) {
    if (i >= offset) {
      edge_data.set(temp_graph.addNeighbor(i, i - offset), -1.);
    }
    if (i > 0 && i % offset) {
      edge_data.set(temp_graph.addNeighbor(i, i - 1), -1.);
    }
    edge_data.set(temp_graph.addNeighbor(i, i), 4.);
    if (i + 1 < n && (i + 1) % offset) {
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

void generate_matrix_3d(graph_t& built_graph, std::size_t n) noexcept {
  galois::graphs::FileGraphWriter temp_graph;
  std::size_t inner_offset = std::round(std::cbrt(n));
  std::size_t outer_offset = inner_offset * inner_offset;
  temp_graph.setNumNodes(n);
  std::size_t num_edges = n;
  num_edges += 2 * (n - 1 - (n - 1) / inner_offset);
  std::size_t num_whole_diagonal_blocks = n / outer_offset;
  std::size_t last_block_partial = n % outer_offset;
  num_edges += 2 * (num_whole_diagonal_blocks * (inner_offset - 1) * inner_offset);
  if (last_block_partial > inner_offset) {
    num_edges += 2 * (last_block_partial - inner_offset);
  }
  num_edges += 2 * (n - outer_offset);
  temp_graph.setNumEdges(num_edges);
  temp_graph.setSizeofEdgeData(sizeof(double));
  galois::LargeArray<graph_t::edge_data_type> edge_data;
  edge_data.create(num_edges);
  temp_graph.phase1();
  for (std::size_t i = 0; i < n; i++) {
    if (i >= outer_offset) {
      temp_graph.incrementDegree(i);
    }
    if (i > inner_offset && (i / inner_offset) % inner_offset) {
      temp_graph.incrementDegree(i);
    }
    if (i && i % inner_offset) {
      temp_graph.incrementDegree(i);
    }
    temp_graph.incrementDegree(i);
    if (i + 1 < n && (i + 1) % inner_offset) {
      temp_graph.incrementDegree(i);
    }
    if (i < n - inner_offset && ((i + inner_offset) / inner_offset) % inner_offset) {
      temp_graph.incrementDegree(i);
    }
    if (i < n - outer_offset) {
      temp_graph.incrementDegree(i);
    }
  }
  temp_graph.phase2();
  for (std::size_t i = 0; i < n; i++) {
    if (i >= outer_offset) {
      edge_data.set(temp_graph.addNeighbor(i, i - outer_offset), 1.);
    }
    if (i > inner_offset && (i / inner_offset) % inner_offset) {
      edge_data.set(temp_graph.addNeighbor(i, i - inner_offset), 1.);
    }
    if (i && i % inner_offset) {
      edge_data.set(temp_graph.addNeighbor(i, i - 1), 1.);
    }
    edge_data.set(temp_graph.addNeighbor(i, i), -6.);
    if (i + 1 < n && (i + 1) % inner_offset) {
      edge_data.set(temp_graph.addNeighbor(i, i + 1), 1.);
    }
    if (i < n - inner_offset && ((i + inner_offset) / inner_offset) % inner_offset) {
      edge_data.set(temp_graph.addNeighbor(i, i + inner_offset), 1.);
    }
    if (i < n - outer_offset) {
      edge_data.set(temp_graph.addNeighbor(i, i + outer_offset), 1.);
    }
  }
  auto* rawEdgeData = temp_graph.finish<graph_t::edge_data_type>();
  std::uninitialized_copy(std::make_move_iterator(edge_data.begin()),
                          std::make_move_iterator(edge_data.end()),
                          rawEdgeData);

  galois::graphs::readGraph(built_graph, temp_graph);
}

int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url);

  typename galois::graphs::LC_CSR_Graph<std::atomic<std::size_t>, double>::with_no_lockable<true>::type graph;

  generate_matrix(graph, n);

  // Sort edges and zero counters.
  // In the symmetric case it is sufficient to make j depend on k if A[k,j] != 0.
  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&](auto node) noexcept {
      graph.sortEdgesByDst(node, galois::MethodFlag::UNPROTECTED);
      graph.getData(node, galois::MethodFlag::UNPROTECTED).store(0, std::memory_order_relaxed);
    },
    galois::loopname("sort_edges_and_zero_counters"));

  // Initialize the atomic counters.
  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&](auto node) {
      for (auto edge : graph.edges(node, galois::MethodFlag::UNPROTECTED)) {
        auto neighbor = graph.getEdgeDst(edge);
        // TODO: binary search for starting point here.
        // Doing this directly is probably faster for nodes with
        // extremely small degree though.
        if (neighbor <= node) continue;
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

  // Now actually compute the ilu factorization.
  galois::for_each(
    galois::iterate(starting_nodes.begin(), starting_nodes.end()),
    [&](auto node, auto &context) noexcept {
      // Can use memory order relaxed all through here because the work list
      // does the atomic acquire/release when a work item (or chunk of work items)
      // is stolen from the thread that created it.
      assert(graph.getData(node, galois::MethodFlag::UNPROTECTED).load(std::memory_order_relaxed));
      for (auto edge : graph.edges(node, galois::MethodFlag::UNPROTECTED)) {
        auto neighbor = graph.getEdgeDst(edge);
        // Initialize to NaN in case it's ever used before being initialized.
        double diag_val = std::numeric_limits<double>::quiet_NaN();
        if (neighbor < node) {
          for (auto neighbor_edge : graph.edges(neighbor, galois::MethodFlag::UNPROTECTED)) {
            auto neighbor_of_neighbor = graph.getEdgeDst(neighbor_edge);
            // TODO: binary search to find starting position here?
            // Again, direct iteration will probably be faster for
            // really low degree nodes.
            if (neighbor_of_neighbor <= neighbor) {
              continue;
            }
            // TODO: Again, do binary search for this?
            for (auto potential_back_edge : graph.edges(neighbor_of_neighbor, galois::MethodFlag::UNPROTECTED)) {
              auto third_neighbor = graph.getEdgeDst(potential_back_edge);
              if (third_neighbor == node) {
                graph.getEdgeData(potential_back_edge) -= graph.getEdgeData(neighbor_edge) * graph.getEdgeData(edge);
              }
            }
          }
        } else if (neighbor == node) {
          diag_val = graph.getEdgeData(edge);
        } else {
          graph.getEdgeData(edge) /= diag_val;
          if (!(graph.getData(neighbor, galois::MethodFlag::UNPROTECTED).fetch_sub(1, std::memory_order_relaxed) - 1)) {
            context.push(neighbor);
          }
        }
      }
    },
    galois::loopname("ilu"),
    galois::no_conflicts(),
    galois::wl<galois::worklists::PerSocketChunkFIFO<64>>());
}
