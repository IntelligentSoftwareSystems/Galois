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

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

// Silence erroneous warnings from within Boost headers
// that show up with gcc 8.1.
#pragma GCC diagnostic ignored "-Wparentheses"
// This warning triggers with the assert(("explanation", check));
// syntax since the left hand argument has no side-effects.
// I prefer using the comma operator over && though because
// the parentheses are more readable, so I'm silencing
// the warning for this file.
#pragma GCC diagnostic ignored "-Wunused-value"

// Vendored from an old version of LLVM for Lonestar app command line handling.
#include "llvm/Support/CommandLine.h"

#include <galois/Galois.h>
#include <galois/graphs/FileGraph.h>
#include <galois/graphs/Graph.h>
#include <galois/graphs/LCGraph.h>

#include "Lonestar/BoilerPlate.h"

static char const *name = "Boltzmann Equation sweeps";
static char const *desc =
    "Computes a numerical solution to the Boltzmann Equation using "
    "the sweeps iterative method.";
static char const *url = "sweeps";

static llvm::cl::opt<std::size_t> nx{"nx", llvm::cl::desc("number of cells in x direction"), llvm::cl::init(10u)};
static llvm::cl::opt<std::size_t> ny{"ny", llvm::cl::desc("number of cells in y direction"), llvm::cl::init(10u)};
static llvm::cl::opt<std::size_t> nz{"nz", llvm::cl::desc("number of cells in z direction"), llvm::cl::init(10u)};
static llvm::cl::opt<double> freq_min{"freq_min", llvm::cl::desc("minimum frequency"), llvm::cl::init(.01)};
static llvm::cl::opt<double> freq_max{"freq_max", llvm::cl::desc("maximum frequency"), llvm::cl::init(1.)};
static llvm::cl::opt<std::size_t> num_groups{"num_groups", llvm::cl::desc("number of frequency groups"), llvm::cl::init(4u)};
static llvm::cl::opt<std::size_t> num_vert_directions{"num_vert_directions", llvm::cl::desc("number of vertical directions"), llvm::cl::init(32u)};
static llvm::cl::opt<std::size_t> num_horiz_directions{"num_horiz_directions", llvm::cl::desc("number of horizontal directions."), llvm::cl::init(32u)};
static llvm::cl::opt<std::size_t> maxiters{"maxiters", llvm::cl::desc("maximum number of iterations"), llvm::cl::init(100u)};

// Some helper functions for atomic operations with doubles:
// TODO: try switching these to a load/compute/load/compare/CAS
// style loop and see if it speeds it up.

// Atomically do += operation on a double.
void atomic_relaxed_double_increment(std::atomic<double> base, double increment) noexcept {
  auto current = base.load(std::memory_order_relaxed);
  while (!base.compare_exchange_weak(current, current + increment, std::memory_order_relaxed, std::memory_order_relaxed));
}

// Atomically do base = max(base, newval)
void atomic_relaxed_update_max(std::atomic<double> base, double newval) noexcept {
  auto current = base.load(std::memory_order_relaxed);
  while (current != std::max(current, newval)) {
    base.compare_exchange_weak(current, newval, std::memory_order_relaxed, std::memory_order_relaxed);
  }
}

// TODO: We need a graph type with dynamically sized node/edge data for this problem.
// For now, indexing into a separate data structure will have to be sufficient.

// Note: I'm going to use a CSR graph, so each node will already have a
// unique integer id that can be used to index other data structures, so
// it's not necessary to store anything other than the actual topology
// on the graph nodes. Since the rest of the data is dynamically sized,
// it'll be in a separate array.

// Each edge holds the unit normal pointing outward
// from the corresponding source cell in the graph.
// Note: this will be the negative of the vector stored
// on the edge coming the opposite direction.
// Note: In the regular grid case, this info is mostly redundant.
// It's still used here to determine the dependency direction
// for each discrete radiation direction.
// Note: The sweeping direction for each edge could just be
// pre-computed, but that'd noticeably increase storage requirements.
// TODO: Try caching sweep directions and see if it's any better.
// TODO: Would shared edge data help at all here?
using edge_t = std::array<double, 3>;

// This type is used to allocate buffers where
// some elements are doubles and some are atomic counters
// and the offsets between them are only known at runtime.
union direction_buffer_element_t {
  double magnitude;
  std::atomic<std::size_t> counter;
};
// Both these limitations could be lifted,
// but in the interest of keeping the buffer management
// code simple, I'm just going to assume them.
static_assert(sizeof(std::atomic<std::size_t>) <= sizeof(double),
              "Current buffer allocation code assumes atomic "
              "counters smaller than sizeof(double).");
static_assert(std::is_trivial_v<std::atomic<std::size_t>> &&
              std::is_standard_layout_v<std::atomic<std::size_t>>,
              "Current buffer allocation code assumes no special "
              "construction/deletion code is needed for atomic counters.");

// Note: in this representation of the mesh,
// boundaries are "cells" that
// have only one outgoing edge.
// No sentinel is actually needed.
struct node_t {
  // opaque pointer to:
  // memory block with one atomic per
  // direction (to track remaining dependencies),
  // and num_groups number of doubles.
  // It'd be nice to not use an opaque pointer here,
  // but solving that would require adding a bunch
  // of extra metadata or doing some non-obvious templating.
  direction_buffer_element_t *magnitudes_and_counters = nullptr;
  // Amounts of scattering in each direction.
  // As a simplifying assumption, I'm assuming that
  // radiation that scatters from any direction is equally likely to scatter into
  // any direction at all, so everything can be accumulated into a single term.
  // In general, there could be a scattering source term for every direction,
  // but I'm assuming that they are all equal.
  // On even iterations use scattering term 1 and accumulate into scattering term 0.
  // Do the opposite for the odd iterations.
  std::atomic<double> previous_accumulated_scattering = 0.;
  std::atomic<double> currently_accumulating_scattering = 0.;
  // Rather than do a separate pass over the data to zero out the previous
  // scattering term for the next iteration, just count the number of
  // times that the old scattering term is used. Once all the computations
  // for a given round are done (i.e. when this counter hits zero),
  // it's safe to move the currently accumulating scattering term value
  // into the previous accumulated scattering variable and
  // reset the dependency counters that track when the upstream dependencies
  // for a given cell/direction have completed.
  // TODO: Swapping values separately could be faster. Try and see.
  std::atomic<std::size_t> scatter_use_counter = 0u;
  // TODO: It wouldn't be hard to just add an iteration counter here too
  // and then just let the whole thing run fully asynchronously. Try that some time.
};

// No fine-grained locks built into the graph.
// Use atomics for ALL THE THINGS!
using graph_t = typename galois::graphs::LC_CSR_Graph<node_t, edge_t>::with_no_lockable<true>::type;

// Routine to initialize graph topology and face normals.
auto generate_grid(graph_t &built_graph, std::size_t nx, std::size_t ny, std::size_t nz) noexcept {

  galois::graphs::FileGraphWriter temp_graph;
  // Each node represents a grid cell.
  // Each edge represents a face.
  // Ghost nodes are added to represent the exterior
  // of the domain on the other side of each face.
  // This is for boundary condition handling.
  std::size_t num_outer_faces = (nx * ny + ny * nz + nx * nz) * 2;
  std::size_t num_cells = nx * ny * nz;
  std::size_t num_nodes = num_cells + num_outer_faces;
  std::size_t num_edges = 6 * nx * ny * nz + num_outer_faces;
  temp_graph.setNumNodes(num_nodes);
  temp_graph.setNumEdges(num_edges);
  temp_graph.setSizeofEdgeData(galois::LargeArray<graph_t::edge_data_type>::size_of::value);
  galois::LargeArray<graph_t::edge_data_type> edge_data;
  edge_data.create(num_edges);

  // Interior cells will have degree 6
  // since they will have either other cells or
  // ghost cells on every side.
  temp_graph.phase1();
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * nx * ny + j * ny + k;
        for (std::size_t l = 0; l < 6; l++) {
          temp_graph.incrementDegree(id);
        }
      }
    }
  }
  // Set the degrees for all the ghost cells to 1.
  // No ghost cell should share a boundary with more
  // than one actual cell in the domain.
  for (std::size_t id = num_cells; id < num_nodes; id++) {
    temp_graph.incrementDegree(id);
  }

  // Now that the degree of each node is known,
  // fill in the actual topology.
  // Also fill in the node data with the vector
  // normal to the face, going out from the current cell.
  temp_graph.phase2();
  std::size_t xy_low_face_start = num_cells;
  std::size_t xy_high_face_start = xy_low_face_start + nx * ny;
  std::size_t yz_low_face_start = xy_high_face_start + nx * ny;
  std::size_t yz_high_face_start = yz_low_face_start + ny * nz;
  std::size_t xz_low_face_start = yz_high_face_start + ny * nz;
  std::size_t xz_high_face_start = xz_low_face_start + nx * nz;
  assert(("Error in logic for dividing up node ids for exterior faces.",
          num_nodes == xz_high_face_start + nx * nz));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * nx * ny + j * ny + k;
        if (i > 0) {
          edge_data.set(temp_graph.addNeighbor(id, id - ny * nz), {-1., 0., 0.});
        } else {
          std::size_t ghost_id = yz_low_face_start + j * ny + k;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {-1., 0., 0.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {1., 0., 0.});
        }
        if (i < nx - 1) {
          edge_data.set(temp_graph.addNeighbor(id, id + ny * nz), {1., 0., 0.});
        } else {
          std::size_t ghost_id = yz_high_face_start + j * ny + k;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {1., 0., 0.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {-1., 0., 0.});
        }
        if (j > 0) {
          edge_data.set(temp_graph.addNeighbor(id, id - nz), {0., -1., 0.});
        } else {
          std::size_t ghost_id = xz_low_face_start + i * nx + k;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {0., -1., 0.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {0., 1., 0.});
        }
        if (j < ny - 1) {
          edge_data.set(temp_graph.addNeighbor(id, id + nz), {0., 1., 0.});
        } else {
          std::size_t ghost_id = xz_high_face_start + i * nx + k;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {0., 1., 0.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {0., -1., 0.});
        }
        if (k > 0) {
          edge_data.set(temp_graph.addNeighbor(id, id - 1), {0., 0., -1.});
        } else {
          std::size_t ghost_id = xy_low_face_start + i * nx + j;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {0., 0., -1.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {0., 0., 1.});
        }
        if (k < nz - 1) {
          edge_data.set(temp_graph.addNeighbor(id, id + 1), {0., 0., 1.});
        } else {
          std::size_t ghost_id = xy_high_face_start + i * nx + j;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {0., 0., 1.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {0., 0., -1.});
        }
      }
    }
  }

  // TODO: is it possible to set the edge data
  // during construction without copying here?
  auto *rawEdgeData = temp_graph.finish<graph_t::edge_data_type>();
  std::uninitialized_copy(std::make_move_iterator(edge_data.begin()), std::make_move_iterator(edge_data.end()), rawEdgeData);

  galois::graphs::readGraph(built_graph, temp_graph);
  return std::make_tuple(num_nodes, num_cells, num_outer_faces);
}

// Series of asserts to check that the graph construction
// code is actually working.
void assert_face_directions(graph_t &graph, std::size_t num_nodes, std::size_t num_cells, std::size_t num_outer_faces) noexcept {
  assert(("artimetic error in mesh generation.",
          num_cells + num_outer_faces == num_nodes));
  assert(("Mismatch between graph size and number of nodes",
          num_nodes == graph.size()));
  for (auto node : graph) {
    // std::distance ought to work on the edge iterator,
    // but it doesn't, so count the degree manually.
    // TODO: fix this in the library.
    std::size_t degree = 0;
    // Note: Rely on node type to decay to std::size_t here.
    // This is a feature of the Galois CSR graph, but not other graph types.
    assert(("Unexpectedly large node id.",
            node < num_nodes));
    for (auto edge : graph.edges(node, galois::MethodFlag::UNPROTECTED)) {
      // More verification is possible, but here's a sanity check.
      // Confirm that edges coming back the other direction
      // have edge data that is the negative of the edge data
      // on the outgoing edge.
      auto &edge_data = graph.getEdgeData(edge);
      auto destination = graph.getEdgeDst(edge);
      for (auto neighbor_edge : graph.edges(destination, galois::MethodFlag::UNPROTECTED)) {
        if (graph.getEdgeDst(neighbor_edge) == node) {
          auto &back_edge_data = graph.getEdgeData(neighbor_edge);
          for (std::size_t i = 0; i < 3; i++) {
            assert(("Back edge must be the negative of the forward edge.",
                    edge_data[i] == -back_edge_data[i]));
          }
          goto back_edge_found;
        }
      }
      // If loop exits without jumping past this assert,
      // no back edge was found.
      assert(("Edge with no back edge found.", false));
      back_edge_found:;
      degree++;
    }
    assert(("Found node with incorrect degree. "
            "Interior nodes should all have "
            "degree 6 and boundary nodes "
            "should all have degree 1.",
            degree == 6 && node < num_cells ||
            degree == 1 && node >= num_cells));
  }
}

// Idk why this hasn't been standardized yet, but here it is.
static constexpr double pi = 3.1415926535897932384626433832795028841971693993751;

// Generate discrete directions corresponding to an
// equal area partition of the sphere.
// Follow the forumlas from
// https://stackoverflow.com/a/24458877/1935144
// There are many more sophisticated things that could be done here,
// but this is good enough since this is really intended as a
// performance proxy anyway.
auto generate_directions(std::size_t latitude_divisions, std::size_t longitude_divisions) noexcept {
  std::size_t num_directions = latitude_divisions * longitude_divisions;
  auto directions = std::make_unique<std::array<double, 3>[]>(num_directions);
  auto average_longitudes = std::make_unique<double[]>(longitude_divisions);

  // For floating point precision improvement it may be
  // better to rewrite these things in terms of std::lerp,
  // but that's only available in c++20.
  for (std::size_t k = 0; k < longitude_divisions; k++) {
    average_longitudes[k] = (double(k + .5) / longitude_divisions) * (2 * pi);
  }

  for (std::size_t j = 0; j < latitude_divisions; j++) {
    // Since the even spacing is in the sine of the latitude,
    // compute the center point in the sine as well to better
    // match what the average direction is for that
    // particular piece of the partition.
    // TODO: actually prove that this is the right thing to do.
    double average_latitude = std::asin(-1 + (j + .5) / (.5 * latitude_divisions));
    for (std::size_t k = 0; k < longitude_divisions; k++) {
      std::size_t direction_index = j * longitude_divisions + k;
      double average_longitude = average_longitudes[k];
      directions[direction_index] = {std::cos(average_longitude), std::sin(average_longitude), std::sin(average_latitude)};
      // Could renormalize here if really precise computation is desired.
    }
  }

  // Sanity check: make sure average in each direction is 0.
  std::array<double, 3> averages = {0., 0., 0.};
  for (std::size_t i = 0; i < num_directions; i++) {
    for (std::size_t j = 0; j < 3; j++) {
      averages[j] += directions[i][j];
    }
  }
  for (std::size_t j = 0; j < 3; j++) {
    averages[j] /= num_directions;
    // This is a pretty generous tolerance,
    // so if this doesn't pass, something is
    // very likely wrong.
    assert(("Dubious values from direction discretization.",
            std::abs(averages[j]) < 1E-7));
  }

  return std::make_tuple(std::move(directions), num_directions);
}

// Of the given discrete directions, find the one
// that's closest to {1., 0., 0.};
std::size_t find_x_direction(std::array<double, 3> const *directions, std::size_t num_directions) noexcept {
  auto comparison = [](auto a1, auto a2) noexcept {return a1[0] < a2[0];};
  return std::max_element(directions, directions + num_directions, comparison) - directions;
}

int main(int argc, char**argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url);

  graph_t graph;
  auto [num_nodes, num_cells, num_outer_faces] = generate_grid(graph, nx, ny, nz);
  assert_face_directions(graph, num_nodes, num_cells, num_outer_faces);
  // node id at least as large as num_cells
  // indicates a boundary node.
  auto ghost_threshold = num_cells;
  auto [directions, num_directions] = generate_directions(num_vert_directions, num_horiz_directions);
  auto approx_x_direction_index = find_x_direction(directions.get(), num_directions);

}

