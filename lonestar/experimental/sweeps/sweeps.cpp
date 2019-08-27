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
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
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

#include <galois/Galois.h>
#include <galois/graphs/FileGraph.h>
#include <galois/graphs/Graph.h>
#include <galois/graphs/LCGraph.h>

// Vendored from an old version of LLVM for Lonestar app command line handling.
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

static char const* name = "Boltzmann Equation sweeps";
static char const* desc =
    "Computes a numerical solution to the Boltzmann Equation using "
    "the sweeps iterative method.";
static char const* url = "sweeps";

static llvm::cl::opt<unsigned long long> nx{
    "nx", llvm::cl::desc("number of cells in x direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned long long> ny{
    "ny", llvm::cl::desc("number of cells in y direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned long long> nz{
    "nz", llvm::cl::desc("number of cells in z direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned long long> num_groups{
    "num_groups", llvm::cl::desc("number of frequency groups"),
    llvm::cl::init(4u)};
static llvm::cl::opt<unsigned long long> num_vert_directions{
    "num_vert_directions", llvm::cl::desc("number of vertical directions"),
    llvm::cl::init(16u)};
static llvm::cl::opt<unsigned long long> num_horiz_directions{
    "num_horiz_directions", llvm::cl::desc("number of horizontal directions."),
    llvm::cl::init(32u)};
static llvm::cl::opt<unsigned long long> num_iters{
    "num_iters", llvm::cl::desc("number of iterations"), llvm::cl::init(10u)};
static llvm::cl::opt<double> pulse_strength{
    "pulse_strength", llvm::cl::desc("radiation pulse strength"),
    llvm::cl::init(1.)};
static llvm::cl::opt<double> absorption_coef{
    "absorption_coef",
    llvm::cl::desc("Absorption coefficient (between 0 and 1), absorption and "
                   "scattering must sum to less than 1."),
    llvm::cl::init(.01)};
static llvm::cl::opt<double> scattering_coef{
    "scattering_coef",
    llvm::cl::desc("Scattering coefficient (between 0 and 1), absorption and "
                   "scattering must sum to less than 1."),
    llvm::cl::init(.25)};
static llvm::cl::opt<bool> print_convergence{
    "print_convergence",
    llvm::cl::desc("Print the max change in amount of scattering at a given "
                   "each iteration."),
    llvm::cl::init(false)};
static llvm::cl::opt<std::string> scattering_outfile{
    "scattering_outfile",
    llvm::cl::desc(
        "Text file name to use to write final scattering term values "
        "after each step."),
    llvm::cl::init("")};

// Some helper functions for atomic operations with doubles:
// TODO: try switching these to a load/compute/load/compare/CAS
// style loop and see if it speeds it up.

// Atomically do += operation on a double.
void atomic_relaxed_double_increment(std::atomic<double>& base,
                                     double increment) noexcept {
  auto current = base.load(std::memory_order_relaxed);
  /*decltype(current) previous;
  while (true) {
    previous = current;
    current = base.load(std::memory_order_relaxed);
    if (previous == current) {
      if (base.compare_exchange_weak(current, current + increment,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
        break;
      }
    }
  }*/
  while (!base.compare_exchange_weak(current, current + increment,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed))
    ;
}

// Atomically do base = max(base, newval)
void atomic_relaxed_double_max(std::atomic<double>& base,
                               double newval) noexcept {
  auto current = base.load(std::memory_order_relaxed);
  while (current != std::max(current, newval)) {
    base.compare_exchange_weak(current, newval, std::memory_order_relaxed,
                               std::memory_order_relaxed);
  }
}

// TODO: In Galois, we need a graph type with dynamically sized
// node/edge data for this problem. For now, indexing into a
// separate data structure will have to be sufficient.

// Note: I'm going to use a CSR graph, so each node will already have a
// unique std::size_t id that can be used to index other data structures.
// I'll also use a std::size_t cutoff to distinguish between ghost cells
// that only exist to provide boundary condition data and actual cells.

// Each edge holds the unit normal pointing outward
// from the corresponding source cell in the graph.
// Note: this will be the negative of the vector stored
// on the edge coming the opposite direction.
// Note: In the regular grid case, this could be considered redundant,
// but this code hopefully will be adapted to handle irregular
// geometry at some point.
// Note: The sweeping direction for each direction along each edge
// could just be pre-computed, but that'd noticeably increase
// storage requirements.
// TODO: Try caching sweep directions and see if it's any better.
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
// They are identified by having a node id
// above a certain threshold.
struct node_t {
  // opaque pointer to:
  // memory block with one atomic per
  // direction (to track remaining dependencies),
  // and num_groups number of doubles.
  // It'd be nice to not use an opaque pointer here,
  // but solving that would require adding a bunch
  // of extra metadata or doing some extensive templating.
  direction_buffer_element_t* magnitudes_and_counters = nullptr;
  // Amounts of scattering in all directions. For simplicity, I'm assuming that
  // radiation that scatters from any direction is equally likely to scatter
  // into any direction at all, so everything can be accumulated into a single
  // term. In general, there could be a scattering source term for every
  // direction, but I'm assuming that they are all equal.
  double previous_accumulated_scattering                = 0.;
  std::atomic<double> currently_accumulating_scattering = 0.;
  // Rather than do a separate pass over the data to zero out the previous
  // scattering term for the next iteration, just count the number of
  // times that the old scattering term is used. Once all the computations
  // for a given round are done (i.e. when this counter hits zero),
  // it's safe to move the currently accumulating scattering term value
  // into the previous accumulated scattering variable and
  // reset the dependency counters that track when the upstream dependencies
  // for a given cell/direction have completed.
  // TODO: Could swapping values separately be faster?
  // TODO: What about pre-computing the number of incoming edges instead
  // of re-computing it at each iteration?
  std::atomic<std::size_t> scatter_use_counter = 0u;
  // TODO: It wouldn't be hard to just add an iteration counter here too
  // and then just let the whole thing run fully asynchronously. Try that some
  // time. Termination detection won't be quite as obvious though if you
  // want to end when the error gets below a threshold.
  // For the time being though, this app just runs a set number of iterations.
};

// No fine-grained locks built into the graph.
// Use atomics for ALL THE THINGS!
using graph_t =
    typename galois::graphs::LC_CSR_Graph<node_t,
                                          edge_t>::with_no_lockable<true>::type;

// The label for a piece of work in the for_each loop
// that actually runs the sweeping computation.
struct work_t {
  std::size_t node_index;
  std::size_t direction_index;
};

// Routine to initialize graph topology and face normals.
auto generate_grid(graph_t& built_graph, std::size_t nx, std::size_t ny,
                   std::size_t nz) noexcept {

  galois::graphs::FileGraphWriter temp_graph;
  // Each node represents a grid cell.
  // Each edge represents a face.
  // Ghost nodes are added to represent the exterior
  // of the domain on the other side of each face.
  // This is for boundary condition handling.
  std::size_t num_outer_faces = (nx * ny + ny * nz + nx * nz) * 2;
  std::size_t num_cells       = nx * ny * nz;
  std::size_t num_nodes       = num_cells + num_outer_faces;
  std::size_t num_edges       = 6 * nx * ny * nz + num_outer_faces;
  temp_graph.setNumNodes(num_nodes);
  temp_graph.setNumEdges(num_edges);
  temp_graph.setSizeofEdgeData(
      galois::LargeArray<graph_t::edge_data_type>::size_of::value);
  galois::LargeArray<graph_t::edge_data_type> edge_data;
  edge_data.create(num_edges);

  // Interior cells will have degree 6
  // since they will have either other cells or
  // ghost cells on every side.
  // This condition isn't true in irregular meshes,
  // but that'd be a separate mesh generation routine.
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
  std::size_t xy_low_face_start  = num_cells;
  std::size_t xy_high_face_start = xy_low_face_start + nx * ny;
  std::size_t yz_low_face_start  = xy_high_face_start + nx * ny;
  std::size_t yz_high_face_start = yz_low_face_start + ny * nz;
  std::size_t xz_low_face_start  = yz_high_face_start + ny * nz;
  std::size_t xz_high_face_start = xz_low_face_start + nx * nz;
  assert(("Error in logic for dividing up node ids for exterior faces.",
          num_nodes == xz_high_face_start + nx * nz));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * nx * ny + j * ny + k;
        if (i > 0) {
          edge_data.set(temp_graph.addNeighbor(id, id - ny * nz),
                        {-1., 0., 0.});
        } else {
          std::size_t ghost_id = yz_low_face_start + j * nz + k;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {1., 0., 0.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {-1., 0., 0.});
        }
        if (i < nx - 1) {
          edge_data.set(temp_graph.addNeighbor(id, id + ny * nz), {1., 0., 0.});
        } else {
          std::size_t ghost_id = yz_high_face_start + j * nz + k;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {-1., 0., 0.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {1., 0., 0.});
        }
        if (j > 0) {
          edge_data.set(temp_graph.addNeighbor(id, id - nz), {0., -1., 0.});
        } else {
          std::size_t ghost_id = xz_low_face_start + i * nz + k;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {0., 1., 0.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {0., -1., 0.});
        }
        if (j < ny - 1) {
          edge_data.set(temp_graph.addNeighbor(id, id + nz), {0., 1., 0.});
        } else {
          std::size_t ghost_id = xz_high_face_start + i * nz + k;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {0., -1., 0.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {0., 1., 0.});
        }
        if (k > 0) {
          edge_data.set(temp_graph.addNeighbor(id, id - 1), {0., 0., -1.});
        } else {
          std::size_t ghost_id = xy_low_face_start + i * ny + j;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {0., 0., 1.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {0., 0., -1.});
        }
        if (k < nz - 1) {
          edge_data.set(temp_graph.addNeighbor(id, id + 1), {0., 0., 1.});
        } else {
          std::size_t ghost_id = xy_high_face_start + i * ny + j;
          edge_data.set(temp_graph.addNeighbor(ghost_id, id), {0., 0., -1.});
          edge_data.set(temp_graph.addNeighbor(id, ghost_id), {0., 0., 1.});
        }
      }
    }
  }

  // TODO: is it possible to set the edge data
  // during construction without copying here?
  auto* rawEdgeData = temp_graph.finish<graph_t::edge_data_type>();
  std::uninitialized_copy(std::make_move_iterator(edge_data.begin()),
                          std::make_move_iterator(edge_data.end()),
                          rawEdgeData);

  galois::graphs::readGraph(built_graph, temp_graph);
  return std::make_tuple(num_nodes, num_cells, num_outer_faces,
                         xy_low_face_start, xy_high_face_start,
                         yz_low_face_start, yz_high_face_start,
                         xz_low_face_start, xz_high_face_start);
}

// Series of asserts to check that the graph construction
// code is actually working.
void assert_face_directions(graph_t& graph, std::size_t num_nodes,
                            std::size_t num_cells,
                            std::size_t num_outer_faces) noexcept {
#if !defined(NDEBUG)

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
    // This is a feature of the Galois CSR graph, but it's
    // not necessarily true when using other graph types.
    assert(("Unexpectedly large node id.", node < num_nodes));
    for (auto edge : graph.edges(node, galois::MethodFlag::UNPROTECTED)) {
      // More verification is possible, but here's a sanity check.
      // Confirm that edges coming back the other direction
      // have edge data that is the negative of the edge data
      // on the outgoing edge.
      auto& edge_data  = graph.getEdgeData(edge);
      auto destination = graph.getEdgeDst(edge);
      for (auto neighbor_edge :
           graph.edges(destination, galois::MethodFlag::UNPROTECTED)) {
        if (graph.getEdgeDst(neighbor_edge) == node) {
          auto& back_edge_data = graph.getEdgeData(neighbor_edge);
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
    assert(
        ("Found node with incorrect degree. "
         "Interior nodes should all have "
         "degree 6 and boundary nodes "
         "should all have degree 1.",
         degree == 6 && node < num_cells || degree == 1 && node >= num_cells));
  }

#endif // !defined(NDEBUG)
}

// Idk why this hasn't been standardized in C++ yet, but here it is.
static constexpr double pi =
    3.1415926535897932384626433832795028841971693993751;

// Generate discrete directions corresponding to an
// equal area partition of the sphere.
// Follow the forumlas from
// https://stackoverflow.com/a/24458877/1935144
// There are many more sophisticated things that could be done here,
// but this is good enough since this is intended for use mainly as a
// performance proxy for better simulation codes anyway.
auto generate_directions(std::size_t latitude_divisions,
                         std::size_t longitude_divisions) noexcept {
  assert(latitude_divisions > 0);
  assert(longitude_divisions > 0);
  std::size_t num_directions = latitude_divisions * longitude_divisions;
  auto directions = std::make_unique<std::array<double, 3>[]>(num_directions);
  auto average_longitudes = std::make_unique<double[]>(longitude_divisions);

  // For floating point precision improvement it may be
  // better to rewrite these things in terms of std::lerp,
  // but that's only available in c++20.
  // Offset if the number of latitudes is small to avoid running
  // directly along the axes of the grids.
  // "is_incoming" now properly handles the case where propagation is
  // exactly orthogonal to a given face, but this avoids benchmarking
  // the unusual branches of execution it needs to take there.
  double longitude_offset = longitude_divisions <=2 ? .25 * pi : 0.;
  for (std::size_t k = 0; k < longitude_divisions; k++) {
    average_longitudes[k] = (double(k + .5) / longitude_divisions) * (2 * pi) + longitude_offset;
  }

  // For the latitudes, if there is only one latitude used,
  // tilt the coordinate axes about the x axis to avoid generating a bunch
  // of directions that are all orthogonal to the z axis used in the
  // generated mesh.
  // Use formula from https://math.stackexchange.com/a/1742758/89171 for
  // the single latitude case.
  if (latitude_divisions == 1) {
    double tilt = .25 * pi;
    for (std::size_t k = 0; k < latitude_divisions; k++) {
      double pre_rotated_average_longitude = average_longitudes[k];
      directions[k] = {
        std::cos(pre_rotated_average_longitude),
        std::sin(pre_rotated_average_longitude) * std::cos(tilt),
        -std::sin(pre_rotated_average_longitude) * std::sin(tilt)};
    }
  } else {
    for (std::size_t j = 0; j < latitude_divisions; j++) {
      // Since the even spacing is in the sine of the latitude,
      // compute the center point in the sine as well to better
      // match what the average direction is for that
      // particular piece of the partition.
      // TODO: actually prove that this is the right thing to do.
      double average_latitude =
          std::asin(-1 + (j + .5) / (.5 * latitude_divisions));
      for (std::size_t k = 0; k < longitude_divisions; k++) {
        std::size_t direction_index = j * longitude_divisions + k;
        double average_longitude    = average_longitudes[k];
        directions[direction_index] = {
            std::cos(average_longitude) * std::cos(average_latitude),
            std::sin(average_longitude) * std::cos(average_latitude),
            std::sin(average_latitude)};
        // Could renormalize here if really precise computation is desired.
      }
    }
  }

  // Sanity check: make sure average in each direction is 0.
  auto elwise_sum = [](auto a1, auto a2) noexcept {
    return std::array<double, 3>({a1[0] + a2[0], a1[1] + a2[1], a1[2] + a2[2]});
  };
  std::array<double, 3> identity = {0., 0., 0.};
  auto totals =
      std::accumulate(directions.get(), directions.get() + num_directions,
                      identity, elwise_sum);
  for (std::size_t j = 0; j < 3; j++) {
    totals[j] /= num_directions;
    // This is a pretty generous tolerance,
    // so if this doesn't pass, something is
    // very wrong.
    assert(("Dubious values from direction discretization.",
            std::abs(totals[j]) < 1E-7) ||
            latitude_divisions == 1 ||
            longitude_divisions == 1);
  }

  return std::make_tuple(std::move(directions), num_directions);
}

bool is_incoming(std::array<double, 3> direction,
                 std::array<double, 3> face_normal) noexcept {
  auto inner_prod = direction[0] * face_normal[0] +
                    direction[1] * face_normal[1] +
                    direction[2] * face_normal[2];
  if (inner_prod < 0) {
    return true;
  } else if (inner_prod > 0) {
    return false;
  }
  // TODO: It may be better not to disambiguate at all and let
  // cells not have as many incoming edges.
  // That's fine from the point of view of the numerical method,
  // but some more restructuring would be needed to handle that
  // gracefully.
  // If they are exactly orthogonal, break ties by only using
  // the normal. Say negative sign indicates incoming.
  for (std::size_t dim_idx = 0; dim_idx < 3; dim_idx++) {
    if (face_normal[dim_idx] < 0) return true;
    if (face_normal[dim_idx] > 0) return false;
  }
  GALOIS_DIE("All zero face_normal passed to is_incoming. Can't disambiguate.");
}

// Of the given discrete directions, find the one
// that's closest to {1., 0., 0.};
std::size_t find_x_direction(std::array<double, 3> const* directions,
                             std::size_t num_directions) noexcept {
  auto comparison = [](auto a1, auto a2) noexcept { return a1[0] < a2[0]; };
  return std::max_element(directions, directions + num_directions, comparison) -
         directions;
}

int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url);

  graph_t graph;
  auto [num_nodes, num_cells, num_outer_faces, xy_low, xy_high, yz_low, yz_high,
        xz_low, xz_high] = generate_grid(graph, nx, ny, nz);
  assert_face_directions(graph, num_nodes, num_cells, num_outer_faces);
  // node id at least as large as num_cells
  // indicates a ghost cell.
  auto ghost_threshold = num_cells;
  auto [directions_owner, num_directions_binding] =
      generate_directions(num_vert_directions, num_horiz_directions);
  auto &directions = directions_owner;
  auto num_directions = num_directions_binding;
  auto approx_x_direction_index =
      find_x_direction(directions.get(), num_directions);

  // Now create buffers to hold all the radiation magnitudes and
  // the atomic counters used to trace dependencies.
  std::size_t num_per_element_and_direction = 1u + num_groups;
  std::size_t num_per_element = num_directions * num_per_element_and_direction;
  // Could potentially save a little space by reducing the storage allocated for
  // fluxes in ghost cells, but that'd be way more complicated and not
  // necessarily helpful.
  std::size_t buffer_size = num_nodes * num_per_element;
  galois::LargeArray<direction_buffer_element_t> radiation_magnitudes;
  radiation_magnitudes.create(buffer_size);
  // Okay to use reinterpret_cast here. See
  // https://stackoverflow.com/a/49495687/1935144
  std::fill(reinterpret_cast<double*>(radiation_magnitudes.begin()),
            reinterpret_cast<double*>(radiation_magnitudes.end()), 0.);

  // Index of center face on yz boundary.
  std::size_t yz_face_center_boundary = yz_low + (ny / 2) * nz + nz / 2;

  // Make the boundary condition be an intense ray entering the domain
  // in the middle of the low yz face in approximately the positive x direction.
  std::size_t boundary_pulse_index =
      yz_face_center_boundary * num_per_element +
      approx_x_direction_index * num_per_element_and_direction;
  radiation_magnitudes[boundary_pulse_index + 1].magnitude = pulse_strength;

  // Constants used in the differencing scheme at each cell/direction.
  std::array<double, 3> grid_spacing_inverses{static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)};

  // For the regular grid, this will just be the corners
  // (one work item for each direction that lies in the octant opposite
  // the corner)
  // Mimic the irregular case though by finding nodes with
  // no dependencies during the pass that initializes the sweep counters.
  galois::InsertBag<work_t> starting_nodes;
  // Initialize the dependency counters so that no cell/direction executes
  // before its predecessors have.
  // This uses a simple Galois parallel loop.
  galois::do_all(
      galois::iterate(graph.begin(), graph.end()),
      [&](auto node) noexcept {
        // Nothing to do on boundary nodes.
        if (node >= ghost_threshold)
          return;
        for (std::size_t dir_idx = 0; dir_idx < num_directions; dir_idx++) {
          std::atomic<std::size_t>& counter =
              radiation_magnitudes[num_per_element * node +
                                   num_per_element_and_direction * dir_idx]
                  .counter;
          std::size_t local_counter = 0;
          for (auto edge : graph.edges(node, galois::MethodFlag::UNPROTECTED)) {
            if (graph.getEdgeDst(edge) >= ghost_threshold)
              continue;
            local_counter += is_incoming(directions[dir_idx], graph.getEdgeData(edge));
          }
          // TODO: This stage is embarassingly parallel,
          // but writing the value as a size_t and then
          // later using it as an atomic is undefined behavior,
          // so, is there some way around this?
          counter.store(local_counter, std::memory_order_relaxed);
          if (!local_counter) {
            work_t work_item{node, dir_idx};
            starting_nodes.emplace(work_item);
          }
        }
        // Also set the counter for how many directions are remaining
        // on the current node.
        auto& node_data = graph.getData(node, galois::MethodFlag::UNPROTECTED);
        // TODO: Can this be done without atomics as well?
        node_data.scatter_use_counter.store(num_directions, std::memory_order_relaxed);
      },
      galois::loopname("Initialize counters"));

// Check that the counters are properly set.
#if !defined(NDEBUG)
  for (auto node : graph) {
    if (node >= ghost_threshold)
      continue;
    for (std::size_t dir_idx = 0; dir_idx < num_directions; dir_idx++) {
      std::size_t local_counter = 0;
      for (auto edge : graph.edges(node, galois::MethodFlag::UNPROTECTED)) {
        if (is_incoming(directions[dir_idx], graph.getEdgeData(edge)) && graph.getEdgeDst(edge) < ghost_threshold)
          local_counter++;
      }
      assert(("Dependency counter not set propertly.",
              local_counter ==
                  radiation_magnitudes[num_per_element * node +
                                       num_per_element_and_direction * dir_idx]
                      .counter));
    }
  }
#endif // !defined(NDEBUG)

  // Various sweeps papers mention that using topological depth as a
  // heuristic to make sure that parallelism opens up quickly is an important
  // optimization, but that isn't true with this implementation. Perhaps it'll
  // matter more in distributed settings or when using GPUs.
  // For shared memory though, a simple chunked FIFO queue is better.
  typedef galois::worklists::PerSocketChunkFIFO<64> PSchunk;

  // Set up separate buffers for accumulating the per-group fluxes.
  // This is just to hold temporaries inside the main parallel loop,
  // but it's better to avoid allocation there (or use an allocator
  // specifically designed for that use case).
  galois::substrate::PerThreadStorage<std::unique_ptr<double[]>>
      accumulation_buffers;
  galois::on_each([&](unsigned int tid, unsigned int nthreads) noexcept {
    *accumulation_buffers.getLocal() = std::make_unique<double[]>(num_groups);
  });

  std::atomic<double> global_abs_change = 0.;

  if (scattering_outfile != "") {
    // empty the output file of previous data.
    std::ofstream outfile{scattering_outfile, std::ofstream::trunc};
    if (!outfile)
      GALOIS_DIE("Unable to write to desired output file.");
  }

  // Iterations in the algorithm.
  // TODO: Try doing this whole thing asynchronously
  // instead of just using a parallel loop for each step.
  for (std::size_t current_iteration = 0; current_iteration < num_iters;
       current_iteration++) {
    // Main parallel loop.
    galois::for_each(
        galois::iterate(starting_nodes.begin(), starting_nodes.end()),
        [&](work_t work_item, auto& ctx) noexcept {
          auto [node, dir_idx] = work_item;
          auto& direction      = directions[dir_idx];
          assert(("Work item for ghost node generated erroneously.",
                  node < ghost_threshold));
          auto node_magnitude_idx =
              num_per_element * node + num_per_element_and_direction * dir_idx;
          auto& counter = radiation_magnitudes[node_magnitude_idx].counter;
          auto counter_check_val = counter.load(std::memory_order_acquire);
          if (counter_check_val) {
            GALOIS_DIE("Work item asked to run before it was actually ready.");
          }
          auto& node_data =
              graph.getData(node, galois::MethodFlag::UNPROTECTED);
          // Re-count incoming edges during this computation.
          std::size_t incoming_edges = 0;
          // Reset accumulation buffers.
          auto& new_magnitude_numerators = *accumulation_buffers.getLocal();
          std::fill(new_magnitude_numerators.get(),
                    new_magnitude_numerators.get() + num_groups,
                    node_data.previous_accumulated_scattering);
          // Partial computation of the coefficient that will divide the
          // previous term later.
          double new_magnitude_denominator = absorption_coef + scattering_coef;
          std::size_t max_size_t = std::numeric_limits<std::size_t>::max();
          // Use max size_t value for initialization here mainly to make things segfault as fast
          // as possible if one of them isn't set in the loop over the neighbors.
          std::array<std::size_t, 3> downstream_neighbors{max_size_t, max_size_t, max_size_t};
          std::size_t downstream_index = 0;
          for (auto edge : graph.edges(node, galois::MethodFlag::UNPROTECTED)) {
            auto other_node = graph.getEdgeDst(edge);
            auto& face_normal = graph.getEdgeData(edge);
            if (!is_incoming(direction, face_normal)) {
              downstream_neighbors[downstream_index++] = other_node;
              continue;
            }
            if (other_node < ghost_threshold)
              incoming_edges++;
            // More partial computation of this node's estimated radiative
            // fluxes in the given direction. This time based off of the
            // incoming fluxes from its upwind neighbors.
            // TODO: Try storing this direction info on the edge.
            std::size_t axis =
                face_normal[0] != 0 ? 0 : (face_normal[1] != 0 ? 1 : 2);
            double sign      = std::signbit(face_normal[axis]) ? -1. : 1.;
            double term_coef = direction[axis] * sign;
            new_magnitude_denominator += term_coef * grid_spacing_inverses[axis];
            std::size_t other_magnitude_idx =
                num_per_element * other_node +
                num_per_element_and_direction * dir_idx;
            for (std::size_t i = 0; i < num_groups; i++) {
              std::size_t other_mag_and_group_idx = other_magnitude_idx + i + 1;
              double& other_magnitude =
                  radiation_magnitudes[other_mag_and_group_idx].magnitude;
              new_magnitude_numerators[i] +=
                  term_coef * other_magnitude * grid_spacing_inverses[axis];
            }
          }
          assert(("Wrong number of downstream neighbors.", downstream_index == 3));
          // Finish computing new flux magnitude.
          // Also compute a new scattering amount
          // for use in the next iteration based
          // off of this new flux.
          double scattering_contribution = 0.;
          double scattering_contribution_coef =
              scattering_coef / (num_groups * num_directions);
          for (std::size_t i = 0; i < num_groups; i++) {
            std::size_t node_mag_and_group_idx = node_magnitude_idx + i + 1;
            double& node_magnitude =
                radiation_magnitudes[node_mag_and_group_idx].magnitude;
            node_magnitude =
                new_magnitude_numerators[i] / new_magnitude_denominator;
            scattering_contribution +=
                scattering_contribution_coef * node_magnitude;
          }
          // Reset dependency counter for next pass.
          counter.store(incoming_edges, std::memory_order_relaxed);
          // Update scattering source for use in next step.
          auto& scattering_atomic = node_data.currently_accumulating_scattering;
          atomic_relaxed_double_increment(scattering_atomic,
                                          scattering_contribution);

          if (!(node_data.scatter_use_counter.fetch_sub(1, std::memory_order_relaxed) - 1)) {
            // Reset counter for next time step.
            node_data.scatter_use_counter.store(num_directions, std::memory_order_relaxed);
            double abs_change =
                std::abs(node_data.currently_accumulating_scattering.load(std::memory_order_relaxed) -
                         node_data.previous_accumulated_scattering);
            atomic_relaxed_double_max(global_abs_change, abs_change);
            // Move currently_accumulating_scattering value into
            // previous_accumulating_scattering
            // and then zero currently_accumulating_scattering for the next
            // iteration.
            node_data.previous_accumulated_scattering =
                node_data.currently_accumulating_scattering.load(std::memory_order_relaxed);
            node_data.currently_accumulating_scattering.store(0., std::memory_order_relaxed);
          }
          for (auto other_node : downstream_neighbors) {
            // Don't send anything to a ghost node.
            if (other_node >= ghost_threshold)
              continue;
            std::size_t other_magnitude_idx =
              num_per_element * other_node +
              num_per_element_and_direction * dir_idx;
            auto& other_counter =
              radiation_magnitudes[other_magnitude_idx].counter;
            if (!(other_counter.fetch_sub(1, std::memory_order_release) - 1)) {
              work_t new_work_item{other_node, dir_idx};
              ctx.push(new_work_item);
            }
          }
        },
        galois::loopname("Sweep"), galois::no_conflicts(),
        galois::wl<PSchunk>());
    if (print_convergence)
      std::cout << global_abs_change << std::endl;

    if (scattering_outfile != "") {
      if (std::ofstream outfile{scattering_outfile, std::ios_base::app}) {
        for (std::size_t node = 0; node < num_cells; node++) {
          outfile << graph.getData(node, galois::MethodFlag::UNPROTECTED)
                         .previous_accumulated_scattering
                  << std::endl;
        }
      } else {
        GALOIS_DIE("Unable to write to desired output file.");
      }
    }

    global_abs_change = 0;
  }
}
