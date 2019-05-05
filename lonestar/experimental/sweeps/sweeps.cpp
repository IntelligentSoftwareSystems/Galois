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

#include <array>
#include <iostream>

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
static llvm::cl::opt<std::size_t> num_directions{"num_directions", llvm::cl::desc("number of directions"), llvm::cl::init(32u)};
static llvm::cl::opt<std::size_t> maxiters{"maxiters", llvm::cl::desc("maximum number of iterations"), llvm::cl::init(100u)};

// TODO: We need a graph type with dynamically sized node/edge data for this problem.
// For now, indexing into a separate data structure will have to be sufficient.

// TODO: This is another example of an undirected graph being important.
// We need one of those as well.

// Note: I'm going to use a CSR graph, so each node will already have a
// unique integer id that can be used to index other data structures, so
// it's not necessary to store anything other than the actual topology
// on the graph nodes. Since the rest of the data is dynamically sized,
// it'll be in a separate array.
// Each edge does need to store the unit normal going outward through
// the corresponding face though.

// TODO: Try making edge data separate and shared to see if it helps
// locality. It's not clear that it will, but it's something to try.

using graph_t = galois::graphs::LC_CSR_Graph<void, std::array<double, 3>>;

// Routine to initialize graph topology and face normals.
// Note: C++17 will allow just returning the graph since copy elision will be guaranteed.
// For now, the reference is needed though, since the built graph may not be copyable.
void generate_grid(graph_t &built_graph, std::size_t nx, std::size_t ny, std::size_t nz) noexcept {

  galois::graphs::FileGraphWriter temp_graph;
  std::size_t num_nodes = nx * ny * nz;
  std::size_t num_edges = 2 * (nx * ny * (nz - 1) + nx * (ny - 1) * nz + (nx - 1) * ny * nz);
  temp_graph.setNumNodes(num_nodes);
  temp_graph.setNumEdges(num_edges);
  temp_graph.setSizeofEdgeData(galois::LargeArray<graph_t::edge_data_type>::size_of::value);
  galois::LargeArray<graph_t::edge_data_type> edge_data;
  edge_data.create(num_edges);

  temp_graph.phase1();
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * nx * ny + j * ny + k;
        if (i > 0) temp_graph.incrementDegree(id);
        if (i < nx - 1) temp_graph.incrementDegree(id);
        if (j > 0) temp_graph.incrementDegree(id);
        if (j < ny - 1) temp_graph.incrementDegree(id);
        if (k > 0) temp_graph.incrementDegree(id);
        if (k < nz - 1) temp_graph.incrementDegree(id);
      }
    }
  }

  temp_graph.phase2();
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * nx * ny + j * ny + k;
        if (i > 0) edge_data.set(temp_graph.addNeighbor(id, id - ny * nz), {1., 0., 0.});
        if (i < nx - 1) edge_data.set(temp_graph.addNeighbor(id, id + ny * nz), {-1., 0., 0.});
        if (j > 0) edge_data.set(temp_graph.addNeighbor(id, id - nz), {0., 1., 0.});
        if (j < ny - 1) edge_data.set(temp_graph.addNeighbor(id, id + nz), {0., -1., 0.});
        if (k > 0) edge_data.set(temp_graph.addNeighbor(id, id - 1), {0., 0., 1.});
        if (k < nz - 1) edge_data.set(temp_graph.addNeighbor(id, id + 1), {0., 0., -1.});
      }
    }
  }

  // TODO: is it possible to set the edge data during construction?
  auto *rawEdgeData = temp_graph.finish<graph_t::edge_data_type>();
  std::uninitialized_copy(std::make_move_iterator(edge_data.begin()), std::make_move_iterator(edge_data.end()), rawEdgeData);

  galois::graphs::readGraph(built_graph, temp_graph);
}

int main(int argc, char**argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url);

  graph_t graph;
  generate_grid(graph, nx, ny, nz);
}

