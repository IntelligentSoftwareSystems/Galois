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

#include <cstddef>
#include <iostream>

// Vendored from an old version of LLVM for Lonestar app command line handling.
#include "llvm/Support/CommandLine.h"

#include <galois/Galois.h>
#include <galois/graphs/Graph.h>
#include <galois/graphs/FileGraph.h>
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

int main(int argc, char**argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url);
}

