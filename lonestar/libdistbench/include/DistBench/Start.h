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

#ifndef GALOIS_DISTBENCH_START_H
#define GALOIS_DISTBENCH_START_H

#include "DistBench/Input.h"
#include "galois/AtomicHelpers.h"
#include "galois/Galois.h"
#include "galois/graphs/GluonSubstrate.h"
#include "galois/Version.h"
#include "llvm/Support/CommandLine.h"

#ifdef GALOIS_ENABLE_GPU
#include "galois/cuda/HostDecls.h"
#else
// dummy struct declaration to allow non-het code to compile without
// having to include cuda_context_decl
struct CUDA_Context;
#endif

//! standard global options to the benchmarks
namespace cll = llvm::cl;

extern cll::opt<int> numThreads;
extern cll::opt<int> numRuns;
extern cll::opt<std::string> statFile;
//! If set, ignore partitioning comm optimizations
extern cll::opt<bool> partitionAgnostic;
//! Set method for metadata sends
extern cll::opt<DataCommMode> commMetadata;
//! Where to write output if output is set
extern cll::opt<std::string> outputLocation;
extern cll::opt<bool> output;

#ifdef GALOIS_ENABLE_GPU
enum Personality { CPU, GPU_CUDA };

std::string personality_str(Personality p);

extern int gpudevice;
extern Personality personality;
extern cll::opt<unsigned> scalegpu;
extern cll::opt<unsigned> scalecpu;
extern cll::opt<int> num_nodes;
extern cll::opt<std::string> personality_set;
#endif

/**
 * Initialize Galois runtime for distributed benchmarks and print/report various
 * information.
 *
 * @param argc argument count
 * @param argv list of arguments
 * @param app Name of the application
 * @param desc Description of the application
 * @param url URL to the application
 */
void DistBenchStart(int argc, char** argv, const char* app,
                    const char* desc = nullptr, const char* url = nullptr);

template <typename NodeData, typename EdgeData>
using DistGraphPtr =
    std::unique_ptr<galois::graphs::DistGraph<NodeData, EdgeData>>;
template <typename NodeData, typename EdgeData>
using DistSubstratePtr = std::unique_ptr<galois::graphs::GluonSubstrate<
    galois::graphs::DistGraph<NodeData, EdgeData>>>;

#ifdef GALOIS_ENABLE_GPU
// in internal namespace because this function shouldn't be called elsewhere
namespace internal {
void heteroSetup(std::vector<unsigned>& scaleFactor);
}; // namespace internal

/**
 * Given a loaded graph, marshal it over to the GPU device for use
 * on the GPU.
 *
 * @param gluonSubstrate Gluon substrate containing info needed to marshal
 * to GPU
 * @param cuda_ctx the CUDA context of the currently running program
 */
template <typename NodeData, typename EdgeData>
static void
marshalGPUGraph(DistSubstratePtr<NodeData, EdgeData>& gluonSubstrate,
                struct CUDA_Context** cuda_ctx) {
  auto& net                 = galois::runtime::getSystemNetworkInterface();
  const unsigned my_host_id = galois::runtime::getHostID();

  galois::StatTimer marshalTimer("TIMER_GRAPH_MARSHAL", "DistBench");

  marshalTimer.start();

  if (personality == GPU_CUDA) {
    *cuda_ctx = get_CUDA_context(my_host_id);

    if (!init_CUDA_context(*cuda_ctx, gpudevice)) {
      GALOIS_DIE("failed to initialize CUDA context");
    }

    MarshalGraph m;
    (*gluonSubstrate).getMarshalGraph(m);
    load_graph_CUDA(*cuda_ctx, m, net.Num);
  }

  marshalTimer.stop();
}
#endif

/**
 * Loads a graph into memory. Details/partitioning will be handled in the
 * construct graph call.
 *
 * The user should NOT call this function.
 *
 * @tparam NodeData struct specifying what kind of data the node contains
 * @tparam EdgeData type specifying the type of the edge data
 * @tparam iterateOutEdges Boolean specifying if the graph should be iterating
 * over outgoing or incoming edges
 *
 * @param scaleFactor Vector that specifies how much of the graph each
 * host should get
 *
 * @returns Pointer to the loaded graph
 */
template <typename NodeData, typename EdgeData, bool iterateOutEdges = true>
static DistGraphPtr<NodeData, EdgeData>
loadDistGraph(std::vector<unsigned>& scaleFactor) {
  galois::StatTimer dGraphTimer("GraphConstructTime", "DistBench");
  dGraphTimer.start();

  DistGraphPtr<NodeData, EdgeData> loadedGraph =
      constructGraph<NodeData, EdgeData, iterateOutEdges>(scaleFactor);
  assert(loadedGraph != nullptr);

  dGraphTimer.stop();

  // Save local graph structure
  // if (saveLocalGraph)
  //  (*loadedGraph).save_local_graph_to_file(localGraphFileName);

  return loadedGraph;
}

/**
 * Loads a symmetric graph into memory.
 * Details/partitioning will be handled in the construct graph call.
 *
 * The user should NOT call this function.
 *
 * @tparam NodeData struct specifying what kind of data the node contains
 * @tparam EdgeData type specifying the type of the edge data
 *
 * @param scaleFactor Vector that specifies how much of the graph each
 * host should get
 *
 * @returns Pointer to the loaded symmetric graph
 */
template <typename NodeData, typename EdgeData>
static DistGraphPtr<NodeData, EdgeData>
loadSymmetricDistGraph(std::vector<unsigned>& scaleFactor) {
  galois::StatTimer dGraphTimer("GraphConstructTime", "DistBench");
  dGraphTimer.start();

  DistGraphPtr<NodeData, EdgeData> loadedGraph = nullptr;

  // make sure that the symmetric graph flag was passed in
  if (inputFileSymmetric) {
    loadedGraph = constructSymmetricGraph<NodeData, EdgeData>(scaleFactor);
  } else {
    GALOIS_DIE("This application requires a symmetric graph input;"
               " please use the -symmetricGraph flag "
               " to indicate the input is a symmetric graph.");
  }

  assert(loadedGraph != nullptr);

  dGraphTimer.stop();

  // Save local graph structure
  // if (saveLocalGraph)
  //  (*loadedGraph).save_local_graph_to_file(localGraphFileName);

  return loadedGraph;
}

/**
 * Loads a graph into memory, setting up heterogeneous execution if
 * necessary. Unlike the dGraph load functions above, this is meant
 * to be exposed to the user.
 *
 * @tparam NodeData struct specifying what kind of data the node contains
 * @tparam EdgeData type specifying the type of the edge data
 * @tparam iterateOutEdges Boolean specifying if the graph should be iterating
 * over outgoing or incoming edges
 *
 * @param cuda_ctx CUDA context of the currently running program; only matters
 * if using GPU
 *
 * @returns Pointer to the loaded graph and Gluon substrate
 */
template <typename NodeData, typename EdgeData, bool iterateOutEdges = true>
std::pair<DistGraphPtr<NodeData, EdgeData>,
          DistSubstratePtr<NodeData, EdgeData>>
#ifdef GALOIS_ENABLE_GPU
distGraphInitialization(struct CUDA_Context** cuda_ctx) {
#else
distGraphInitialization() {
#endif
  using Graph     = galois::graphs::DistGraph<NodeData, EdgeData>;
  using Substrate = galois::graphs::GluonSubstrate<Graph>;
  std::vector<unsigned> scaleFactor;
  DistGraphPtr<NodeData, EdgeData> g;
  DistSubstratePtr<NodeData, EdgeData> s;

#ifdef GALOIS_ENABLE_GPU
  internal::heteroSetup(scaleFactor);
#endif
  g = loadDistGraph<NodeData, EdgeData, iterateOutEdges>(scaleFactor);
  // load substrate
  const auto& net = galois::runtime::getSystemNetworkInterface();
  s = std::make_unique<Substrate>(*g, net.ID, net.Num, g->isTransposed(),
                                  g->cartesianGrid(), partitionAgnostic,
                                  commMetadata);

// marshal graph to GPU as necessary
#ifdef GALOIS_ENABLE_GPU
  marshalGPUGraph(s, cuda_ctx);
#endif

  return std::make_pair(std::move(g), std::move(s));
}

/**
 * Loads a symmetric graph into memory, setting up heterogeneous execution if
 * necessary. Unlike the dGraph load functions above, this is meant
 * to be exposed to the user.
 *
 * @tparam NodeData struct specifying what kind of data the node contains
 * @tparam EdgeData type specifying the type of the edge data
 *
 * @param cuda_ctx CUDA context of the currently running program; only matters
 * if using GPU
 *
 * @returns Pointer to the loaded symmetric graph
 */
template <typename NodeData, typename EdgeData>
std::pair<DistGraphPtr<NodeData, EdgeData>,
          DistSubstratePtr<NodeData, EdgeData>>
#ifdef GALOIS_ENABLE_GPU
symmetricDistGraphInitialization(struct CUDA_Context** cuda_ctx) {
#else
symmetricDistGraphInitialization() {
#endif
  using Graph     = galois::graphs::DistGraph<NodeData, EdgeData>;
  using Substrate = galois::graphs::GluonSubstrate<Graph>;
  std::vector<unsigned> scaleFactor;
  DistGraphPtr<NodeData, EdgeData> g;
  DistSubstratePtr<NodeData, EdgeData> s;

#ifdef GALOIS_ENABLE_GPU
  internal::heteroSetup(scaleFactor);
#endif
  g = loadSymmetricDistGraph<NodeData, EdgeData>(scaleFactor);
  // load substrate
  const auto& net = galois::runtime::getSystemNetworkInterface();
  s = std::make_unique<Substrate>(*g, net.ID, net.Num, g->isTransposed(),
                                  g->cartesianGrid(), partitionAgnostic,
                                  commMetadata);

// marshal graph to GPU as necessary
#ifdef GALOIS_ENABLE_GPU
  marshalGPUGraph(s, cuda_ctx);
#endif

  return std::make_pair(std::move(g), std::move(s));
}

#endif
