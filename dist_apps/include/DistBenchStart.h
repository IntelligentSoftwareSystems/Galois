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

#ifndef DIST_BENCH_START_H
#define DIST_BENCH_START_H

#include "galois/Galois.h"
#include "galois/Version.h"
#include "llvm/Support/CommandLine.h"
#include "galois/graphs/DistributedGraphLoader.h"
#include "galois/AtomicHelpers.h"
#ifdef GALOIS_USE_EXP
#include "galois/CompilerHelpers.h"
#endif

#ifdef __GALOIS_HET_CUDA__
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
extern cll::opt<bool> verify;

#ifdef __GALOIS_HET_CUDA__
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

#ifdef __GALOIS_HET_CUDA__
// in internal namespace because this function shouldn't be called elsewhere
namespace internal {
void heteroSetup(std::vector<unsigned>& scaleFactor);
}; // namespace internal

/**
 * Given a loaded graph, marshal it over to the GPU device for use
 * on the GPU.
 *
 * @param loadedGraph a loaded graph that you want to marshal
 * @param cuda_ctx the CUDA context of the currently running program
 */
template <typename NodeData, typename EdgeData>
static void
marshalGPUGraph(galois::graphs::DistGraph<NodeData, EdgeData>* loadedGraph,
                struct CUDA_Context** cuda_ctx) {
  auto& net                 = galois::runtime::getSystemNetworkInterface();
  const unsigned my_host_id = galois::runtime::getHostID();

  galois::StatTimer marshalTimer("TIMER_GRAPH_MARSHAL", "DistBench");

  marshalTimer.start();

  if (personality == GPU_CUDA) {
    *cuda_ctx = get_CUDA_context(my_host_id);

    if (!init_CUDA_context(*cuda_ctx, gpudevice)) {
      GALOIS_DIE("Failed to initialize CUDA context");
    }

    MarshalGraph m;
    (*loadedGraph).getMarshalGraph(m);
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
 * @param cuda_ctx CUDA context of the currently running program; only matters
 * if using GPU
 *
 * @returns Pointer to the loaded graph
 */
template <typename NodeData, typename EdgeData, bool iterateOutEdges = true>
static galois::graphs::DistGraph<NodeData, EdgeData>*
loadDGraph(std::vector<unsigned>& scaleFactor,
           struct CUDA_Context** cuda_ctx = nullptr) {
  galois::StatTimer dGraphTimer("GraphConstructTime", "DistBench");
  dGraphTimer.start();

  galois::graphs::DistGraph<NodeData, EdgeData>* loadedGraph = nullptr;
  loadedGraph =
      galois::graphs::constructGraph<NodeData, EdgeData, iterateOutEdges>(
          scaleFactor);
  assert(loadedGraph != nullptr);

#ifdef __GALOIS_HET_CUDA__
  marshalGPUGraph(loadedGraph, cuda_ctx);
#endif

  dGraphTimer.stop();

  // Save local graph structure
  if (saveLocalGraph)
    (*loadedGraph).save_local_graph_to_file(localGraphFileName);

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
 * @param cuda_ctx CUDA context of the currently running program; only matters
 * if using GPU
 *
 * @returns Pointer to the loaded symmetric graph
 */
template <typename NodeData, typename EdgeData>
static galois::graphs::DistGraph<NodeData, EdgeData>*
loadSymmetricDGraph(std::vector<unsigned>& scaleFactor,
                    struct CUDA_Context** cuda_ctx = nullptr) {
  galois::StatTimer dGraphTimer("GraphConstructTime", "DistBench");
  dGraphTimer.start();

  galois::graphs::DistGraph<NodeData, EdgeData>* loadedGraph = nullptr;

  // make sure that the symmetric graph flag was passed in
  if (inputFileSymmetric) {
    loadedGraph = galois::graphs::constructSymmetricGraph<NodeData, EdgeData>(
        scaleFactor);
  } else {
    GALOIS_DIE("must use -symmetricGraph flag with a symmetric graph for "
               "this benchmark");
  }

  assert(loadedGraph != nullptr);

#ifdef __GALOIS_HET_CUDA__
  marshalGPUGraph(loadedGraph, cuda_ctx);
#endif

  dGraphTimer.stop();

  // Save local graph structure
  if (saveLocalGraph)
    (*loadedGraph).save_local_graph_to_file(localGraphFileName);

  return loadedGraph;
}

/**
 * Loads a 2-way graph into memory.
 * Details/partitioning will be handled in the construct graph call.
 *
 * The user should NOT call this function.
 *
 * @tparam NodeData struct specifying what kind of data the node contains
 * @tparam EdgeData type specifying the type of the edge data
 *
 * @param scaleFactor Vector that specifies how much of the graph each
 * host should get
 * @param cuda_ctx CUDA context of the currently running program; only matters
 * if using GPU
 *
 * @returns Pointer to the loaded 2-way graph
 */
template <typename NodeData, typename EdgeData>
static galois::graphs::DistGraph<NodeData, EdgeData, true>*
loadBDGraph(std::vector<unsigned>& scaleFactor,
            struct CUDA_Context** cuda_ctx = nullptr) {
  galois::StatTimer dGraphTimer("GraphConstructTime", "DistBench");
  dGraphTimer.start();

  galois::graphs::DistGraph<NodeData, EdgeData, true>* loadedGraph = nullptr;

  // make sure that the symmetric graph flag was passed in
  loadedGraph =
      galois::graphs::constructTwoWayGraph<NodeData, EdgeData>(scaleFactor);
  assert(loadedGraph != nullptr);

#ifdef __GALOIS_HET_CUDA__
  // TODO marshal the incoming edges as well....... (for now we don't have
  // support for it)
  marshalGPUGraph(loadedGraph, cuda_ctx);
#endif

  dGraphTimer.stop();

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
 * @returns Pointer to the loaded graph
 */
template <typename NodeData, typename EdgeData, bool iterateOutEdges = true>
galois::graphs::DistGraph<NodeData, EdgeData>*
distGraphInitialization(struct CUDA_Context** cuda_ctx = nullptr) {
  std::vector<unsigned> scaleFactor;
#ifdef __GALOIS_HET_CUDA__
  internal::heteroSetup(scaleFactor);
  return loadDGraph<NodeData, EdgeData, iterateOutEdges>(scaleFactor, cuda_ctx);
#else
  return loadDGraph<NodeData, EdgeData, iterateOutEdges>(scaleFactor);
#endif
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
galois::graphs::DistGraph<NodeData, EdgeData>*
symmetricDistGraphInitialization(struct CUDA_Context** cuda_ctx = nullptr) {
  std::vector<unsigned> scaleFactor;

#ifdef __GALOIS_HET_CUDA__
  internal::heteroSetup(scaleFactor);
  return loadSymmetricDGraph<NodeData, EdgeData>(scaleFactor, cuda_ctx);
#else
  return loadSymmetricDGraph<NodeData, EdgeData>(scaleFactor);
#endif
}

/**
 * TODO
 */
template <typename NodeData, typename EdgeData>
galois::graphs::DistGraph<NodeData, EdgeData, true>*
twoWayDistGraphInitialization(struct CUDA_Context** cuda_ctx = nullptr) {
  std::vector<unsigned> scaleFactor;
#ifdef __GALOIS_HET_CUDA__
  internal::heteroSetup(scaleFactor);
  return loadBDGraph<NodeData, EdgeData>(scaleFactor, cuda_ctx);
#else
  return loadBDGraph<NodeData, EdgeData>(scaleFactor);
#endif
}

#endif
