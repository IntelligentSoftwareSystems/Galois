/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

// TODO document + merge with regular bench start? have to figure out how to
// have both substrates coexist

#ifndef _DIST_MINE_BENCH_START_H_
#define _DIST_MINE_BENCH_START_H_

#include "galois/Galois.h"
#include "galois/Version.h"
#include "llvm/Support/CommandLine.h"
#include "galois/graphs/MiningPartitioner.h"
#include "galois/graphs/GenericPartitioners.h"
#include "galois/graphs/GluonEdgeSubstrate.h"
#include "DistributedGraphLoader.h"
#include "galois/AtomicHelpers.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/cuda/EdgeHostDecls.h"
#else
// dummy struct declaration to allow non-het code to compile without
// having to include cuda_context_decl
struct CUDA_Context;
#endif

//! standard global options to the benchmarks
namespace cll = llvm::cl;

// note these come from distbenchstart rather than mining bench
extern cll::opt<int> numThreads;
extern cll::opt<int> numRuns;
extern cll::opt<std::string> statFile;
extern cll::opt<bool> verify;
//! Set method for metadata sends
extern cll::opt<DataCommMode> commMetadata;

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
 * @param GluonEdgeSubstrate Gluon substrate containing info needed to marshal
 * to GPU
 * @param cuda_ctx the CUDA context of the currently running program
 */
template <typename NodeData, typename EdgeData>
static void
marshalGPUGraph(galois::graphs::GluonEdgeSubstrate<
                  galois::graphs::MiningGraph<NodeData, EdgeData, MiningPolicyDegrees>
                >* GluonEdgeSubstrate,
                struct CUDA_Context** cuda_ctx, bool LoadProxyEdges = true) {
  auto& net                 = galois::runtime::getSystemNetworkInterface();
  const unsigned my_host_id = galois::runtime::getHostID();

  galois::StatTimer marshalTimer("TIMER_GRAPH_MARSHAL", "DistBench");

  marshalTimer.start();

  if (personality == GPU_CUDA) {
    *cuda_ctx = get_CUDA_context(my_host_id);

    if (!init_CUDA_context(*cuda_ctx, gpudevice)) {
      GALOIS_DIE("Failed to initialize CUDA context");
    }

    EdgeMarshalGraph m;
    (*GluonEdgeSubstrate).getEdgeMarshalGraph(m, LoadProxyEdges);
    load_graph_CUDA(*cuda_ctx, m, net.Num);
  }
  marshalTimer.stop();
}
#endif

/**
 */
template <typename NodeData, typename EdgeData, bool iterateOutEdges = true>
static galois::graphs::MiningGraph<NodeData, EdgeData, MiningPolicyDegrees>*
loadDGraph(std::vector<unsigned>& scaleFactor, bool loadProxyEdges,
           struct CUDA_Context** cuda_ctx = nullptr) {
  using Graph = galois::graphs::MiningGraph<NodeData, EdgeData, MiningPolicyDegrees>;
  galois::StatTimer dGraphTimer("GraphConstructTime", "DistBench");

  dGraphTimer.start();
  const auto& net = galois::runtime::getSystemNetworkInterface();
  auto* loadedGraph = new Graph(inputFile, net.ID, net.Num, loadProxyEdges, loadProxyEdges);
  assert(loadedGraph != nullptr);
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
 * @returns Pointer to the loaded graph and Gluon substrate
 */
template <typename NodeData, typename EdgeData, bool iterateOutEdges = true>
std::pair<galois::graphs::MiningGraph<NodeData, EdgeData, MiningPolicyDegrees>*,
          galois::graphs::GluonEdgeSubstrate<
            galois::graphs::MiningGraph<NodeData, EdgeData, MiningPolicyDegrees>
          >*>
distGraphInitialization(struct CUDA_Context** cuda_ctx = nullptr, bool loadProxyEdges = true) {
  galois::StatTimer initTimer("DistGraphInitialization", "DistMiningBench");
  using Graph = galois::graphs::MiningGraph<NodeData, EdgeData, MiningPolicyDegrees>;
  using Substrate = galois::graphs::GluonEdgeSubstrate<Graph>;

  initTimer.start();
  std::vector<unsigned> scaleFactor;
  Graph* g;
  Substrate* s;

  #ifdef __GALOIS_HET_CUDA__
  internal::heteroSetup(scaleFactor);
  g = loadDGraph<NodeData, EdgeData, iterateOutEdges>(scaleFactor, loadProxyEdges,
                                                      cuda_ctx);
  #else
  g = loadDGraph<NodeData, EdgeData, iterateOutEdges>(scaleFactor, loadProxyEdges);
  #endif

  // load substrate
  const auto& net = galois::runtime::getSystemNetworkInterface();
  // if you want to load proxy edges (true), then do nothing should be false
  // hence the use of ! to negate
  s = new Substrate(*g, net.ID, net.Num, !loadProxyEdges, commMetadata);

  // marshal graph to GPU as necessary
  #ifdef __GALOIS_HET_CUDA__
  if (net.ID == 0) {
    galois::gPrint("Beginning to marshal graph to GPU\n");
  }
  marshalGPUGraph(s, cuda_ctx, loadProxyEdges);
  #endif

  initTimer.stop();

  return std::make_pair(g, s);
}


#endif
