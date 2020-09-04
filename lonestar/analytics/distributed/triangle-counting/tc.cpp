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

/* This is an implementation of Distributed multi-GPU triangle counting code.
 * The single GPU code which is executed on GPU is generated using the IrGL
 * compiler. Currently, it does not support distributed multi-CPU code.
 *
 * TODO implement CPU kernel
 */

#include "DistBench/MiningStart.h"
#include "galois/DistGalois.h"
#include "galois/DReducible.h"
#include "galois/DTerminationDetector.h"
#include "galois/gstl.h"
#include "galois/graphs/GenericPartitioners.h"
#include "galois/graphs/MiningPartitioner.h"
#include "galois/runtime/Tracer.h"

#include <iostream>
#include <limits>

#ifdef GALOIS_ENABLE_GPU
#include "tc_cuda.h"
struct CUDA_Context* cuda_ctx;
#else
enum { CPU, GPU_CUDA };
int personality = CPU;
#endif

namespace cll = llvm::cl;

constexpr static const char* const REGION_NAME = "TC";

/*******************************************************************************
 * Graph structure declarations + other initialization
 ******************************************************************************/

typedef galois::graphs::MiningGraph<void, void, MiningPolicyDegrees> Graph;
typedef typename Graph::GraphNode GNode;

std::unique_ptr<galois::graphs::GluonEdgeSubstrate<Graph>> syncSubstrate;

template <bool async>
struct TC {
  Graph* graph;
  using DGAccumulatorTy = galois::DGAccumulator<uint64_t>;
  DGAccumulatorTy& numTriangles;

  TC(Graph* _graph, DGAccumulatorTy& _numTriangles)
      : graph(_graph), numTriangles(_numTriangles) {}

  // use the below line once CPU code is added
  void static go(Graph& _graph) {
    unsigned _num_iterations = 0;
    DGAccumulatorTy numTriangles;
    syncSubstrate->set_num_round(_num_iterations);
    numTriangles.reset();
    const auto& allMasterNodes = _graph.masterNodesRange();

#ifdef GALOIS_ENABLE_GPU
    if (personality == GPU_CUDA) { ///< GPU TC.
      std::string impl_str(syncSubstrate->get_run_identifier("TC"));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      uint64_t num_local_triangles = 0;
      TC_masterNodes_cuda(num_local_triangles, cuda_ctx);
      numTriangles += num_local_triangles;
      StatTimer_cuda.stop();
    } else { ///< CPU TC.
#endif
      galois::do_all(
          galois::iterate(allMasterNodes), TC(&_graph, numTriangles),
          galois::steal(),
          galois::loopname(syncSubstrate->get_run_identifier("TC").c_str()));
#ifdef GALOIS_ENABLE_GPU
    }
#endif

    uint64_t total_triangles = numTriangles.reduce();
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("Total number of triangles ", total_triangles, "\n");
    }
  }

  void operator()(GNode v) const {
    size_t numTriangles_local = 0;
    for (auto vIter : graph->edges(v)) {
      GNode w                       = graph->getEdgeDst(vIter);
      Graph::edge_iterator vIterBeg = graph->edge_begin(v);
      Graph::edge_iterator vIterEnd = graph->edge_end(v);

      for (auto wIter : graph->edges(w)) {
        auto x                      = graph->getEdgeDst(wIter);
        Graph::edge_iterator vvIter = vIterBeg;
        while (graph->getEdgeDst(vvIter) < x && vvIter < vIterEnd) {
          vvIter++;
        }
        if (vvIter < vIterEnd && x == graph->getEdgeDst(vvIter)) {
          ++numTriangles_local;
        }
      }
    } ///< Finding triangles is done.
    numTriangles += numTriangles_local;
  } ///< CPU operator is done.
};

/*******************************************************************************
 * Main
 ******************************************************************************/

constexpr static const char* const name =
    "TC - Distributed Multi-GPU Triangle Counting ";
constexpr static const char* const desc = "TC on Distributed GPU (D-IrGL).";
constexpr static const char* const url  = nullptr;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  if (!symmetricGraph) {
    GALOIS_DIE("This application requires a symmetric graph input;"
               " please use the -symmetricGraph flag "
               " to indicate the input is a symmetric graph.");
  }

  const auto& net = galois::runtime::getSystemNetworkInterface();

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);

  StatTimer_total.start();
  std::unique_ptr<Graph> hg;
#ifdef GALOIS_ENABLE_GPU
  std::tie(hg, syncSubstrate) =
      distGraphInitialization<void, void>(&cuda_ctx, false);
#else
  std::tie(hg, syncSubstrate) = distGraphInitialization<void, void>(false);
#endif

  if (personality == GPU_CUDA) {
#ifdef GALOIS_ENABLE_GPU
    std::string timer_str("SortEdgesGPU");
    galois::StatTimer edgeSortTime("SortEdgesGPU", REGION_NAME);
    edgeSortTime.start();
    sortEdgesByDestination_cuda(cuda_ctx);
    edgeSortTime.stop();
#else
    abort();
#endif
  } else if (personality == CPU) {
    galois::StatTimer edgeSortTime("SortEdgesCPU", REGION_NAME);
    edgeSortTime.start();
    hg->sortEdgesByDestination();
    edgeSortTime.stop();
  }
  ///! accumulators for use in operators
  galois::DGAccumulator<uint64_t> DGAccumulator_numTriangles;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] TC::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
    TC<false>::go(*hg);
    StatTimer_main.stop();

    syncSubstrate->set_num_run(run + 1);
  }
  StatTimer_total.stop();

  if (output) {
    galois::gError("output requested but this application doesn't support it");
    return 1;
  }

  return 0;
}
