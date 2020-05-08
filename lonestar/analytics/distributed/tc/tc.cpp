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

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "galois/graphs/MiningPartitioner.h"
#include "galois/graphs/GenericPartitioners.h"
#include "DistMiningBenchStart.h"
#include "galois/DReducible.h"
#include "galois/DTerminationDetector.h"
#include "galois/runtime/Tracer.h"

#ifdef GALOIS_ENABLE_GPU
#include "tc_cuda.h"
struct CUDA_Context* cuda_ctx;
#endif

constexpr static const char* const regionname = "TC";

namespace cll = llvm::cl;

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

struct NodeData {
  char dummy;
};

typedef galois::graphs::MiningGraph<NodeData, void, MiningPolicyDegrees> Graph;
typedef typename Graph::GraphNode GNode;

galois::graphs::GluonEdgeSubstrate<Graph>* syncSubstrate;

template <bool async>
struct TC {
  Graph* graph;
  using DGAccumulatorTy = galois::DGAccumulator<uint64_t>;
  DGAccumulatorTy& num_triangles;

  TC(Graph* _graph, DGAccumulatorTy& _num_triangles)
      : graph(_graph), num_triangles(_num_triangles) {}

  // use the below line once CPU code is added
  //void static go(Graph& _graph) {
  void static go() {
    unsigned _num_iterations = 0;
    DGAccumulatorTy num_triangles;
    syncSubstrate->set_num_round(_num_iterations);
    num_triangles.reset();

#ifdef GALOIS_ENABLE_GPU
    if (personality == GPU_CUDA) {
      std::string impl_str(syncSubstrate->get_run_identifier("TC"));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), regionname);
      StatTimer_cuda.start();
      uint64_t num_local_triangles = 0;
      TC_masterNodes_cuda(num_local_triangles, cuda_ctx);
      num_triangles += num_local_triangles;
      StatTimer_cuda.stop();
    }
#endif

    uint64_t total_triangles = num_triangles.reduce();
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("Total number of triangles ", total_triangles, "\n");
    }
  }
};

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name =
    "TC - Distributed Multi-GPU Triangle Counting ";
constexpr static const char* const desc = "TC on Distributed GPU (D-IrGL).";
constexpr static const char* const url  = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  const auto& net = galois::runtime::getSystemNetworkInterface();

  galois::StatTimer StatTimer_total("TimerTotal", regionname);

  StatTimer_total.start();
  Graph* hg;
#ifdef GALOIS_ENABLE_GPU
  std::tie(hg, syncSubstrate) =
      distGraphInitialization<NodeData, void>(&cuda_ctx, false);
#else
  std::tie(hg, syncSubstrate) = distGraphInitialization<NodeData, void>();
#endif

  std::string timer_str("SortEdgesGPU");
  galois::StatTimer edgeSortTime("SortEdgesGPU", regionname);
  edgeSortTime.start();
  sort_cuda(cuda_ctx);
  edgeSortTime.stop();

  // accumulators for use in operators
  galois::DGAccumulator<uint64_t> DGAccumulator_num_triangles;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] TC::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), regionname);

    StatTimer_main.start();
    // use the below line once CPU code is added
    //TC<false>::go(*hg);
    TC<false>::go();
    StatTimer_main.stop();

    syncSubstrate->set_num_run(run + 1);
  }
  StatTimer_total.stop();

  return 0;
}
