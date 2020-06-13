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

/**
 * This version of BC-Level uses an option in the synchronization runtime to
 * avoid the overheads of having 2 extra accumulator variables.
 */

//#define BCDEBUG

#include "DistBench/Output.h"
#include "DistBench/Start.h"
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

#include <iomanip>
#include <iostream>
#include <limits>

#ifdef GALOIS_ENABLE_GPU
#include "bc_level_cuda.h"
struct CUDA_Context* cuda_ctx;
#else
// type of the num shortest paths variable
using ShortPathType = double;
enum { CPU, GPU_CUDA };
int personality = CPU;
#endif

constexpr static const char* const REGION_NAME = "BC";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;
static cll::opt<std::string>
    sourcesToUse("sourcesToUse",
                 cll::desc("Whitespace separated list "
                           "of sources in a file to "
                           "use in BC (default empty)"),
                 cll::init(""));
static cll::opt<bool>
    singleSourceBC("singleSource",
                   cll::desc("Use for single source BC (default off)"),
                   cll::init(false));
static cll::opt<uint64_t>
    startSource("startNode",
                cll::desc("Starting source node used for "
                          "betweeness-centrality (default 0)"),
                cll::init(0));
static cll::opt<unsigned int>
    numberOfSources("numOfSources",
                    cll::desc("Number of sources to use for "
                              "betweeness-centraility (default all)"),
                    cll::init(0));

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/
const uint32_t infinity          = std::numeric_limits<uint32_t>::max() / 4;
static uint64_t current_src_node = 0;
// global round numbers; 1 for forward, 1 for back; used in sync structs as well
uint32_t globalRoundNumber = 0;
uint32_t backRoundCount    = 0;

// NOTE: types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  // SSSP vars
  std::atomic<uint32_t> current_length;
  // Betweeness centrality vars
  std::atomic<ShortPathType> num_shortest_paths;
  float dependency;
  float betweeness_centrality;

  //#ifdef BCDEBUG
  void dump() {
    galois::gPrint("DUMP: ", current_length.load(), " ",
                   num_shortest_paths.load(), " ", dependency, "\n");
  }
  //#endif
};

// reading in list of sources to operate on if provided
std::ifstream sourceFile;
std::vector<uint64_t> sourceVector;

using Graph = galois::graphs::DistGraph<NodeData, void>;
using GNode = typename Graph::GraphNode;

// bitsets for tracking updates
galois::DynamicBitSet bitset_num_shortest_paths;
galois::DynamicBitSet bitset_current_length;
galois::DynamicBitSet bitset_dependency;

std::unique_ptr<galois::graphs::GluonSubstrate<Graph>> syncSubstrate;

// sync structures
#include "bc_level_sync.hh"

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/

struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}

  /* Initialize the graph */
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    if (personality == GPU_CUDA) {
#ifdef GALOIS_ENABLE_GPU
      std::string impl_str(
          syncSubstrate->get_run_identifier("InitializeGraph"));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeGraph_allNodes_cuda(cuda_ctx);
      StatTimer_cuda.stop();
#else
      abort();
#endif
    } else if (personality == CPU) {
      galois::do_all(
          // pass in begin/end to not use local thread ranges
          galois::iterate(allNodes.begin(), allNodes.end()),
          InitializeGraph{&_graph}, galois::no_stats(),
          galois::loopname("InitializeGraph"));
    }
  }

  /* Functor passed into the Galois operator to carry out initialization;
   * reset everything */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.betweeness_centrality = 0;
    src_data.num_shortest_paths    = 0;
    src_data.dependency            = 0;
  }
};

/* This is used to reset node data when switching to a difference source */
struct InitializeIteration {
  const uint32_t& local_infinity;
  const uint64_t& local_current_src_node;
  Graph* graph;

  InitializeIteration(const uint32_t& _local_infinity,
                      const uint64_t& _local_current_src_node, Graph* _graph)
      : local_infinity(_local_infinity),
        local_current_src_node(_local_current_src_node), graph(_graph) {}

  /* Reset necessary graph metadata for next iteration of SSSP */
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    if (personality == GPU_CUDA) {
#ifdef GALOIS_ENABLE_GPU
      std::string impl_str(
          syncSubstrate->get_run_identifier("InitializeIteration"));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeIteration_allNodes_cuda(infinity, current_src_node, cuda_ctx);
      StatTimer_cuda.stop();
#else
      abort();
#endif
    } else if (personality == CPU) {
      galois::do_all(
          galois::iterate(allNodes.begin(), allNodes.end()),
          InitializeIteration{infinity, current_src_node, &_graph},
          galois::loopname(syncSubstrate
                               ->get_run_identifier(std::string(REGION_NAME) +
                                                    "_InitializeIteration")
                               .c_str()),
          galois::no_stats());
    }
  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    bool is_source = graph->getGID(src) == local_current_src_node;

    if (!is_source) {
      src_data.current_length     = local_infinity;
      src_data.num_shortest_paths = 0;
    } else {
      src_data.current_length     = 0;
      src_data.num_shortest_paths = 1;
    }
    src_data.dependency = 0;
  }
};

/**
 * Forward pass does level by level BFS to find distances and number of
 * shortest paths
 */
struct ForwardPass {
  Graph* graph;
  galois::DGAccumulator<uint32_t>& dga;
  uint32_t local_r;

  ForwardPass(Graph* _graph, galois::DGAccumulator<uint32_t>& _dga,
              uint32_t roundNum)
      : graph(_graph), dga(_dga), local_r(roundNum) {}

  /**
   * Level by level BFS while also finding number of shortest paths to a
   * particular node in the BFS tree.
   *
   * @param _graph Graph to use
   * @param _dga distributed accumulator
   * @param[out] roundNumber Number of rounds taken to finish
   */
  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& _dga) {
    globalRoundNumber          = 0;
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    bool moreThanOne = galois::runtime::getSystemNetworkInterface().Num > 1;

    do {
      _dga.reset();

      if (personality == GPU_CUDA) {
#ifdef GALOIS_ENABLE_GPU
        std::string impl_str(syncSubstrate->get_run_identifier(
            std::string(REGION_NAME) + "_ForwardPass"));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        unsigned int __retval = 0;
        ForwardPass_nodesWithEdges_cuda(__retval, globalRoundNumber, cuda_ctx);
        _dga += __retval;
        StatTimer_cuda.stop();
#else
        abort();
#endif
      } else if (personality == CPU) {
        galois::do_all(
            galois::iterate(nodesWithEdges),
            ForwardPass(&_graph, _dga, globalRoundNumber),
            galois::loopname(syncSubstrate
                                 ->get_run_identifier(std::string(REGION_NAME) +
                                                      "_ForwardPass")
                                 .c_str()),
            galois::steal(), galois::no_stats());
      }

      // synchronize distances and shortest paths
      // read any because a destination node without the correct distance
      // may use a different distance (leading to incorrectness)
      if (moreThanOne) {
        syncSubstrate->sync<writeDestination, readAny,
                            Reduce_min_current_length, Bitset_current_length>(
            std::string(REGION_NAME) + "_ForwardPass");
        syncSubstrate
            ->sync<writeDestination, readSource, Reduce_add_num_shortest_paths,
                   Bitset_num_shortest_paths>(std::string(REGION_NAME) +
                                              "_ForwardPass");
      }

      globalRoundNumber++;
    } while (_dga.reduce(syncSubstrate->get_run_identifier()));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length == local_r) {
      for (auto current_edge : graph->edges(src)) {
        GNode dst         = graph->getEdgeDst(current_edge);
        auto& dst_data    = graph->getData(dst);
        uint32_t new_dist = 1 + src_data.current_length;
        uint32_t old = galois::atomicMin(dst_data.current_length, new_dist);

        if (old > new_dist) {
          // assert(dst_data.current_length == r + 1);
          // assert(src_data.num_shortest_paths > 0);

          bitset_current_length.set(dst);
          double nsp = src_data.num_shortest_paths;
          galois::atomicAdd(dst_data.num_shortest_paths, nsp);
          bitset_num_shortest_paths.set(dst);

          dga += 1;
        } else if (old == new_dist) {
          // assert(src_data.num_shortest_paths > 0);
          // assert(dst_data.current_length == r + 1);

          double nsp = src_data.num_shortest_paths;
          galois::atomicAdd(dst_data.num_shortest_paths, nsp);
          bitset_num_shortest_paths.set(dst);

          dga += 1;
        }
      }
    }
  }
};

/**
 * Synchronize num shortest paths on destinations (should already
 * exist on all sources).
 */
struct MiddleSync {
  Graph* graph;
  const uint32_t local_infinity;

  MiddleSync(Graph* _graph, const uint32_t li)
      : graph(_graph), local_infinity(li){};

  void static go(Graph& _graph, const uint32_t _li) {
    // step only required if more than one host
    if (galois::runtime::getSystemNetworkInterface().Num > 1) {
      const auto& masters = _graph.masterNodesRange();

      if (personality == GPU_CUDA) {
#ifdef GALOIS_ENABLE_GPU
        std::string impl_str(syncSubstrate->get_run_identifier("MiddleSync"));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        MiddleSync_masterNodes_cuda(infinity, cuda_ctx);
        StatTimer_cuda.stop();
#else
        abort();
#endif
      } else if (personality == CPU) {
        galois::do_all(
            galois::iterate(masters.begin(), masters.end()),
            MiddleSync(&_graph, _li),
            galois::loopname(
                syncSubstrate->get_run_identifier("MiddleSync").c_str()),
            galois::no_stats());
      }

      syncSubstrate->sync<writeSource, readAny, Reduce_set_num_shortest_paths>(
          std::string(REGION_NAME) + "_MiddleSync");
    }
  }

  /**
   * Set node for sync if it has a non-zero distance
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      bitset_num_shortest_paths.set(src);
    }
  }
};

/**
 * Propagate dependency backward by iterating backward over levels of BFS tree
 */
struct BackwardPass {
  Graph* graph;
  uint32_t local_r;

  BackwardPass(Graph* _graph, uint32_t roundNum)
      : graph(_graph), local_r(roundNum) {}

  void static go(Graph& _graph, uint32_t roundNumber) {
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    bool moreThanOne = galois::runtime::getSystemNetworkInterface().Num > 1;

    backRoundCount = roundNumber - 1;

    for (; backRoundCount > 0; backRoundCount--) {
      if (personality == GPU_CUDA) {
#ifdef GALOIS_ENABLE_GPU
        std::string impl_str(syncSubstrate->get_run_identifier("BackwardPass"));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        BackwardPass_nodesWithEdges_cuda(backRoundCount, cuda_ctx);
        StatTimer_cuda.stop();
#else
        abort();
#endif
      } else if (personality == CPU) {
        galois::do_all(
            galois::iterate(nodesWithEdges),
            BackwardPass(&_graph, backRoundCount),
            galois::loopname(syncSubstrate
                                 ->get_run_identifier(std::string(REGION_NAME) +
                                                      "_BackwardPass")
                                 .c_str()),
            galois::steal(), galois::no_stats());
      }

      if (moreThanOne) {
        syncSubstrate->sync<writeSource, readDestination, Reduce_add_dependency,
                            Bitset_dependency>(std::string(REGION_NAME) +
                                               "_BackwardPass");
      }
    }
  }

  /**
   * If on the correct level, calculate self-depndency by checking successor
   * nodes.
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length == local_r) {
      uint32_t dest_to_find = src_data.current_length + 1;
      for (auto current_edge : graph->edges(src)) {
        GNode dst      = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        if (dest_to_find == dst_data.current_length) {
          float contrib =
              ((float)1 + dst_data.dependency) / dst_data.num_shortest_paths;
          src_data.dependency = src_data.dependency + contrib;
          bitset_dependency.set(src);
        }
      }
      src_data.dependency *= src_data.num_shortest_paths;
    }
  }
};

struct BC {
  Graph* graph;

  BC(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& dga) {
    globalRoundNumber = 0;
    // reset the graph aside from the between-cent measure
    InitializeIteration::go(_graph);
    // get distances and num paths
    ForwardPass::go(_graph, dga);

    // dependency calc only matters if there's a node with distance at
    // least 2
    if (globalRoundNumber > 2) {
      MiddleSync::go(_graph, infinity);
      BackwardPass::go(_graph, globalRoundNumber - 1);

      const auto& masters = _graph.masterNodesRange();
      // finally, since dependencies are finalized for this round at this
      // point, add them to the betweeness centrality measure on each node
      if (personality == GPU_CUDA) {
#ifdef GALOIS_ENABLE_GPU
        std::string impl_str(syncSubstrate->get_run_identifier("BC"));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        BC_masterNodes_cuda(cuda_ctx);
        StatTimer_cuda.stop();
#else
        abort();
#endif
      } else if (personality == CPU) {
        galois::do_all(
            galois::iterate(masters.begin(), masters.end()), BC(&_graph),
            galois::no_stats(),
            galois::loopname(
                syncSubstrate->get_run_identifier(std::string(REGION_NAME))
                    .c_str()));
      }
    }
  }

  /**
   * Adds dependency measure to BC measure
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.dependency > 0) {
      src_data.betweeness_centrality += src_data.dependency;
    }
  }
};

/******************************************************************************/
/* Sanity check */
/******************************************************************************/

struct Sanity {
  Graph* graph;

  galois::DGReduceMax<float>& DGAccumulator_max;
  galois::DGReduceMin<float>& DGAccumulator_min;
  galois::DGAccumulator<float>& DGAccumulator_sum;

  Sanity(Graph* _graph, galois::DGReduceMax<float>& _DGAccumulator_max,
         galois::DGReduceMin<float>& _DGAccumulator_min,
         galois::DGAccumulator<float>& _DGAccumulator_sum)
      : graph(_graph), DGAccumulator_max(_DGAccumulator_max),
        DGAccumulator_min(_DGAccumulator_min),
        DGAccumulator_sum(_DGAccumulator_sum) {}

  void static go(Graph& _graph, galois::DGReduceMax<float>& DGA_max,
                 galois::DGReduceMin<float>& DGA_min,
                 galois::DGAccumulator<float>& DGA_sum) {

    DGA_max.reset();
    DGA_min.reset();
    DGA_sum.reset();

    if (personality == GPU_CUDA) {
#ifdef GALOIS_ENABLE_GPU
      // std::string impl_str(syncSubstrate->get_run_identifier("Sanity"));
      std::string impl_str("Sanity");
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      float sum, max, min;
      Sanity_masterNodes_cuda(sum, max, min, cuda_ctx);
      DGA_sum += sum;
      DGA_max.update(max);
      DGA_min.update(min);
      StatTimer_cuda.stop();
#else
      abort();
#endif
    } else if (personality == CPU) {
      galois::do_all(galois::iterate(_graph.masterNodesRange().begin(),
                                     _graph.masterNodesRange().end()),
                     Sanity(&_graph, DGA_max, DGA_min, DGA_sum),
                     galois::no_stats(), galois::loopname("Sanity"));
    }

    float max_bc = DGA_max.reduce();
    float min_bc = DGA_min.reduce();
    float bc_sum = DGA_sum.reduce();

    // Only node 0 will print data
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("Max BC is ", max_bc, "\n");
      galois::gPrint("Min BC is ", min_bc, "\n");
      galois::gPrint("BC sum is ", bc_sum, "\n");
    }
  }

  /* Gets the max, min rank from all owned nodes and
   * also the sum of ranks */
  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);

    DGAccumulator_max.update(sdata.betweeness_centrality);
    DGAccumulator_min.update(sdata.betweeness_centrality);
    DGAccumulator_sum += sdata.betweeness_centrality;
  }
};

/******************************************************************************/
/* Make results */
/******************************************************************************/

std::vector<float> makeResultsCPU(std::unique_ptr<Graph>& hg) {
  std::vector<float> values;

  values.reserve(hg->numMasters());
  for (auto node : hg->masterNodesRange()) {
    values.push_back(hg->getData(node).betweeness_centrality);
  }

  return values;
}

#ifdef GALOIS_ENABLE_GPU
std::vector<float> makeResultsGPU(std::unique_ptr<Graph>& hg) {
  std::vector<float> values;

  values.reserve(hg->numMasters());
  for (auto node : hg->masterNodesRange()) {
    values.push_back(get_node_betweeness_centrality_cuda(cuda_ctx, node));
  }

  return values;
}
#else
std::vector<float> makeResultsGPU(std::unique_ptr<Graph>& /*unused*/) {
  abort();
}
#endif

std::vector<float> makeResults(std::unique_ptr<Graph>& hg) {
  switch (personality) {
  case CPU:
    return makeResultsCPU(hg);
  case GPU_CUDA:
    return makeResultsGPU(hg);
  default:
    abort();
  }
}

/******************************************************************************/
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name =
    "Betweeness Centrality Level by Level";
constexpr static const char* const desc =
    "Betweeness Centrality Level by Level on Distributed Galois.";
constexpr static const char* const url = nullptr;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);

  StatTimer_total.start();

  std::unique_ptr<Graph> h_graph;
#ifdef GALOIS_ENABLE_GPU
  std::tie(h_graph, syncSubstrate) =
      distGraphInitialization<NodeData, void>(&cuda_ctx);
#else
  std::tie(h_graph, syncSubstrate) = distGraphInitialization<NodeData, void>();
#endif

  if (!sourcesToUse.empty()) {
    sourceFile.open(sourcesToUse);
    std::vector<uint64_t> t(std::istream_iterator<uint64_t>{sourceFile},
                            std::istream_iterator<uint64_t>{});
    sourceVector = t;
    sourceFile.close();
  }

  bitset_num_shortest_paths.resize(h_graph->size());
  bitset_current_length.resize(h_graph->size());
  bitset_dependency.resize(h_graph->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  InitializeGraph::go((*h_graph));
  galois::runtime::getHostBarrier().wait();

  // shared DG accumulator among all steps
  galois::DGAccumulator<uint32_t> dga;

  // sanity dg accumulators
  galois::DGReduceMax<float> dga_max;
  galois::DGReduceMin<float> dga_min;
  galois::DGAccumulator<float> dga_sum;

  galois::runtime::reportStat_Single(std::string(REGION_NAME),
                                     std::string("NumSources"),
                                     (unsigned int)numberOfSources);
  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] BC::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));

    uint64_t loop_end = 1;
    bool sSources     = false;

    if (!singleSourceBC) {
      if (!numberOfSources) {
        loop_end = h_graph->globalSize();
      } else {
        loop_end = numberOfSources;
      }

      // if provided a file of sources to work with, use that
      if (!sourceVector.empty()) {
        if (loop_end > sourceVector.size()) {
          loop_end = sourceVector.size();
        }
        sSources = true;
      }
    }

    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    for (uint64_t i = 0; i < loop_end; i++) {
      if (singleSourceBC) {
        // only 1 source; specified start source in command line
        assert(loop_end == 1);
        galois::gDebug("This is single source node BC");
        current_src_node = startSource;
      } else if (sSources) {
        current_src_node = sourceVector[i];
      } else {
        // all sources
        current_src_node = i;
      }

      globalRoundNumber = 0;
      backRoundCount    = 0;

      StatTimer_main.start();
      BC::go(*h_graph, dga);
      StatTimer_main.stop();

      // Round reporting
      if (galois::runtime::getSystemNetworkInterface().ID == 0) {
        galois::runtime::reportStat_Single(
            REGION_NAME, syncSubstrate->get_run_identifier("NumRounds", i),
            globalRoundNumber);
        uint32_t backRounds;
        if (globalRoundNumber > 2) {
          backRounds = globalRoundNumber - 2;
        } else {
          backRounds = 0;
        }
        galois::runtime::reportStat_Single(
            REGION_NAME,
            syncSubstrate->get_run_identifier("NumForwardRounds", i),
            globalRoundNumber);
        galois::runtime::reportStat_Single(
            REGION_NAME, syncSubstrate->get_run_identifier("NumBackRounds", i),
            backRounds);
        galois::runtime::reportStat_Single(
            REGION_NAME, std::string("TotalRounds_") + std::to_string(run),
            globalRoundNumber + backRounds);
      }
    }

    Sanity::go(*h_graph, dga_max, dga_min, dga_sum);

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      galois::runtime::getHostBarrier().wait();
      (*syncSubstrate).set_num_run(run + 1);

      if (personality == GPU_CUDA) {
#ifdef GALOIS_ENABLE_GPU
        bitset_num_shortest_paths_reset_cuda(cuda_ctx);
        bitset_current_length_reset_cuda(cuda_ctx);
        bitset_dependency_reset_cuda(cuda_ctx);
#else
        abort();
#endif
      } else if (personality == CPU) {
        bitset_num_shortest_paths.reset();
        bitset_current_length.reset();
        bitset_dependency.reset();
      }

      InitializeGraph::go(*h_graph);
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  if (output) {
    std::vector<float> results = makeResults(h_graph);
    auto globalIDs             = h_graph->getMasterGlobalIDs();
    assert(results.size() == globalIDs.size());

    writeOutput(outputLocation, "betweenness_centrality", results.data(),
                results.size(), globalIDs.data());
  }

  return 0;
}
