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

//#define BCDEBUG

constexpr static const char* const REGION_NAME = "BC";

#include <iostream>
#include <limits>

#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

#include "resilience.h"

// type of the num shortest paths variable
using ShortPathType = double;

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
static cll::opt<unsigned long long>
    startSource("startNode", // not uint64_t due to a bug in llvm cl
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
static uint64_t globalRound = 0;
static uint64_t toRepeat = 0;
static uint64_t backInRound = 0;

static uint64_t addForward = 0;
static uint64_t addBackward = 0;

// NOTE: types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  // SSSP vars
  std::atomic<uint32_t> current_length;
  // Betweeness centrality vars
  ShortPathType num_shortest_paths;
  std::atomic<ShortPathType> path_accum;
  float dependency;
  float dep_accum;
  float betweeness_centrality;

  void dump() {
    galois::gPrint("DUMP: ", current_length.load(), " ",
                   num_shortest_paths, " ", dependency, "\n");
  }
};

// reading in list of sources to operate on if provided
std::ifstream sourceFile;
std::vector<uint64_t> sourceVector;

using Graph = galois::graphs::DistGraph<NodeData, void>;
using GNode = typename Graph::GraphNode;

// bitsets for tracking updates
galois::DynamicBitSet bitset_current_length;
galois::DynamicBitSet bitset_path_accum;
galois::DynamicBitSet bitset_dep_accum;

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

      galois::do_all(
          // pass in begin/end to not use local thread ranges
          galois::iterate(allNodes.begin(), allNodes.end()),
          InitializeGraph{&_graph}, galois::no_stats(),
          galois::loopname("InitializeGraph"));
  }

  /* Functor passed into the Galois operator to carry out initialization;
   * reset everything */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.betweeness_centrality = 0;
    src_data.num_shortest_paths    = 0;
    src_data.path_accum            = 0;
    src_data.dependency            = 0;
    src_data.dep_accum             = 0;
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

      galois::do_all(
          galois::iterate(allNodes.begin(), allNodes.end()),
          InitializeIteration{infinity, current_src_node, &_graph},
          galois::loopname(_graph.get_run_identifier("InitializeIteration").c_str()),
          galois::no_stats());
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
    src_data.path_accum       = 0;
    src_data.dep_accum        = 0;
    src_data.dependency       = 0;
  }
};

/* This is used to reset node data when switching to a difference source */
struct InitializeIteration_ForwardCrash {
  const uint32_t& local_infinity;
  const uint64_t& local_current_src_node;
  Graph* graph;

  InitializeIteration_ForwardCrash(const uint32_t& _local_infinity,
                      const uint64_t& _local_current_src_node, Graph* _graph)
      : local_infinity(_local_infinity),
        local_current_src_node(_local_current_src_node), graph(_graph) {}

  /* Reset necessary graph metadata for next iteration of SSSP */
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

      galois::do_all(
          galois::iterate(allNodes.begin(), allNodes.end()),
          InitializeIteration_ForwardCrash{infinity, current_src_node, &_graph},
          galois::loopname(_graph.get_run_identifier("InitializeIteration_ForwardCrash").c_str()),
          galois::no_stats());
  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.betweeness_centrality = 0;
    src_data.path_accum            = 0;
    src_data.dep_accum             = 0;

    bool is_source = graph->getGID(src) == local_current_src_node;

    if (!is_source) {
      src_data.current_length     = local_infinity;
      src_data.num_shortest_paths = 0;
    } else {
      src_data.current_length     = 0;
      src_data.num_shortest_paths = 1;
    }
    src_data.dependency       = 0;
  }
};

void PathAccum(Graph& _graph) {
  const auto& allNodes = _graph.allNodesRange();
  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()),
    [&] (auto src) {
      NodeData& src_data = _graph.getData(src);

      if (src_data.path_accum > 0 && src_data.num_shortest_paths == 0) {
        src_data.num_shortest_paths = src_data.path_accum;
      }
      src_data.path_accum = 0;
    },
    galois::loopname(_graph.get_run_identifier("ForwardPass").c_str()),
    galois::no_stats()
  );
};

void RecoveryPathAccum(Graph& _graph) {
  const auto& allNodes = _graph.allNodesRange();
  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()),
    [&] (auto src) {
      NodeData& src_data = _graph.getData(src);
      if (src_data.path_accum > 0 && src_data.num_shortest_paths == 0) {
        src_data.num_shortest_paths = src_data.path_accum;
      }
      src_data.path_accum = 0;
    },
    galois::loopname(_graph.get_run_identifier("RECOVERYForwardPass").c_str()),
    galois::no_stats()
  );
};

struct RecoveryForward {
  Graph* graph;
  uint32_t r;

  RecoveryForward(Graph* _graph, uint32_t _r) : graph(_graph), r(_r) {}

  void static go(Graph& _graph) {
    _graph.sync<writeAny, readAny, Reduce_min_current_length, Broadcast_current_length>(
        "RECOVERYForwardPass");
    _graph.sync<writeAny, readAny, Reduce_max_num_shortest_paths, Broadcast_num_shortest_paths>(
        "RECOVERYForwardPass");
    
    bool crashed = isCrashed();
    bitset_current_length.reset();
    bitset_path_accum.reset();
    
    if (crashed) {
      for (unsigned i = 0; i < toRepeat; i++) {
        // length operator
        const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
        galois::do_all(
          galois::iterate(nodesWithEdges),
          RecoveryForward(&_graph, i),
          galois::loopname(_graph.get_run_identifier("RECOVERYForwardPass").c_str()),
          galois::steal(),
          galois::no_stats()
        );

        _graph.sync<writeDestination, readAny, Reduce_min_current_length,
                    Bitset_current_length>("RECOVERYForwardPass");
        _graph.sync<writeDestination, readAny, Reduce_add_path_accum,
                    Bitset_path_accum>("RECOVERYForwardPass");
     
        // path accum operator
        RecoveryPathAccum(_graph);
      }
    } else {
      for (unsigned i = 0; i < toRepeat; i++) {
        // dummy syncs
        _graph.sync<writeDestination, readAny, Reduce_min_current_length,
                    Broadcast_current_length,
                    Bitset_current_length>("RECOVERYForwardPass");
        _graph.sync<writeDestination, readAny, Reduce_add_path_accum,
                    Bitset_path_accum>("RECOVERYForwardPass");
        RecoveryPathAccum(_graph);
      }
    }
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length == r) {
      for (auto current_edge : graph->edges(src)) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);
        uint32_t new_dist = 1 + src_data.current_length;
        uint32_t old = galois::atomicMin(dst_data.current_length, new_dist);

        if (old > new_dist) {
          bitset_current_length.set(dst);
          galois::atomicAdd(dst_data.path_accum, src_data.num_shortest_paths);
          bitset_path_accum.set(dst);
        } else if (old == new_dist) {
          galois::atomicAdd(dst_data.path_accum, src_data.num_shortest_paths);
          bitset_path_accum.set(dst);
        }
      }
    }

  }
};


/**
 * Forward pass does level by level BFS to find distances and number of
 * shortest paths
 */
struct ForwardPass {
  Graph* graph;
  galois::DGAccumulator<uint32_t>& dga;
  uint32_t r;

  ForwardPass(Graph* _graph, galois::DGAccumulator<uint32_t>& _dga, 
              uint32_t roundNum)
    : graph(_graph), dga(_dga), r(roundNum) {}

  /**
   * Level by level BFS while also finding number of shortest paths to a
   * particular node in the BFS tree.
   *
   * @param _graph Graph to use
   * @param _dga distributed accumulator
   * @param[out] roundNumber Number of rounds taken to finish
   */
  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& _dga, 
                 uint32_t& roundNumber) {

    roundNumber = 0;
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      if (enableFT && recoveryScheme == CP) {
        saveCheckpointToDisk(globalRound, _graph);
      }

      _dga.reset();

      galois::do_all(
        galois::iterate(nodesWithEdges),
        ForwardPass(&_graph, _dga, roundNumber),
        galois::loopname(_graph.get_run_identifier("ForwardPass").c_str()),
        galois::steal(),
        galois::no_stats()
      );

      // synchronize distances and shortest paths
      // read any because a destination node without the correct distance
      // may use a different distance (leading to incorrectness)
      _graph.sync<writeDestination, readAny, Reduce_min_current_length,
                  Bitset_current_length>("ForwardPass");
      _graph.sync<writeDestination, readAny, Reduce_add_path_accum,
                  Bitset_path_accum>("ForwardPass");

      PathAccum(_graph);

      if (enableFT && (globalRound == crashIteration)) {
        toRepeat = roundNumber + 1;
        crashSite<RecoveryForward, InitializeIteration_ForwardCrash>(_graph);

        if (recoveryScheme == CP) {
          // redo last round
          galois::do_all(
            galois::iterate(nodesWithEdges),
            ForwardPass(&_graph, _dga, roundNumber),
            galois::loopname(_graph.get_run_identifier("ForwardPass").c_str()),
            galois::steal(),
            galois::no_stats()
          );
    
          // synchronize distances and shortest paths
          // read any because a destination node without the correct distance
          // may use a different distance (leading to incorrectness)
          _graph.sync<writeDestination, readAny, Reduce_min_current_length,
                      Bitset_current_length>("ForwardPass");
          _graph.sync<writeDestination, readAny, Reduce_add_path_accum,
                      Bitset_path_accum>("ForwardPass");
          PathAccum(_graph);

          addForward = 1;
        } else {
          addForward = toRepeat;
        }
      }

      globalRound++;
      roundNumber++;
    } while (_dga.reduce(_graph.get_run_identifier()));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length == r) {
      for (auto current_edge : graph->edges(src)) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);
        uint32_t new_dist = 1 + src_data.current_length;
        uint32_t old = galois::atomicMin(dst_data.current_length, new_dist);

        if (old > new_dist) {
          bitset_current_length.set(dst);
          galois::atomicAdd(dst_data.path_accum, src_data.num_shortest_paths);
          bitset_path_accum.set(dst);
          dga += 1;
        } else if (old == new_dist) {
          galois::atomicAdd(dst_data.path_accum, src_data.num_shortest_paths);
          bitset_path_accum.set(dst);
          dga += 1;
        }
      }
    }
  }
};

void DepAccum(Graph& _graph) {
  const auto& allNodes = _graph.allNodesRange();
  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()),
    [&] (auto src) {
      NodeData& src_data = _graph.getData(src);

      if (src_data.dep_accum > 0 && src_data.dependency == 0) {
        src_data.dependency = src_data.dep_accum;
      }
      src_data.dep_accum = 0;
    },
    galois::loopname(_graph.get_run_identifier("BackwardPass").c_str()),
    galois::no_stats()
  );
};

void RecoveryDepAccum(Graph& _graph) {
  const auto& allNodes = _graph.allNodesRange();
  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()),
    [&] (auto src) {
      NodeData& src_data = _graph.getData(src);

      if (src_data.dep_accum > 0 && src_data.dependency == 0) {
        src_data.dependency = src_data.dep_accum;
      }
      src_data.dep_accum = 0;
    },
    galois::loopname(_graph.get_run_identifier("RECOVERYBackwardPass").c_str()),
    galois::no_stats()
  );
};

struct RecoveryBackward {
  Graph* graph;
  uint32_t r;

  RecoveryBackward(Graph* _graph, uint32_t _r) : graph(_graph), r(_r) {}

  void static go(Graph& _graph) {
    // get dependency from proxies if it exists
    _graph.sync<writeAny, readAny, Reduce_max_dependency, 
                Broadcast_dependency>("RECOVERYBackwardPass");

    bool crashed = isCrashed();
    bitset_dep_accum.reset();
    
    if (crashed) {
      for (unsigned i = backInRound; i > toRepeat; i--) {
        // length operator
        const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
        galois::do_all(
          galois::iterate(nodesWithEdges),
          RecoveryBackward(&_graph, i),
          galois::loopname(_graph.get_run_identifier("RECOVERYBackwardPass").c_str()),
          galois::steal(),
          galois::no_stats()
        );

        _graph.sync<writeSource, readAny, Reduce_add_dep_accum,
                    Bitset_dep_accum>("RECOVERYBackwardPass");
     
        // path accum operator
        RecoveryDepAccum(_graph);
      }
    } else {
      for (unsigned i = backInRound; i > toRepeat; i--) {
        // dummy sync
        _graph.sync<writeSource, readAny, Reduce_add_dep_accum,
                    Bitset_dep_accum>("RECOVERYBackwardPass");
        RecoveryDepAccum(_graph);
      }
    }
  }

  /**
   * If on the correct level, calculate self-depndency by checking successor
   * nodes.
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length == r) {
      uint32_t dest_to_find = src_data.current_length + 1;
      for (auto current_edge : graph->edges(src)) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        if (dest_to_find == dst_data.current_length.load()) {
          float contrib = (1.0 + dst_data.dependency) /
                          dst_data.num_shortest_paths;
          galois::add(src_data.dep_accum, contrib);
          bitset_dep_accum.set(src);
        }
      }
      src_data.dep_accum *= src_data.num_shortest_paths;
    }
  }
};

struct Dummy {
  void static go(Graph& _graph) {
  }
};

/**
 * Propagate dependency backward by iterating backward over levels of BFS tree
 */
struct BackwardPass {
  Graph* graph;
  uint32_t r;

  BackwardPass(Graph* _graph, uint32_t roundNum) : graph(_graph), r(roundNum) {}

  void static go(Graph& _graph, uint32_t roundNumber) {
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    backInRound = roundNumber;

    for (uint32_t i = roundNumber; i > 0; i--) {
      if (enableFT && recoveryScheme == CP) {
        saveCheckpointToDisk(globalRound, _graph);
      }

      galois::do_all(
        galois::iterate(nodesWithEdges),
        BackwardPass(&_graph, i),
        galois::loopname(_graph.get_run_identifier("BackwardPass").c_str()),
        galois::steal(),
        galois::no_stats()
      );

      _graph.sync<writeSource, readAny, Reduce_add_dep_accum,
                  Bitset_dep_accum>("BackwardPass");
      DepAccum(_graph);

      if (enableFT && (globalRound == crashIteration)) {
        toRepeat = i - 1;
        crashSite<RecoveryBackward, Dummy>(_graph);

        if (recoveryScheme == CP) {
          // repeat last round
          galois::do_all(
            galois::iterate(nodesWithEdges),
            BackwardPass(&_graph, i),
            galois::loopname(_graph.get_run_identifier("BackwardPass").c_str()),
            galois::steal(),
            galois::no_stats()
          );
    
          _graph.sync<writeSource, readAny, Reduce_add_dep_accum,
                      Bitset_dep_accum>("BackwardPass");
          DepAccum(_graph);
          addBackward = 1;
        } else {
          addBackward = roundNumber - toRepeat;
        }
      }

      globalRound++;
    }
  }

  /**
   * If on the correct level, calculate self-depndency by checking successor
   * nodes.
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length == r) {
      uint32_t dest_to_find = src_data.current_length + 1;
      for (auto current_edge : graph->edges(src)) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        if (dest_to_find == dst_data.current_length.load()) {
          float contrib = (1.0 + dst_data.dependency) /
                          dst_data.num_shortest_paths;
          galois::add(src_data.dep_accum, contrib);
          bitset_dep_accum.set(src);
        }
      }
      src_data.dep_accum *= src_data.num_shortest_paths;
    }
  }
};


struct BC {
  Graph* graph;

  BC(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& dga,
                 uint32_t& roundNum) {
    roundNum = 0;
    // reset the graph aside from the between-cent measure
    InitializeIteration::go(_graph);
    // get distances and num paths
    ForwardPass::go(_graph, dga, roundNum);

    // switch to hybrid if using RS otherwise keep CP
    if (recoveryScheme == RS) {
      recoveryScheme = HR;
    }
    // dependency calc only matters if there's a node with distance at
    // least 2
    if (roundNum > 2) {
      // create checkpoint after phase
      galois::gPrint("Checkpoint after Forward Phase\n");
      saveCheckpointToDisk(0, _graph);

      BackwardPass::go(_graph, roundNum - 2);

      const auto& masters = _graph.masterNodesRange();
      // finally, since dependencies are finalized for this round at this
      // point, add them to the betweeness centrality measure on each node
      galois::do_all(
        galois::iterate(masters.begin(), masters.end()),
        BC(&_graph),
        galois::no_stats(),
        galois::loopname(_graph.get_run_identifier("BC").c_str())
      );
    }
  }

  /**
   * Adds dependency measure to BC measure 
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.dependency > 0) {
      galois::add(src_data.betweeness_centrality, src_data.dependency);
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

    galois::do_all(galois::iterate(_graph.masterNodesRange().begin(),
                                   _graph.masterNodesRange().end()),
                   Sanity(&_graph, DGA_max, DGA_min, DGA_sum),
                   galois::no_stats(), galois::loopname("Sanity"));

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
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name = "Betweeness Centrality Level by Level";
constexpr static const char* const desc =
    "Betweeness Centrality Level by Level on Distributed Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);

  StatTimer_total.start();

  Graph* h_graph = distGraphInitialization<NodeData, void>();

  if (sourcesToUse != "") {
    sourceFile.open(sourcesToUse);
    std::vector<uint64_t> t(std::istream_iterator<uint64_t>{sourceFile},
                            std::istream_iterator<uint64_t>{});
    sourceVector = t;
    sourceFile.close();
  }

  bitset_path_accum.resize(h_graph->size());
  bitset_current_length.resize(h_graph->size());
  bitset_dep_accum.resize(h_graph->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  InitializeGraph::go((*h_graph));
  galois::runtime::getHostBarrier().wait();

  // shared DG accumulator among all steps
  galois::DGAccumulator<uint32_t> dga;

  // sanity dg accumulators
  galois::DGReduceMax<float> dga_max;
  galois::DGReduceMin<float> dga_min;
  galois::DGAccumulator<float> dga_sum;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] BC::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    current_src_node = startSource;
    globalRound = 0;
    toRepeat = 0;
    backInRound = 0;
    addForward = 0;
    addBackward = 0;

    uint32_t roundNum = 0;

    StatTimer_main.start();
    BC::go(*h_graph, dga, roundNum);
    StatTimer_main.stop();

    // Round reporting
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("global round count ", globalRound, "\n");
      galois::runtime::reportStat_Single(REGION_NAME,
        h_graph->get_run_identifier("NumRounds"), roundNum);
      uint32_t backRounds;
      if (roundNum > 2) {
        backRounds = roundNum - 2;
      } else {
        backRounds = 0;
      }
      galois::runtime::reportStat_Single(REGION_NAME,
        h_graph->get_run_identifier("NumForwardRounds"), roundNum + addForward);
      galois::runtime::reportStat_Single(REGION_NAME,
        h_graph->get_run_identifier("NumBackRounds"), backRounds + addBackward);
      galois::runtime::reportStat_Tsum(REGION_NAME,
        std::string("TotalRounds_") + std::to_string(run), roundNum + backRounds +
                                                           addForward + addBackward);
    }

    Sanity::go(*h_graph, dga_max, dga_min, dga_sum);

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      galois::runtime::getHostBarrier().wait();
      (*h_graph).set_num_run(run + 1);

      {
        bitset_path_accum.reset();
        bitset_current_length.reset();
        bitset_dep_accum.reset();
      }


      InitializeGraph::go((*h_graph));
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  // Verify, i.e. print out graph data for examination
  if (verify) {
    char* v_out = (char*)malloc(40);
      for (auto ii = (*h_graph).masterNodesRange().begin();
           ii != (*h_graph).masterNodesRange().end(); ++ii) {
        // outputs betweenness centrality
        sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
                (*h_graph).getData(*ii).betweeness_centrality);

        // outputs length
        //sprintf(v_out, "%lu %d\n", (*h_graph).getGID(*ii),
        //        (*h_graph).getData(*ii).current_length.load());

        // outputs length + num paths
        //sprintf(v_out, "%lu %d %f\n", (*h_graph).getGID(*ii),
        //        (*h_graph).getData(*ii).current_length.load(),
        //        (*h_graph).getData(*ii).num_shortest_paths);

        galois::runtime::printOutput(v_out);
      }
    free(v_out);
  }

  return 0;
}
