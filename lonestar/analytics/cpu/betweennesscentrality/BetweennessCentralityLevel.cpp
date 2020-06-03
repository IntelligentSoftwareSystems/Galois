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

constexpr static const char* const REGION_NAME = "BC";

#include "galois/AtomicHelpers.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "Lonestar/BoilerPlate.h"

#include "llvm/Support/CommandLine.h"

#include <limits>
#include <fstream>

// type of the num shortest paths variable
using ShortPathType = double;

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string>
    sourcesToUse("sourcesToUse",
                 cll::desc("Whitespace separated list of sources in a file to "
                           "use in BC (default empty)"),
                 cll::init(""));
static cll::opt<bool>
    singleSourceBC("singleSource",
                   cll::desc("Use for single source BC (default off)"),
                   cll::init(false));
static cll::opt<uint64_t>
    startSource("startNode",
                cll::desc("Starting source node used for "
                          "betweeness-centrality (default 0); works with "
                          "singleSource flag only"),
                cll::init(0));
static cll::opt<unsigned int>
    numberOfSources("numOfSources",
                    cll::desc("Number of sources to use for "
                              "betweeness-centraility (default all)"),
                    cll::init(0));
static cll::opt<bool> verify("verify",
                             cll::desc("Flag to verify (default: false)"),
                             cll::init(false));

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/
const uint32_t infinity        = std::numeric_limits<uint32_t>::max() / 4;
static uint64_t currentSrcNode = 0;

// NOTE: types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  uint32_t currentDistance;
  std::atomic<ShortPathType> numShortestPaths;
  float dependency;
  float bc;
};

// reading in list of sources to operate on if provided
std::ifstream sourceFile;
std::vector<uint64_t> sourceVector;

using Graph = galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<
    true>::type::with_numa_alloc<true>::type;
using GNode        = Graph::GraphNode;
using WorklistType = galois::InsertBag<GNode, 4096>;

constexpr static const unsigned CHUNK_SIZE = 256u;

/******************************************************************************/
/* Functions for running the algorithm */
/******************************************************************************/
/**
 * Initialize node fields all to 0
 * @param graph Graph to initialize
 */
void InitializeGraph(Graph& graph) {
  galois::do_all(
      galois::iterate(graph),
      [&](GNode n) {
        NodeData& nodeData        = graph.getData(n);
        nodeData.currentDistance  = 0;
        nodeData.numShortestPaths = 0;
        nodeData.dependency       = 0;
        nodeData.bc               = 0;
      },
      galois::no_stats(), galois::loopname("InitializeGraph"));
}

/**
 * Resets data associated to start a new SSSP with a new source.
 *
 * @param graph Graph to reset iteration data
 */
void InitializeIteration(Graph& graph) {
  galois::do_all(
      galois::iterate(graph),
      [&](GNode n) {
        NodeData& nodeData = graph.getData(n);
        bool isSource      = (n == currentSrcNode);
        // source nodes have distance 0 and initialize short paths to 1, else
        // distance is infinity with 0 short paths
        if (!isSource) {
          nodeData.currentDistance  = infinity;
          nodeData.numShortestPaths = 0;
        } else {
          nodeData.currentDistance  = 0;
          nodeData.numShortestPaths = 1;
        }
        // dependency reset for new source
        nodeData.dependency = 0;
      },
      galois::no_stats(), galois::loopname("InitializeIteration"));
};

/**
 * Forward phase: SSSP to determine DAG and get shortest path counts.
 *
 * Worklist-based push. Save worklists on a stack for reuse in backward
 * Brandes dependency propagation.
 */
galois::gstl::Vector<WorklistType> SSSP(Graph& graph) {
  galois::gstl::Vector<WorklistType> stackOfWorklists;
  uint32_t currentLevel = 0;

  // construct first level worklist which consists only of source
  stackOfWorklists.emplace_back();
  stackOfWorklists[0].emplace(currentSrcNode);

  // loop as long as current level's worklist is non-empty
  while (!stackOfWorklists[currentLevel].empty()) {
    // create worklist for next level
    stackOfWorklists.emplace_back();
    uint32_t nextLevel = currentLevel + 1;

    galois::do_all(
        galois::iterate(stackOfWorklists[currentLevel]),
        [&](GNode n) {
          NodeData& curData = graph.getData(n);
          GALOIS_ASSERT(curData.currentDistance == currentLevel);

          for (auto e : graph.edges(n)) {
            GNode dest         = graph.getEdgeDst(e);
            NodeData& destData = graph.getData(dest);

            if (destData.currentDistance == infinity) {
              uint32_t oldVal = __sync_val_compare_and_swap(
                  &(destData.currentDistance), infinity, nextLevel);
              // only 1 thread should add to worklist
              if (oldVal == infinity) {
                stackOfWorklists[nextLevel].emplace(dest);
              }

              galois::atomicAdd(destData.numShortestPaths,
                                curData.numShortestPaths.load());
            } else if (destData.currentDistance == nextLevel) {
              galois::atomicAdd(destData.numShortestPaths,
                                curData.numShortestPaths.load());
            }
          }
        },
        galois::steal(), galois::chunk_size<CHUNK_SIZE>(), galois::no_stats(),
        galois::loopname("SSSP"));

    // move on to next level
    currentLevel++;
  }
  return stackOfWorklists;
}

/**
 * Backward phase: use worklist of nodes at each level to back-propagate
 * dependency values.
 *
 * @param graph Graph to do backward Brandes dependency prop on
 */
void BackwardBrandes(Graph& graph,
                     galois::gstl::Vector<WorklistType>& stackOfWorklists) {
  // minus 3 because last one is empty, one after is leaf nodes, and one
  // to correct indexing to 0 index
  if (stackOfWorklists.size() >= 3) {
    uint32_t currentLevel = stackOfWorklists.size() - 3;

    // last level is ignored since it's just the source
    while (currentLevel > 0) {
      WorklistType& currentWorklist = stackOfWorklists[currentLevel];
      uint32_t succLevel            = currentLevel + 1;

      galois::do_all(
          galois::iterate(currentWorklist),
          [&](GNode n) {
            NodeData& curData = graph.getData(n);
            GALOIS_ASSERT(curData.currentDistance == currentLevel);

            for (auto e : graph.edges(n)) {
              GNode dest         = graph.getEdgeDst(e);
              NodeData& destData = graph.getData(dest);

              if (destData.currentDistance == succLevel) {
                // grab dependency, add to self
                float contrib = ((float)1 + destData.dependency) /
                                destData.numShortestPaths;
                curData.dependency = curData.dependency + contrib;
              }
            }

            // multiply at end to get final dependency value
            curData.dependency *= curData.numShortestPaths;
            // accumulate dependency into bc
            curData.bc += curData.dependency;
          },
          galois::steal(), galois::chunk_size<CHUNK_SIZE>(), galois::no_stats(),
          galois::loopname("Brandes"));

      // move on to next level lower
      currentLevel--;
    }
  }
}

/******************************************************************************/
/* Sanity check */
/******************************************************************************/

/**
 * Get some sanity numbers (max, min, sum of BC)
 *
 * @param graph Graph to sanity check
 */
void Sanity(Graph& graph) {
  galois::GReduceMax<float> accumMax;
  galois::GReduceMin<float> accumMin;
  galois::GAccumulator<float> accumSum;
  accumMax.reset();
  accumMin.reset();
  accumSum.reset();

  // get max, min, sum of BC values using accumulators and reducers
  galois::do_all(
      galois::iterate(graph),
      [&](GNode n) {
        NodeData& nodeData = graph.getData(n);
        accumMax.update(nodeData.bc);
        accumMin.update(nodeData.bc);
        accumSum += nodeData.bc;
      },
      galois::no_stats(), galois::loopname("Sanity"));

  galois::gPrint("Max BC is ", accumMax.reduce(), "\n");
  galois::gPrint("Min BC is ", accumMin.reduce(), "\n");
  galois::gPrint("BC sum is ", accumSum.reduce(), "\n");
}

/******************************************************************************/
/* Main method for running */
/******************************************************************************/
constexpr static const char* const name =
    "Betweeness Centrality Level by Level";
constexpr static const char* const desc =
    "Betweeness Centrality, level by level, using synchronous BFS and Brandes "
    "backward dependency propagation.";

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, nullptr, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  // some initial stat reporting
  galois::gInfo("Worklist chunk size of ", CHUNK_SIZE,
                ": best size may depend"
                " on input.");
  galois::runtime::reportStat_Single(REGION_NAME, "ChunkSize", CHUNK_SIZE);
  galois::reportPageAlloc("MemAllocPre");

  // Graph construction
  galois::StatTimer graphConstructTimer("TimerConstructGraph", "BFS");
  graphConstructTimer.start();
  Graph graph;
  galois::graphs::readGraph(graph, inputFile);
  graphConstructTimer.stop();
  galois::gInfo("Graph construction complete");

  // preallocate pages in memory so allocation doesn't occur during compute
  galois::StatTimer preallocTime("PreAllocTime", REGION_NAME);
  preallocTime.start();
  galois::preAlloc(
      std::max(size_t{galois::getActiveThreads()} * (graph.size() / 2000000),
               std::max(10U, galois::getActiveThreads()) * size_t{10}));
  preallocTime.stop();
  galois::reportPageAlloc("MemAllocMid");

  // If particular set of sources was specified, use them
  if (sourcesToUse != "") {
    sourceFile.open(sourcesToUse);
    std::vector<uint64_t> t(std::istream_iterator<uint64_t>{sourceFile},
                            std::istream_iterator<uint64_t>{});
    sourceVector = t;
    sourceFile.close();
  }

  // determine how many sources to loop over based on command line args
  uint64_t loop_end = 1;
  bool sSources     = false;
  if (!singleSourceBC) {
    if (!numberOfSources) {
      loop_end = graph.size();
    } else {
      loop_end = numberOfSources;
    }

    // if provided a file of sources to work with, use that
    if (sourceVector.size() != 0) {
      if (loop_end > sourceVector.size()) {
        loop_end = sourceVector.size();
      }
      sSources = true;
    }
  }

  // graph initialization, then main loop
  InitializeGraph(graph);

  galois::gInfo("Beginning main computation");
  galois::StatTimer execTime("Timer_0");

  // loop over all specified sources for SSSP/Brandes calculation
  for (uint64_t i = 0; i < loop_end; i++) {
    if (singleSourceBC) {
      // only 1 source; specified start source in command line
      assert(loop_end == 1);
      galois::gDebug("This is single source node BC");
      currentSrcNode = startSource;
    } else if (sSources) {
      currentSrcNode = sourceVector[i];
    } else {
      // all sources
      currentSrcNode = i;
    }

    // here begins main computation
    execTime.start();
    InitializeIteration(graph);
    // worklist; last one will be empty
    galois::gstl::Vector<WorklistType> worklists = SSSP(graph);
    BackwardBrandes(graph, worklists);
    execTime.stop();
  }

  galois::reportPageAlloc("MemAllocPost");

  // sanity checking numbers
  Sanity(graph);

  // Verify, i.e. print out graph data for examination
  if (verify) {
    char* v_out = (char*)malloc(40);
    for (auto ii = graph.begin(); ii != graph.end(); ++ii) {
      // outputs betweenness centrality
      sprintf(v_out, "%u %.9f\n", (*ii), graph.getData(*ii).bc);
      galois::gPrint(v_out);
    }
    free(v_out);
  }

  totalTime.stop();

  return 0;
}
