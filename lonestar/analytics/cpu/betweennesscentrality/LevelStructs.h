#ifndef GALOIS_BC_LEVEL
#define GALOIS_BC_LEVEL

#include "galois/AtomicHelpers.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/graphs/LCGraph.h"

#include <limits>
#include <fstream>

////////////////////////////////////////////////////////////////////////////////

static uint64_t levelCurrentSrcNode = 0;
// type of the num shortest paths variable
using LevelShortPathType = double;

// NOTE: types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct LevelNodeData {
  uint32_t currentDistance;
  std::atomic<LevelShortPathType> numShortestPaths;
  float dependency;
  float bc;
};

using LevelGraph =
    galois::graphs::LC_CSR_Graph<LevelNodeData, void>::with_no_lockable<
        true>::type::with_numa_alloc<true>::type;
using LevelGNode        = LevelGraph::GraphNode;
using LevelWorklistType = galois::InsertBag<LevelGNode, 4096>;

constexpr static const unsigned LEVEL_CHUNK_SIZE = 256u;

/******************************************************************************/
/* Functions for running the algorithm */
/******************************************************************************/
/**
 * Initialize node fields all to 0
 * @param graph LevelGraph to initialize
 */
void LevelInitializeGraph(LevelGraph& graph) {
  galois::do_all(
      galois::iterate(graph),
      [&](LevelGNode n) {
        LevelNodeData& nodeData   = graph.getData(n);
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
 * @param graph LevelGraph to reset iteration data
 */
void LevelInitializeIteration(LevelGraph& graph) {
  galois::do_all(
      galois::iterate(graph),
      [&](LevelGNode n) {
        LevelNodeData& nodeData = graph.getData(n);
        bool isSource           = (n == levelCurrentSrcNode);
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
galois::gstl::Vector<LevelWorklistType> LevelSSSP(LevelGraph& graph) {
  galois::gstl::Vector<LevelWorklistType> stackOfWorklists;
  uint32_t currentLevel = 0;

  // construct first level worklist which consists only of source
  stackOfWorklists.emplace_back();
  stackOfWorklists[0].emplace(levelCurrentSrcNode);

  // loop as long as current level's worklist is non-empty
  while (!stackOfWorklists[currentLevel].empty()) {
    // create worklist for next level
    stackOfWorklists.emplace_back();
    uint32_t nextLevel = currentLevel + 1;

    galois::do_all(
        galois::iterate(stackOfWorklists[currentLevel]),
        [&](LevelGNode n) {
          LevelNodeData& curData = graph.getData(n);
          GALOIS_ASSERT(curData.currentDistance == currentLevel);

          for (auto e : graph.edges(n)) {
            LevelGNode dest         = graph.getEdgeDst(e);
            LevelNodeData& destData = graph.getData(dest);

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
        galois::steal(), galois::chunk_size<LEVEL_CHUNK_SIZE>(),
        galois::no_stats(), galois::loopname("SSSP"));

    // move on to next level
    currentLevel++;
  }
  return stackOfWorklists;
}

/**
 * Backward phase: use worklist of nodes at each level to back-propagate
 * dependency values.
 *
 * @param graph LevelGraph to do backward Brandes dependency prop on
 */
void LevelBackwardBrandes(
    LevelGraph& graph,
    galois::gstl::Vector<LevelWorklistType>& stackOfWorklists) {
  // minus 3 because last one is empty, one after is leaf nodes, and one
  // to correct indexing to 0 index
  if (stackOfWorklists.size() >= 3) {
    uint32_t currentLevel = stackOfWorklists.size() - 3;

    // last level is ignored since it's just the source
    while (currentLevel > 0) {
      LevelWorklistType& currentWorklist = stackOfWorklists[currentLevel];
      uint32_t succLevel                 = currentLevel + 1;

      galois::do_all(
          galois::iterate(currentWorklist),
          [&](LevelGNode n) {
            LevelNodeData& curData = graph.getData(n);
            GALOIS_ASSERT(curData.currentDistance == currentLevel);

            for (auto e : graph.edges(n)) {
              LevelGNode dest         = graph.getEdgeDst(e);
              LevelNodeData& destData = graph.getData(dest);

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
          galois::steal(), galois::chunk_size<LEVEL_CHUNK_SIZE>(),
          galois::no_stats(), galois::loopname("Brandes"));

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
 * @param graph LevelGraph to sanity check
 */
void LevelSanity(LevelGraph& graph) {
  galois::GReduceMax<float> accumMax;
  galois::GReduceMin<float> accumMin;
  galois::GAccumulator<float> accumSum;
  accumMax.reset();
  accumMin.reset();
  accumSum.reset();

  // get max, min, sum of BC values using accumulators and reducers
  galois::do_all(
      galois::iterate(graph),
      [&](LevelGNode n) {
        LevelNodeData& nodeData = graph.getData(n);
        accumMax.update(nodeData.bc);
        accumMin.update(nodeData.bc);
        accumSum += nodeData.bc;
      },
      galois::no_stats(), galois::loopname("LevelSanity"));

  galois::gPrint("Max BC is ", accumMax.reduce(), "\n");
  galois::gPrint("Min BC is ", accumMin.reduce(), "\n");
  galois::gPrint("BC sum is ", accumSum.reduce(), "\n");
}

/******************************************************************************/
/* Running */
/******************************************************************************/

void doLevelBC() {
  // reading in list of sources to operate on if provided
  std::ifstream sourceFile;
  std::vector<uint64_t> sourceVector;

  // some initial stat reporting
  galois::gInfo("Worklist chunk size of ", LEVEL_CHUNK_SIZE,
                ": best size may depend on input.");
  galois::runtime::reportStat_Single(REGION_NAME, "ChunkSize",
                                     LEVEL_CHUNK_SIZE);
  galois::reportPageAlloc("MemAllocPre");

  // LevelGraph construction
  galois::StatTimer graphConstructTimer("TimerConstructGraph", "BFS");
  graphConstructTimer.start();
  LevelGraph graph;
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
    if (!numOfSources) {
      loop_end = graph.size();
    } else {
      loop_end = numOfSources;
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
  LevelInitializeGraph(graph);

  galois::gInfo("Beginning main computation");
  galois::StatTimer execTime("Timer_0");

  // loop over all specified sources for SSSP/Brandes calculation
  for (uint64_t i = 0; i < loop_end; i++) {
    if (singleSourceBC) {
      // only 1 source; specified start source in command line
      assert(loop_end == 1);
      galois::gDebug("This is single source node BC");
      levelCurrentSrcNode = startSource;
    } else if (sSources) {
      levelCurrentSrcNode = sourceVector[i];
    } else {
      // all sources
      levelCurrentSrcNode = i;
    }

    // here begins main computation
    execTime.start();
    LevelInitializeIteration(graph);
    // worklist; last one will be empty
    galois::gstl::Vector<LevelWorklistType> worklists = LevelSSSP(graph);
    LevelBackwardBrandes(graph, worklists);
    execTime.stop();
  }

  galois::reportPageAlloc("MemAllocPost");

  // sanity checking numbers
  LevelSanity(graph);

  // Verify, i.e. print out graph data for examination
  // @todo print to file instead of stdout
  if (output) {
    char* v_out = (char*)malloc(40);
    for (auto ii = graph.begin(); ii != graph.end(); ++ii) {
      // outputs betweenness centrality
      sprintf(v_out, "%u %.9f\n", (*ii), graph.getData(*ii).bc);
      galois::gPrint(v_out);
    }
    free(v_out);
  }
}
#endif
