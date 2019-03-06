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

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/AtomicHelpers.h"
#include "galois/graphs/LCGraph.h"
#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"

constexpr static const char* const REGION_NAME = "k-core";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;

//! Input file: should be symmetric graph
static cll::opt<std::string> inputFilename(cll::Positional,
                                          cll::desc("<input file (symmetric)>"),
                                          cll::Required);

//! Required k specification for k-core
static cll::opt<unsigned int> k_core_num("kcore", cll::desc("k-core value"),
                                         cll::Required);

//! Flag that forces user to be aware that they should be passing in a
//! symmetric graph
static cll::opt<bool> symmetricGraph("symmetricGraph",
  cll::desc("Flag should be used to make user aware they should be passing a "
            "symmetric graph to this program"),
  cll::init(false));

/******************************************************************************/
/* Graph structure declarations + other inits */
/******************************************************************************/
// Node deadness can be derived from current degree and k value, so no field
// necessary
struct NodeData {
  std::atomic<uint32_t> currentDegree;
};

//! Typedef for graph used, CSR graph
using Graph =
  galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<true>::type;
//! Typedef for node type in the CSR graph
using GNode = Graph::GraphNode;

//! Chunksize for for_each worklist: best chunksize will depend on input
constexpr static const unsigned CHUNK_SIZE = 64u;

/******************************************************************************/
/* Functions for running the algorithm */
/******************************************************************************/
/**
 * Initialize degree fields in graph with current degree. Since symmetric,
 * out edge count is equivalent to in-edge count.
 *
 * @param graph Graph to initialize degrees in
 */
void degreeCounting(Graph& graph) {
  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&] (GNode curNode) {
      NodeData& curData = graph.getData(curNode);
      curData.currentDegree.store(std::distance(graph.edge_begin(curNode),
                                                graph.edge_end(curNode)));
    },
    galois::loopname("DegreeCounting"),
    galois::no_stats()
  );
};

/**
 * Setup initial worklist of dead nodes.
 *
 * @param graph Graph to operate on
 * @param initialWorklist Empty worklist to be filled with dead nodes.
 */
void setupInitialWorklist(Graph& graph,
                          galois::InsertBag<GNode>& initialWorklist) {
  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&] (GNode curNode) {
      NodeData& curData = graph.getData(curNode);
      if (curData.currentDegree < k_core_num) {
        // dead node, add to initialWorklist for processing later
        initialWorklist.emplace(curNode);
      }
    },
    galois::loopname("InitialWorklistSetup"),
    galois::no_stats()
  );
}

/**
 * Starting with initial dead nodes, degree decrement and add to worklist
 * as they drop below k threshold until worklist is empty (i.e. no more dead
 * nodes).
 *
 * @param graph Graph to operate on
 * @param initialWorklist Worklist containing initial dead nodes
 */
void cascadeKCore(Graph& graph, galois::InsertBag<GNode>& initialWorklist) {
  galois::for_each(
    galois::iterate(initialWorklist),
    [&] (GNode deadNode, auto& ctx) {
      // decrement degree of all neighbors
      for (auto e : graph.edges(deadNode)) {
        GNode dest = graph.getEdgeDst(e);
        NodeData& destData = graph.getData(dest);
        uint32_t oldDegree = galois::atomicSubtract(destData.currentDegree, 1u);

        if (oldDegree == k_core_num) {
          // this thread was responsible for putting degree of destination
          // below threshold: add to worklist
          ctx.push(dest);
        }
      }
    },
    galois::no_conflicts(),
    galois::chunk_size<CHUNK_SIZE>(),
    galois::loopname("CascadeDeadNodes")
  );
}

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/
/**
 * Print number of nodes that are still alive.
 *
 * @param graph Graph to get alive count of
 */
void kCoreSanity(Graph& graph) {
  galois::GAccumulator<uint32_t> aliveNodes;
  aliveNodes.reset();

  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&] (GNode curNode) {
      NodeData& curData = graph.getData(curNode);
      if (curData.currentDegree >= k_core_num) {
        aliveNodes += 1;
      }
    },
    galois::loopname("KCoreSanityCheck"),
    galois::no_stats()
  );

  galois::gPrint("Number of nodes in the ", k_core_num, "-core is ",
                 aliveNodes.reduce(), "\n");
}

/******************************************************************************/
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name = "k-core";
constexpr static const char* const desc = "Finds the k-core of a graph, defined "
                                          "as the subgraph where all vertices "
                                          "have degree at least k.";
constexpr static const char* const url  = 0;

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  if (!symmetricGraph) {
    GALOIS_DIE("User did not pass in symmetric graph flag signifying they are "
               "aware this program needs to be passed a symmetric graph.");
  }

  // some initial stat reporting
  galois::gInfo("Worklist chunk size of ", CHUNK_SIZE, ": best size may depend"
                " on input.");
  galois::runtime::reportStat_Single(REGION_NAME, "ChunkSize", CHUNK_SIZE);
  galois::reportPageAlloc("MemAllocPre");

  galois::StatTimer totalTimer("TotalTime", REGION_NAME);
  totalTimer.start();

  // graph reading from disk
  galois::StatTimer graphReadingTimer("GraphConstructTime", REGION_NAME);
  graphReadingTimer.start();
  Graph graph;
  galois::graphs::readGraph(graph, inputFilename);
  graphReadingTimer.stop();

  // preallocate pages in memory so allocation doesn't occur during compute
  galois::StatTimer preallocTime("PreAllocTime", REGION_NAME);
  preallocTime.start();
  galois::preAlloc(std::max(
    (uint64_t)galois::getActiveThreads() * (graph.size() / 1000000),
    std::max(10u, galois::getActiveThreads()) * (size_t)10
  ));
  preallocTime.stop();
  galois::reportPageAlloc("MemAllocMid");

  // intialization of degrees
  degreeCounting(graph);

  // here begins main computation
  galois::StatTimer runtimeTimer;
  runtimeTimer.start();

  // worklist setup of initial dead ndoes
  galois::InsertBag<GNode> initialWorklist;
  setupInitialWorklist(graph, initialWorklist);

  // actual work; propagate deadness by decrementing degrees and adding dead
  // nodes to worklist
  cascadeKCore(graph, initialWorklist);
  runtimeTimer.stop();

  totalTimer.stop();
  galois::reportPageAlloc("MemAllocPost");

  // sanity check
  if (!skipVerify) {
    kCoreSanity(graph);
  }

  return 0;
}
