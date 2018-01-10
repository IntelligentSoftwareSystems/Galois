/** Betweeness Centrality (PR) -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
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
 *
 * @section Description
 *
 * Compute Betweeness-Centrality on distributed Galois; Matteo
 * Pontecorvi and Vijaya Ramachandran's BC 
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

constexpr static const char* const REGION_NAME = "PR_BC";

/******************************************************************************/
/* Sync code/calls was manually written, not compiler generated */
/******************************************************************************/

#include <iostream>
#include <limits>

#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;

static cll::opt<unsigned int> numSourcesPerRound("numRoundSources", 
                                cll::desc("Number of sources to use for APSP"),
                                cll::init(10));
static cll::opt<unsigned int> totalNumSources("numOfSources", 
                                cll::desc("Total number of sources to do BC"),
                                cll::init(0));
static cll::opt<unsigned int> vIndex("index", 
                                cll::desc("DEBUG: Index to print"),
                                cll::init(0),
                                cll::Hidden);

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/

const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;

// NOTE: types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  std::vector<uint32_t> oldMinDistances;
  std::vector<uint32_t> minDistances;
  std::vector<char> sentFlag;

  std::vector<uint32_t> shortestPathToAdd;
  std::vector<uint32_t> shortestPathNumbers;

  uint32_t APSPIndexToSend;
  uint32_t shortPathValueToSend;

  std::vector<uint32_t> savedRoundNumbers;
  std::atomic<float>* dependencyValues;

  float bc;
};

// Dist Graph using a bidirectional CSR graph + abstract locks
using Graph = galois::graphs::DistGraph<NodeData, void, true>;
using GNode = typename Graph::GraphNode;

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/
uint64_t offset = 0;
uint32_t roundNumber = 0;

void InitializeGraph(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode src) {
      NodeData& src_data = graph.getData(src);
  
      src_data.oldMinDistances.resize(numSourcesPerRound);
      src_data.minDistances.resize(numSourcesPerRound);
      src_data.shortestPathToAdd.resize(numSourcesPerRound);
      src_data.shortestPathNumbers.resize(numSourcesPerRound);
  
      src_data.APSPIndexToSend = numSourcesPerRound + 1;
      src_data.shortPathValueToSend = 0;
  
      src_data.savedRoundNumbers.resize(numSourcesPerRound);
      src_data.sentFlag.resize(numSourcesPerRound);

      src_data.dependencyValues =
       (std::atomic<float>*)malloc(sizeof(std::atomic<float>) * 
                                   numSourcesPerRound);
      GALOIS_ASSERT(src_data.dependencyValues != nullptr);
  
      src_data.bc = 0.0;
    },
    galois::loopname(graph.get_run_identifier("InitializeGraph").c_str()), 
    galois::no_stats()
  );
}

/* This is used to reset node data when switching to a different 
 * source set */
void InitializeIteration(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode src) {
      NodeData& src_data = graph.getData(src);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        // min distance and short path count setup
        if ((offset + i) == graph.getGID(src)) {
          src_data.minDistances[i] = 0;
          src_data.shortestPathNumbers[i] = 1;
        } else {
          src_data.minDistances[i] = infinity;
          src_data.shortestPathNumbers[i] = 0;
        }

        src_data.shortestPathToAdd[i] = 0;
  
        src_data.APSPIndexToSend = numSourcesPerRound + 1;
        src_data.shortPathValueToSend = 0;
  
        src_data.savedRoundNumbers[i] = infinity;
        src_data.dependencyValues[i] = 0.0;
        src_data.oldMinDistances[i] = src_data.minDistances[i];

        src_data.sentFlag[i] = 0;
      }
    },
    galois::loopname(graph.get_run_identifier("InitializeIteration").c_str()), 
    galois::no_stats()
  );
};

/**
 * TODO
 */
void MetadataUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode src) {
      NodeData& src_data = graph.getData(src);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (src_data.oldMinDistances[i] != src_data.minDistances[i]) {
          src_data.shortestPathNumbers[i] = 0;
          src_data.sentFlag[i] = 0; // reset sent flag
          // also, this means shortPathToAdd needs to be set 
        }
      }
    },
    galois::loopname(graph.get_run_identifier("MetadataUpdate").c_str()),
    galois::no_stats()
  );
};

// TODO
// This has dumb overhead even if nothing to update; find a way to make it
// faster
/**
 * TODO
 */
void ShortPathUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();
  galois::do_all(
    galois::iterate(allNodes), 
    [&] (GNode src) {
      NodeData& src_data = graph.getData(src);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (src_data.shortestPathToAdd[i] > 0) {
          src_data.shortestPathNumbers[i] += src_data.shortestPathToAdd[i];
        }
        src_data.shortestPathToAdd[i] = 0;
      }
    },
    galois::loopname(graph.get_run_identifier("ShortPathUpdate").c_str()),
    galois::no_stats()
  );
};

//struct ShortPathSends {
//  Graph* graph;
//  ShortPathSends(Graph* _graph) : graph(_graph) { }
//
//  void static go(Graph& _graph) {
//    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
//    galois::do_all(
//      galois::iterate(nodesWithEdges), 
//      ShortPathSends(&_graph),
//      galois::loopname(_graph.get_run_identifier("ShortPathSends").c_str()),
//      galois::no_stats()
//    );
//
//    // TODO graph sync of short paths here
//    
//    // TODO residual update + reset
//    ShortPathUpdate::go(_graph);
//  }
//
//  void operator()(GNode src) const {
//    NodeData& src_data = graph->getData(src);
//
//    for (auto outEdge : graph->edges(src)) {
//      GNode dst = graph->getEdgeDst(outEdge);
//      auto& dnode = graph->getData(dst);
//
//      // at this point minDist should be synchronized across all hosts
//      if (src_data.shortPathValueToSend != 0) {
//        uint32_t indexToUse = src_data.sIndexToSend;
//
//        if ((src_data.oldMinDistances[indexToUse] + 1) == 
//             dnode.minDistances[indexToUse]) {
//          dnode.shortestPathToAdd[indexToUse] += src_data.shortPathValueToSend;
//        }
//      }
//    }
//  }
//};

/**
 * TODO
 */
void OldDistUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode src) {
      NodeData& src_data = graph.getData(src);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        src_data.oldMinDistances[i] = src_data.minDistances[i];
      }
    },
    galois::loopname(graph.get_run_identifier("OldDistUpdate").c_str()),
    galois::no_stats()
  );
};

/**
 * TODO
 */
struct DWrapper {
  uint32_t dist;
  uint32_t index;
  
  DWrapper(uint32_t _dist, uint32_t _index)
    : dist(_dist), index(_index) { }

  bool operator<(const DWrapper& b) const {
    return dist < b.dist;
  }
};

std::vector<DWrapper> wrapDistVector(const std::vector<uint32_t>& dVector) {
  std::vector<DWrapper> wrappedVector;
  wrappedVector.reserve(numSourcesPerRound);

  for (unsigned i = 0; i < numSourcesPerRound; i++) {
    wrappedVector.emplace_back(DWrapper(dVector[i], i));
  }

  return wrappedVector;
}

/**
 * TODO
 */
void APSP(Graph& graph, galois::DGAccumulator<uint32_t>& dga) {
  const auto& nodesWithEdges = graph.allNodesWithEdgesRange();

  do {
    dga.reset();
    galois::gPrint("Round ", roundNumber, "\n");
    graph.set_num_iter(roundNumber);

    galois::do_all(
      galois::iterate(nodesWithEdges), 
      [&] (GNode src) {
        NodeData& src_data = graph.getData(src);
        galois::StatTimer wrapTime(graph.get_run_identifier("WrapTime").c_str());
    
        //wrapTime.start();
        std::vector<DWrapper> toSort = wrapDistVector(src_data.oldMinDistances);
        //wrapTime.stop();
    
        // TODO can have timer here
        std::stable_sort(toSort.begin(), toSort.end());
    
        uint32_t indexToSend = numSourcesPerRound + 1;

        // true if a shortest path message should be sent
        bool shortFound = false; 
        // true if a message is flipped from not sent to sent
        bool sentMarked = false;

        src_data.shortPathValueToSend = 0;
        src_data.APSPIndexToSend = indexToSend;
    
        //galois::StatTimer findMessage(graph.get_run_identifier("FindMessage").c_str());
    
        //findMessage.start();
        // TODO I can optimize this loop
        for (unsigned i = 0; i < numSourcesPerRound; i++) {
          DWrapper& currentSource = toSort[i]; // safe
          uint32_t currentIndex = currentSource.index; // safe
          unsigned sumToConsider = i + currentSource.dist;
    
          // determine if we need to send out a message in this round
          if (!shortFound && sumToConsider == roundNumber) {
            // save round num
            src_data.savedRoundNumbers[currentIndex] = roundNumber; // safe
    
            // note that this index needs to have shortest path number sent out
            // by saving it to another vector
            src_data.shortPathValueToSend = src_data.shortestPathNumbers[currentIndex];
            indexToSend = currentIndex;
            src_data.APSPIndexToSend = indexToSend;
            src_data.sentFlag[currentIndex] = true;
    
            shortFound = true;
            sentMarked = true;
          } else if (sumToConsider > roundNumber) {
            // not going to be sending any short path message this round
            // TODO reason if it is possible to break
          }

          // if we haven't found a message to mark ready yet, mark one (since
          // we only terminate if all things are marked ready)
          if (!sentMarked) {
            if (!src_data.sentFlag[currentIndex]) {
              src_data.sentFlag[currentIndex] = true;
              sentMarked = true;
            }
          }

          // TODO if we have marked a message ready, is it safe to bail and 
          // break? (i.e. will there be a short path message to send?)
          if (sentMarked && shortFound) {
            break;
          }
        }

        // if ready message was found, this node should not terminate this
        // round
        if (sentMarked) {
          dga += 1;
        }
        //findMessage.stop();
      },
      galois::loopname(graph.get_run_identifier("APSP").c_str()),
      galois::steal()
    );

    const auto& allNodesWithEdgesIn = graph.allNodesWithEdgesRangeIn();

    galois::do_all(
      galois::iterate(allNodesWithEdgesIn),
      [&] (GNode dst) {
        auto& dnode = graph.getData(dst);

        for (auto inEdge : graph.in_edges(dst)) {
          NodeData& src_data = graph.getData(graph.getInEdgeDst(inEdge));
          uint32_t indexToSend = src_data.APSPIndexToSend;
          
          if (indexToSend != numSourcesPerRound + 1) {
            uint32_t distValue = src_data.oldMinDistances[indexToSend];
            uint32_t newValue = distValue + 1;
            uint32_t oldValue = galois::min(dnode.minDistances[indexToSend],
                                            newValue);
    
            if (oldValue > newValue) {
              // overwrite short path with this node's shortest path
              dnode.shortestPathToAdd[indexToSend] = src_data.shortPathValueToSend;
            } else if (oldValue == newValue) {
              // add to short path
              dnode.shortestPathToAdd[indexToSend] += src_data.shortPathValueToSend;
            }

            dga += 1;
          }
        }
      },
      galois::loopname(graph.get_run_identifier("APSP").c_str()),
      galois::steal()
    );

    MetadataUpdate(graph);
    ShortPathUpdate(graph);
    OldDistUpdate(graph);

    //ShortPathSends::go(graph);

    roundNumber++;
  } while (dga.reduce());
}

/**
 * TODO
 */
void RoundUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();
  graph.set_num_iter(0);

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode src) {
      NodeData& src_data = graph.getData(src);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (src_data.oldMinDistances[i] < infinity) {
          src_data.savedRoundNumbers[i] = 
              roundNumber - src_data.savedRoundNumbers[i];
          assert(src_data.savedRoundNumbers[i] <= roundNumber);
        }
      }
    },
    galois::loopname(graph.get_run_identifier("RoundUpdate").c_str()),
    galois::no_stats()
  );
}

/**
 * TODO
 */
void BackProp(Graph& graph) {
  const auto& allNodesWithEdgesIn = graph.allNodesWithEdgesRangeIn();

  uint32_t currentRound = 0;

  while (currentRound <= roundNumber) {
    graph.set_num_iter(currentRound);

    galois::do_all(
      galois::iterate(allNodesWithEdgesIn),
      [&] (GNode src) {

        NodeData& src_data = graph.getData(src);
    
        std::vector<uint32_t> toBackProp;
    
        for (unsigned i = 0; i < numSourcesPerRound; i++) {
          if (src_data.savedRoundNumbers[i] == currentRound) {
            toBackProp.emplace_back(i);
            break;
          }
        }
    
        assert(toBackProp.size() <= 1);
    
        for (auto i : toBackProp) {
          uint32_t myDistance = src_data.oldMinDistances[i];
    
          // calculate final dependency value
          src_data.dependencyValues[i] = src_data.dependencyValues[i] * 
                                         src_data.shortestPathNumbers[i];
    
          // get the value to add to predecessors
          float toAdd = ((float)1 + src_data.dependencyValues[i]) / 
                          src_data.shortestPathNumbers[i];
    
          for (auto inEdge : graph.in_edges(src)) {
            GNode dst = graph.getInEdgeDst(inEdge);
            auto& dnode = graph.getData(dst);
    
            // determine if this dnode is a predecessor
            if (myDistance == (dnode.oldMinDistances[i] + 1)) {
              // add to dependency of predecessor using our finalized one
              galois::atomicAdd(dnode.dependencyValues[i], toAdd);
            }
          }
        }
      },
      galois::loopname(graph.get_run_identifier("BackProp").c_str()),
      galois::no_stats()
    );

    currentRound++;
  }
}

/**
 * TODO
 */
void BC(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();
  graph.set_num_iter(0);

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode src) {
      NodeData& src_data = graph.getData(src);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        // exclude sources themselves from BC calculation
        if (graph.getGID(src) != (i + offset)) {
          src_data.bc += src_data.dependencyValues[i];
        }
      }
    },
    galois::loopname(graph.get_run_identifier("BC").c_str()),
    galois::no_stats()
  );
};

/******************************************************************************/
/* Sanity check */
/******************************************************************************/

// TODO

/******************************************************************************/
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name = "PR-Betweeness Centrality"; 
constexpr static const char* const desc = "PR-Betweeness Centrality on Distributed "
                                          "Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  galois::StatTimer StatTimer_total("TIMER_TOTAL", REGION_NAME);
  StatTimer_total.start();

  Graph* hg = twoWayDistGraphInitialization<NodeData, void>();

  galois::gPrint("[", net.ID, "] InitializeGraph\n");

  galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT", REGION_NAME);

  StatTimer_graph_init.start();
  InitializeGraph(*hg);
  StatTimer_graph_init.stop();

  galois::runtime::getHostBarrier().wait();

  // shared DG accumulator among all steps
  galois::DGAccumulator<uint32_t> dga;

  offset = 0;

  if (totalNumSources == 0) {
    galois::gDebug("Total num sources unspecified");
    totalNumSources = hg->globalSize();
  }

  galois::StatTimer sortTimer("SortTimer", REGION_NAME);
  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] Run ", run, " started\n");
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();

    // TODO declare round number here and pass it in instead of having it be a
    // global

    while (offset < totalNumSources) {
      galois::gDebug("[", net.ID, "] Offset ", offset, " started");

      // correct in case numsources overflows total num of sources
      if (offset + numSourcesPerRound > totalNumSources) {
        numSourcesPerRound = totalNumSources - offset;
        galois::gDebug("Num sources for this final round will be ", 
                       numSourcesPerRound);
      }

      InitializeIteration(*hg);
      APSP(*hg, dga);

      roundNumber--; // subtract to get to terminating round; i.e. last round 

      RoundUpdate(*hg);
      BackProp(*hg);
      BC(*hg);

      offset += numSourcesPerRound;
      roundNumber = 0;
    }

    StatTimer_main.stop();

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      galois::runtime::getHostBarrier().wait();
      (*hg).set_num_run(run + 1);
      (*hg).set_num_iter(0);
      offset = 0;
      roundNumber = 0;

      InitializeGraph(*hg);
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  //galois::gPrint("total sort time ", sortTimer.get(), "\n");

  // Verify, i.e. print out graph data for examination
  if (verify) {
    char *v_out = (char*)malloc(40);
    for (auto ii = (*hg).masterNodesRange().begin(); 
              ii != (*hg).masterNodesRange().end(); 
              ++ii) {
      //sprintf(v_out, "%lu %u %u\n", (*hg).getGID(*ii),
      //        (*hg).getData(*ii).minDistances[vIndex],
      //        (*hg).getData(*ii).shortestPathNumbers[vIndex]);
      // outputs betweenness centrality
      sprintf(v_out, "%lu %.9f\n", (*hg).getGID(*ii),
              (*hg).getData(*ii).bc);
      //sprintf(v_out, "%lu %u\n", (*hg).getGID(*ii),
      //        (*hg).getData(*ii).shortestPathNumbers[0]);

      galois::runtime::printOutput(v_out);
    }
    free(v_out);
  }

  return 0;
}
