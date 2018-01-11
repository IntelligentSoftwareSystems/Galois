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

galois::DynamicBitSet bitset_minDistances;
galois::DynamicBitSet bitset_shortestPathToAdd;

// Dist Graph using a bidirectional CSR graph
using Graph = galois::graphs::DistGraph<NodeData, void, true>;
using GNode = typename Graph::GraphNode;

#include "gen_sync.hh"

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/
uint64_t offset = 0;
uint32_t roundNumber = 0;

/**
 * Graph initialization
 */
void InitializeGraph(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      cur_data.oldMinDistances.resize(numSourcesPerRound);
      cur_data.minDistances.resize(numSourcesPerRound);
      cur_data.shortestPathToAdd.resize(numSourcesPerRound);
      cur_data.shortestPathNumbers.resize(numSourcesPerRound);
  
      cur_data.APSPIndexToSend = numSourcesPerRound + 1;
      cur_data.shortPathValueToSend = 0;
  
      cur_data.savedRoundNumbers.resize(numSourcesPerRound);
      cur_data.sentFlag.resize(numSourcesPerRound);

      cur_data.dependencyValues =
       (std::atomic<float>*)malloc(sizeof(std::atomic<float>) * 
                                   numSourcesPerRound);
      GALOIS_ASSERT(cur_data.dependencyValues != nullptr);
  
      cur_data.bc = 0.0;
    },
    galois::loopname(graph.get_run_identifier("InitializeGraph").c_str()), 
    galois::no_stats()
  );
}


/** 
 * This is used to reset node data when switching to a different 
 * source set 
 **/
void InitializeIteration(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        // min distance and short path count setup
        if ((offset + i) == graph.getGID(curNode)) {
          cur_data.minDistances[i] = 0;
          cur_data.shortestPathNumbers[i] = 1;
        } else {
          cur_data.minDistances[i] = infinity;
          cur_data.shortestPathNumbers[i] = 0;
        }

        cur_data.shortestPathToAdd[i] = 0;
  
        cur_data.APSPIndexToSend = numSourcesPerRound + 1;
        cur_data.shortPathValueToSend = 0;
  
        cur_data.savedRoundNumbers[i] = infinity;
        cur_data.dependencyValues[i] = 0.0;
        cur_data.oldMinDistances[i] = cur_data.minDistances[i];

        cur_data.sentFlag[i] = 0;
      }
    },
    galois::loopname(graph.get_run_identifier("InitializeIteration").c_str()), 
    galois::no_stats()
  );
};

/**
 * If min distance has changed, then the number of shortest paths is reset
 * and the sent flag is reset.
 */
void MetadataUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (cur_data.oldMinDistances[i] != cur_data.minDistances[i]) {
          cur_data.shortestPathNumbers[i] = 0;
          cur_data.sentFlag[i] = 0; // reset sent flag
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
 * Adds the accumulated shortest path values to the actual shortest path
 * value and resets the accumulator.
 */
void ShortPathUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();
  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (cur_data.shortestPathToAdd[i] > 0) {
          cur_data.shortestPathNumbers[i] += cur_data.shortestPathToAdd[i];
        }
        cur_data.shortestPathToAdd[i] = 0;
      }
    },
    galois::loopname(graph.get_run_identifier("ShortPathUpdate").c_str()),
    galois::no_stats()
  );
};

/**
 * Update old dist with the latest value of the min distance.
 */
void OldDistUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        cur_data.oldMinDistances[i] = cur_data.minDistances[i];
      }
    },
    galois::loopname(graph.get_run_identifier("OldDistUpdate").c_str()),
    galois::no_stats()
  );
};

/**
 * Wrapper around the distance array for sorting purposes.
 *
 * TODO find a way to make this more efficient
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

/**
 * Does the wrapping of a vector into a DWrapper for sorting.
 */
std::vector<DWrapper> wrapDistVector(const std::vector<uint32_t>& dVector) {
  std::vector<DWrapper> wrappedVector;
  wrappedVector.reserve(numSourcesPerRound);

  for (unsigned i = 0; i < numSourcesPerRound; i++) {
    wrappedVector.emplace_back(DWrapper(dVector[i], i));
  }

  return wrappedVector;
}

/**
 * Find all pairs shortest paths for the sources currently being worked on
 * as well as the number of shortest paths for each source.
 */
void APSP(Graph& graph, galois::DGAccumulator<uint32_t>& dga) {
  //const auto& nodesWithEdges = graph.allNodesWithEdgesRange();
  const auto& allNodes = graph.allNodesRange();

  do {
    dga.reset();
    galois::gPrint("Round ", roundNumber, "\n");
    graph.set_num_iter(roundNumber);

    galois::do_all(
      galois::iterate(allNodes), 
      [&] (GNode curNode) {
        NodeData& cur_data = graph.getData(curNode);
        galois::StatTimer wrapTime(graph.get_run_identifier("WrapTime").c_str());
    
        //wrapTime.start();
        std::vector<DWrapper> toSort = wrapDistVector(cur_data.oldMinDistances);
        //wrapTime.stop();
    
        // TODO can have timer here
        std::stable_sort(toSort.begin(), toSort.end());
    
        uint32_t indexToSend = numSourcesPerRound + 1;

        // true if a shortest path message should be sent
        bool shortFound = false; 
        // true if a message is flipped from not sent to sent
        bool sentMarked = false;

        cur_data.shortPathValueToSend = 0;
        cur_data.APSPIndexToSend = indexToSend;
    
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
            cur_data.savedRoundNumbers[currentIndex] = roundNumber; // safe
    
            // note that this index needs to have shortest path number sent out
            // by saving it to another vector
            cur_data.shortPathValueToSend = cur_data.shortestPathNumbers[currentIndex];
            indexToSend = currentIndex;
            cur_data.APSPIndexToSend = indexToSend;
            cur_data.sentFlag[currentIndex] = true;
    
            shortFound = true;
            sentMarked = true;
          } else if (sumToConsider > roundNumber) {
            // not going to be sending any short path message this round
            // TODO reason if it is possible to break
          }

          // if we haven't found a message to mark ready yet, mark one (since
          // we only terminate if all things are marked ready)
          if (!sentMarked) {
            if (!cur_data.sentFlag[currentIndex]) {
              cur_data.sentFlag[currentIndex] = true;
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
            bitset_minDistances.set(dst);
            bitset_shortestPathToAdd.set(dst);
          }
        }
      },
      galois::loopname(graph.get_run_identifier("APSP").c_str()),
      galois::steal()
    );

    // sync min distance (also resets short path add if necessary)
    graph.sync<writeDestination, readAny, ReducePairwiseMinAndResetDist, 
               Broadcast_minDistances, Bitset_minDistances>("MinDistSync");

    MetadataUpdate(graph); 

    // sync short path add
    graph.sync<writeDestination, readAny, 
               Reduce_pair_wise_add_array_shortestPathToAdd, 
               Broadcast_shortestPathToAdd, 
               Bitset_shortestPathToAdd>("ShortPathSync");

    ShortPathUpdate(graph);

    OldDistUpdate(graph);

    roundNumber++;
  } while (dga.reduce());
}

/**
 * Get the round number for the backward propagation phase using the round
 * number from the APSP phase.
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
 * Back propagate dependency values depending on the round that a node
 * sent out the shortest path message.
 */
void BackProp(Graph& graph) {
  const auto& allNodesWithEdgesIn = graph.allNodesWithEdgesRangeIn();

  uint32_t currentRound = 0;

  while (currentRound <= roundNumber) {
    graph.set_num_iter(currentRound);

    galois::do_all(
      galois::iterate(allNodesWithEdgesIn),
      [&] (GNode dst) {
        NodeData& dst_data = graph.getData(dst);
    
        std::vector<uint32_t> toBackProp;

        for (unsigned i = 0; i < numSourcesPerRound; i++) {
          if (dst_data.savedRoundNumbers[i] == currentRound) {
            toBackProp.emplace_back(i);
            break;
          }
        }
    
        assert(toBackProp.size() <= 1);
    
        for (auto i : toBackProp) {
          uint32_t myDistance = dst_data.oldMinDistances[i];
    
          // calculate final dependency value
          dst_data.dependencyValues[i] = dst_data.dependencyValues[i] * 
                                         dst_data.shortestPathNumbers[i];
    
          // get the value to add to predecessors
          float toAdd = ((float)1 + dst_data.dependencyValues[i]) / 
                          dst_data.shortestPathNumbers[i];
    
          for (auto inEdge : graph.in_edges(dst)) {
            GNode src = graph.getInEdgeDst(inEdge);
            auto& src_data = graph.getData(src);
    
            // determine if this source is a predecessor
            if (myDistance == (src_data.oldMinDistances[i] + 1)) {
              // add to dependency of predecessor using our finalized one
              galois::atomicAdd(src_data.dependencyValues[i], toAdd);
            }
          }
        }
      },
      galois::loopname(graph.get_run_identifier("BackProp").c_str()),
      galois::no_stats()
    );

    // dependency written to source
    // dep needs to be on dst nodes, but final round needs them on all nodes
    if (currentRound != roundNumber) {
      // sync to only dests

    } else {
      // sync all

    }
    currentRound++;
  }
}

/**
 * BC sum: take the dependency value for each source and add it to the
 * final BC value.
 */
void BC(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();
  graph.set_num_iter(0);

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode node) {
      NodeData& cur_data = graph.getData(node);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        // exclude sources themselves from BC calculation
        if (graph.getGID(node) != (i + offset)) {
          cur_data.bc += cur_data.dependencyValues[i];
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

  bitset_minDistances.resize(hg->size());
  bitset_shortestPathToAdd.resize(hg->size());

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

      galois::runtime::getHostBarrier().wait();

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

      bitset_minDistances.reset();

      InitializeGraph(*hg);
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  // Verify, i.e. print out graph data for examination
  if (verify) {
    char *v_out = (char*)malloc(40);
    for (auto ii = (*hg).masterNodesRange().begin(); 
              ii != (*hg).masterNodesRange().end(); 
              ++ii) {
      // outputs min distance and short path numbers
      //sprintf(v_out, "%lu %u %u\n", (*hg).getGID(*ii),
      //        (*hg).getData(*ii).minDistances[vIndex],
      //        (*hg).getData(*ii).shortestPathNumbers[vIndex]);

      // outputs betweenness centrality
      sprintf(v_out, "%lu %.9f\n", (*hg).getGID(*ii),
              (*hg).getData(*ii).bc);

      galois::runtime::printOutput(v_out);
    }
    free(v_out);
  }

  return 0;
}
