/** Common file for adding resilience -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * Common functions for adding resilience to applications
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#include <vector>
#include <algorithm>
#include <set>
#include <numeric>

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

namespace cll = llvm::cl;

enum RECOVERY_SCHEME {CP, RS};

static cll::opt<bool> enableFT("enableFT",
                                            cll::desc("Enable fault tolerance: "
                                                      "Default false"),
                                            cll::init(false));
static cll::opt<RECOVERY_SCHEME> recoveryScheme("recoveryScheme",
                                            cll::desc("Use CP for checkpointing"
                                                      "and RS for resilience"
                                                      "Default CP"),
                                            cll::init(CP));
static cll::opt<unsigned int> checkpointInterval("checkpointInterval",
                                            cll::desc("Interval of taking checkpoints"
                                                       "to disk. N means checkpoint graph"
                                                       "data every Nth iteration: "
                                                      "Default 1"),
                                            cll::init(1));
static cll::opt<unsigned int> crashIteration("crashIteration",
                                            cll::desc("iteration to crash: "
                                                      "Default -1"),
                                            cll::init(-1));

static cll::opt<unsigned int> crashNumHosts("crashNumHosts",
                                            cll::desc("number of hosts to crash: "
                                                      "Default -1"),
                                            cll::init(-1));





std::set<uint32_t> getRandomHosts(){
  const auto& net = galois::runtime::getSystemNetworkInterface();
  uint32_t numHosts = net.Num;

  std::vector<uint32_t> allHosts(numHosts);
  std::iota(allHosts.begin(), allHosts.end(), 0);

  //Seed is used by random_shuffle.
  std::srand(1);
  std::random_shuffle(allHosts.begin(), allHosts.end());

  return std::set<uint32_t>(allHosts.begin(), allHosts.begin() + crashNumHosts);
}

/*
 * Checkpointing the all the node data
 */
template<typename GraphTy>
void saveCheckpointToDisk(unsigned _num_iterations, GraphTy& _graph){
  if(enableFT && recoveryScheme == CP){
    if(_num_iterations%checkpointInterval == 0){
      _graph.checkpointSaveNodeData();
    }
  }
}

template<typename RecoveryTy, typename GraphTy>
void crashSite(GraphTy& _graph){
  const auto& net = galois::runtime::getSystemNetworkInterface();
  std::set<uint32_t> crashHostSet = getRandomHosts();

  //Use resilience to recover
  if(recoveryScheme == RS){
    galois::gPrint(net.ID, " :  Using RS\n");
    // Crashed hosts need to reconstruct local graphs
    if(crashHostSet.find(net.ID) != crashHostSet.end()){
      galois::gPrint(net.ID, " : CRASHED!!!\n");

      //Reconstruct local graph
      _graph.read_local_graph_from_file(localGraphFileName);
      RecoveryTy::go(_graph);

    } else {
      RecoveryTy::go(_graph);
    }
  } else if (recoveryScheme == CP){
    galois::gPrint(net.ID, " :  Using CP\n");
    // Crashed hosts need to reconstruct local graphs
    if(crashHostSet.find(net.ID) != crashHostSet.end()){
      galois::gPrint(net.ID, " : CRASHED!!!\n");

      //Reconstruct local graph
      //Assumes that local graph file is present
      _graph.read_local_graph_from_file(localGraphFileName);
      _graph.checkpointApplyNodeData();
    } else {
      _graph.checkpointApplyNodeData();
    }
  }
}
