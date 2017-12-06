/** KCore -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 * Compute KCore on distributed Galois using top-filter. The degree used is
 * the in-degree.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

/******************************************************************************/
/* Sync code/calls was manually written, not compiler generated */
/******************************************************************************/

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/runtime/dGraph_edgeCut.h"
#include "galois/runtime/dGraph_cartesianCut.h"
#include "galois/runtime/dGraph_hybridCut.h"

#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

#include "galois/runtime/dGraphLoader.h"

static const char* const name = "KCore - Distributed Heterogeneous Push Filter.";
static const char* const desc = "KCore on Distributed Galois.";
static const char* const url = 0;

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;
static cll::opt<unsigned int> maxIterations("maxIterations", 
                               cll::desc("Maximum iterations: Default 10000"), 
                               cll::init(10000));
static cll::opt<bool> verify("verify", 
                             cll::desc("Verify ranks by printing to "
                                       "'page_ranks.#hid.csv' file"),
                             cll::init(false));

// required k specification for k-core
static cll::opt<unsigned int> k_core_num("kcore",
                                     cll::desc("KCore value"),
                                     cll::Required);


/******************************************************************************/
/* Graph structure declarations + other inits */
/******************************************************************************/

struct NodeData {
  std::atomic<uint32_t> current_degree;
  uint8_t flag;
};

typedef DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/

/* Degree counting
 * Called by InitializeGraph1 */
struct InitializeGraph2 {
  Graph *graph;

  InitializeGraph2(Graph* _graph) : graph(_graph){}

  /* Initialize the entire graph node-by-node */
  void static go(Graph& _graph) {
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    galois::do_all(
      nodesWithEdges,
      InitializeGraph2{ &_graph },
      galois::no_stats(),
      galois::loopname(_graph.get_run_identifier("InitializeGraph2").c_str()),
      galois::steal());
  }

  /* Calculate degree of nodes by checking how many nodes have it as a dest and
   * adding for every dest */
  void operator()(GNode src) const {
    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src);
         current_edge != end_edge;
         current_edge++) {
      GNode dest_node = graph->getEdgeDst(current_edge);

      NodeData& dest_data = graph->getData(dest_node);
      galois::atomicAdd(dest_data.current_degree, (uint32_t)1);
    }
  }
};


/* Initialize: initial field setup */
struct InitializeGraph1 {
  Graph *graph;

  InitializeGraph1(Graph* _graph) : graph(_graph){}

  /* Initialize the entire graph node-by-node */
  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();

    galois::do_all(
      allNodes.begin(), allNodes.end(),
      InitializeGraph1{ &_graph },
      galois::no_stats(),
      galois::loopname(_graph.get_run_identifier("InitializeGraph1").c_str())
    );

    // degree calculation
    InitializeGraph2::go(_graph);
  }

  /* Setup intial fields */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    src_data.flag = true;
    src_data.current_degree = 0;
  }
};


struct KCoreStep1 {
  cll::opt<uint32_t>& local_k_core_num;
  Graph* graph;

  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  KCoreStep1(cll::opt<uint32_t>& _kcore, Graph* _graph,
             galois::DGAccumulator<unsigned int>& _dga) : 
    local_k_core_num(_kcore), graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned iterations = 0;
    
    auto& allNodes = _graph.allNodesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      galois::do_all(
        allNodes,
        KCoreStep1{ k_core_num, &_graph, dga },
        galois::no_stats(),
        galois::loopname(_graph.get_run_identifier("KCoreStep1").c_str()),
        galois::steal());

      iterations++;
    } while ((iterations < maxIterations) && dga.reduce());

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat("(NULL)", 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)iterations, 0);
    }

  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // only if node is alive we do things
    if (src_data.flag) {
      if (src_data.current_degree < local_k_core_num) {
        // set flag to 0 (false) and increment trim on outgoing neighbors
        // (if they exist)
        src_data.flag = false;
        DGAccumulator_accum += 1; // can be optimized: node may not have edges

        for (auto current_edge = graph->edge_begin(src), 
                  end_edge = graph->edge_end(src);
             current_edge != end_edge; 
             ++current_edge) {
           GNode dst = graph->getEdgeDst(current_edge);
           auto& dst_data = graph->getData(dst);

           dst_data.current_degree -= 1;
        }
      }
    }
  }
};

/******************************************************************************/
/* Main method for running */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    galois::DistMemSys G(getStatsFile());
    DistBenchStart(argc, argv, name, desc, url);

    {
    auto& net = galois::runtime::getSystemNetworkInterface();
    if (net.ID == 0) {
      galois::runtime::reportStat("(NULL)", "Max Iterations", 
                                  (unsigned long)maxIterations, 0);
    }

    galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT"),
                      StatTimer_total("TIMER_TOTAL"),
                      StatTimer_hg_init("TIMER_HG_INIT");

    StatTimer_total.start();

    std::vector<unsigned> scalefactor;

    StatTimer_hg_init.start();

    Graph* h_graph = nullptr;

    if (inputFileSymmetric) {
      h_graph = constructSymmetricGraph<NodeData, void>(scalefactor);
    } else {
      GALOIS_DIE("must pass symmetricGraph flag with symmetric graph to "
                 "kcore");
    }

    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go functions called\n";
    StatTimer_graph_init.start();
      InitializeGraph1::go((*h_graph));
    StatTimer_graph_init.stop();

    galois::DGAccumulator<unsigned int> DGAccumulator_accum;
    galois::DGAccumulator<uint64_t> dga1;
    galois::DGAccumulator<uint64_t> dga2;

    for (auto run = 0; run < numRuns; ++run) {
      std::cout << "[" << net.ID << "] KCoreStep1::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        KCoreStep1::go(*h_graph, DGAccumulator_accum);
      StatTimer_main.stop();

      // sanity check
      GetAliveDead::go(*h_graph, dga1, dga2);

      // re-init graph for next run
      if ((run + 1) != numRuns) {
        galois::runtime::getHostBarrier().wait();
        (*h_graph).reset_num_iter(run+1);

        InitializeGraph1::go((*h_graph));
      }
    }

    StatTimer_total.stop();

    // Verify, i.e. print out graph data for examination
    if (verify) {
      for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
        if ((*h_graph).isOwned((*h_graph).getGID(*ii))) 
          // prints the flag (alive/dead)
          galois::runtime::printOutput("% %\n", (*h_graph).getGID(*ii), 
                                       (bool)(*h_graph).getData(*ii).flag);


        // does a sanity check as well: 
        // degree higher than kcore if node is alive
        if (!((*h_graph).getData(*ii).flag)) {
          assert((*h_graph).getData(*ii).current_degree < k_core_num);
        } 
      }
    }
    galois::runtime::getHostBarrier().wait();

    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
