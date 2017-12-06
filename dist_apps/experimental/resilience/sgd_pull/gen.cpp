/** SGD -*- C++ -*-
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
 * Compute SGD (matrix completion) on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#include <iostream>
#include <limits>
#include <cmath>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"

#include "galois/DReducible.h"
#include "galois/AtomicWrapper.h"
#include "galois/ArrayWrapper.h"
#include "galois/runtime/Tracer.h"

#include "galois/graphs/DistributedGraphLoader.h"

//For resilience
#include "resilience.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/cuda/cuda_device.h"
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;
#endif

constexpr static const char* const regionname = "SGD";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

namespace cll = llvm::cl;

static cll::opt<unsigned int> maxIterations("maxIterations", 
    cll::desc("Maximum iterations: Default 10000"), 
    cll::init(10000));
static cll::opt<bool> bipartite("bipartite", 
    cll::desc("Is graph bipartite? if yes, it expects first N nodes to have "
              "edges."), 
    cll::init(false));
static cll::opt<double> LEARNING_RATE("LEARNING_RATE", 
    cll::desc("Learning rate (GAMMA): Default 0.00001"), 
    cll::init(0.00001));
static cll::opt<double> LAMBDA("LAMBDA",
    cll::desc("LAMBDA: Default 0.0001"),
    cll::init(0.0001));
static cll::opt<double> DECAY_RATE("DECAY_RATE",
    cll::desc("Decay rate to be used in step size function (DECAY_RATE): Default 0.9"),
    cll::init(0.9));
/******************************************************************************/
/* Graph structure declarations + helper functions + other initialization */
/******************************************************************************/

#define LATENT_VECTOR_SIZE 20
//static const double LEARNING_RATE = 0.00001; // GAMMA, Purdue: 0.01 Intel: 0.001
//static const double DECAY_RATE = 0.9; // STEP_DEC, Purdue: 0.1 Intel: 0.9
//static const double LAMBDA = 0.0001; // Purdue: 1.0 Intel: 0.001
static const double MINVAL = -1e+100;
static const double MAXVAL = 1e+100;

const unsigned int infinity = std::numeric_limits<unsigned int>::max() / 4;

struct NodeData {

  galois::CopyableArray<galois::CopyableAtomic<double>, LATENT_VECTOR_SIZE> residual_latent_vector;
  galois::CopyableArray<double, LATENT_VECTOR_SIZE> latent_vector;
};


typedef hGraph<NodeData, double> Graph;
//typedef hGraph<NodeData, uint32_t> Graph;
typedef typename Graph::GraphNode GNode;

#include "gen_sync.hh"
//TODO: Set seed
static double genRand() {
  // generate a random double in (-1,1)
  return 2.0 * ((double)std::rand() / (double)RAND_MAX) - 1.0;
}

// Purdue learning function
double getstep_size(unsigned int round) {
  return LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(round + 1, 1.5));
}

/**
 * Prediction of edge weight based on 2 latent vectors
 */
double calcPrediction (const NodeData& movie_data, const NodeData& user_data) {
  double pred = galois::innerProduct(movie_data.latent_vector, 
                                     user_data.latent_vector, 
                                     0.0); 
  double p = pred;

  pred = std::min(MAXVAL, pred);
  pred = std::max(MINVAL, pred);

  #ifndef NDEBUG
  if (p != pred)
    std::cerr << "clamped " << p << " to " << pred << "\n";
  #endif

  return pred;
}

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();

    #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str(
        _graph.get_run_identifier("CUDA_DO_ALL_IMPL_InitializeGraph")
      );
      galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      InitializeGraph_cuda(*allNodes.begin(), *allNodes.end(), cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
    #endif
    galois::do_all(galois::iterate(allNodes.begin(), allNodes.end()), 
                   InitializeGraph {&_graph}, galois::loopname("Init"));

    // due to latent_vector being generated randomly, it should be sync'd
    // to 1 consistent version across all hosts
    _graph.sync<writeSource, readAny, Reduce_set_latent_vector,
                Broadcast_latent_vector>("InitializeGraph");
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);

    for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
      sdata.latent_vector[i] = genRand(); // randomly create latent vector 
      sdata.residual_latent_vector[i] = 0 ; // randomly create latent vector 

      #ifndef NDEBUG
      if(!std::isnormal(sdata.latent_vector[i]))
        galois::gDebug("GEN for ", i, " ",  sdata.latent_vector[i]);
      #endif
    }
  }
};

/* Recovery to be called by resilience based fault tolerance
 */
struct recovery {
  Graph * graph;

  recovery(Graph * _graph) : graph(_graph) {}

  void static go(Graph& _graph) {

    const auto& allNodes = _graph.allNodesRange();
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      recovery{&_graph},
      galois::no_stats(),
      galois::loopname(_graph.get_run_identifier("RECOVERY").c_str()));

    //_graph.sync<writeDestination, readSource, Reduce_min_dist_current, 
                //Broadcast_dist_current, Bitset_dist_current>("RECOVERY");
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

  }
};

struct SGD_mergeResidual {
  Graph* graph;

  SGD_mergeResidual(Graph* _graph) :
      graph(_graph){}

  void static go(Graph& _graph) {

    auto& allNodes = _graph.allNodesRange();

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_SGD_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      int __retval = 0;
      SGD_all_cuda(__retval, cuda_ctx);
      //DGAccumulator_accum += __retval;
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif

      galois::do_all(
          galois::iterate(allNodes.begin(), allNodes.end()),
          SGD_mergeResidual { &_graph },
          galois::loopname(_graph.get_run_identifier("SGD_merge").c_str()),
          galois::steal(),
          galois::no_stats());
  }

  void operator()(GNode src) const {
    NodeData& sdata= graph->getData(src);
    auto& latent_vector = sdata.latent_vector;
    auto& residual_latent_vector = sdata.residual_latent_vector;

    for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
      latent_vector[i] += residual_latent_vector[i];
      residual_latent_vector[i] = 0;

      #ifndef NDEBUG
      if(!std::isnormal(sdata.latent_vector[i]))
        galois::gDebug("GEN for ", i, " ",  sdata.latent_vector[i]);
      #endif
    }
  }
};

struct SGD {
  Graph* graph;
  double step_size;
  galois::DGAccumulator<double>& DGAccumulator_accum;

  SGD(Graph* _graph, double _step_size, galois::DGAccumulator<double>& _dga) : 
      graph(_graph), step_size(_step_size), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<double>& dga) {
    unsigned _num_iterations = 0;
    double rms_normalized = 0.0;
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    do {
      std::cerr << "ITERATION : " << _num_iterations << "\n";
      //Checkpointing the all the node data
      saveCheckpointToDisk(_num_iterations, _graph);

      auto step_size = getstep_size(_num_iterations);
      dga.reset();
      galois::do_all(
          galois::iterate(nodesWithEdges),
          SGD( &_graph, step_size, dga),
          galois::loopname(_graph.get_run_identifier("SGD").c_str()),
          galois::steal(),
          galois::no_stats());

    // sync all latent vectors
    //_graph.sync<writeAny, readAny, Reduce_pair_wise_avg_array_residual_latent_vector,
                //Broadcast_residual_latent_vector>("SGD");

    _graph.sync<writeAny, readAny, Reduce_pair_wise_add_array_residual_latent_vector,
                Broadcast_residual_latent_vector>("SGD");

      SGD_mergeResidual::go(_graph);

      /**************************CRASH SITE : start *****************************************/
      if(enableFT && (_num_iterations == crashIteration)){
        crashSite<recovery>(_graph);
      }
      /**************************CRASH SITE : end *****************************************/

      ++_num_iterations;

      // calculate root mean squared error
      rms_normalized = std::sqrt(dga.reduce() /
                                 _graph.globalSizeEdges());
      galois::gDebug("RMS Normalized : ", rms_normalized);
      galois::gPrint("RMS : ", rms_normalized, "\n");
    } while((_num_iterations < maxIterations) && (rms_normalized > 0.1));

  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    auto& movie_node = sdata.latent_vector;
    auto& residual_movie_node = sdata.residual_latent_vector;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src);
         jj != ej;
         ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& ddata = graph->getData(dst);

      auto& user_node = ddata.latent_vector;
      auto& residual_user_node = ddata.residual_latent_vector;
      //auto& sdata_up = sdata.updates;

      double edge_rating = graph->getEdgeData(dst);

      // doGradientUpdate
      double old_dp = galois::innerProduct(user_node, movie_node, double(0));

      double cur_error = edge_rating - old_dp;
      DGAccumulator_accum += (cur_error * cur_error);

      assert(cur_error < 10000 && cur_error > -10000);

      // update both vectors based on error derived from 2 previous vectors
      for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {

        double prevUser = user_node[i];
        double prevMovie = movie_node[i];

        galois::atomicAdd(residual_user_node[i],  double(step_size * (cur_error * prevMovie - LAMBDA * prevUser)));
        assert(std::isnormal(residual_user_node[i]));

        galois::atomicAdd(residual_movie_node[i],  double(step_size * (cur_error * prevUser - LAMBDA * prevMovie)));
        assert(std::isnormal(residual_movie_node[i]));
      }
    }
  }
};

/******************************************************************************/
/* Main */
/******************************************************************************/
constexpr static const char* const name = "SGD - Distributed Heterogeneous";
constexpr static const char* const desc = "SGD on Distributed Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  const auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.ID == 0) {
    galois::runtime::reportParam(regionname, "Max Iterations", 
        (unsigned long)maxIterations);
  }

  galois::StatTimer StatTimer_total("TIMER_TOTAL", regionname); 

  StatTimer_total.start();
#ifdef __GALOIS_HET_CUDA__
  Graph* hg = distGraphInitialization<NodeData, double>(&cuda_ctx);
#else
  Graph* hg = distGraphInitialization<NodeData, double>();
#endif

  // bitset comm setup
  //bitset_dist_current.resize(hg->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT", regionname); 
  StatTimer_init.start();
  InitializeGraph::go((*hg));
  StatTimer_init.stop();

  galois::runtime::getHostBarrier().wait();

  // accumulators for use in operators
  galois::DGAccumulator<double> DGAccumulator_accum;
  //galois::DGAccumulator<uint64_t> DGAccumulator_sum;
  //galois::DGAccumulator<uint32_t> DGAccumulator_max;
  //galois::GReduceMax<uint32_t> m;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] SGD::go run ", run, " called\n");
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), regionname);

    StatTimer_main.start();
    SGD::go((*hg), DGAccumulator_accum);
    StatTimer_main.stop();

    if ((run + 1) != numRuns) {
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) { 
        //bitset_dist_current_reset_cuda(cuda_ctx);
      } else
#endif

      (*hg).set_num_run(run+1);
      InitializeGraph::go((*hg));
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();


  return 0;
}
