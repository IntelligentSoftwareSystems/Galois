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
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/runtime/dGraph_edgeCut.h"
#include "galois/runtime/dGraph_cartesianCut.h"
#include "galois/runtime/dGraph_hybridCut.h"

#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

#include "galois/runtime/dGraphLoader.h"

#ifdef __GALOIS_HET_CUDA__
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;

enum Personality {
   CPU, GPU_CUDA, GPU_OPENCL
};

std::string personality_str(Personality p) {
   switch (p) {
   case CPU:
      return "CPU";
   case GPU_CUDA:
      return "GPU_CUDA";
   case GPU_OPENCL:
      return "GPU_OPENCL";
   }
   assert(false && "Invalid personality");
   return "";
}
#endif

static const char* const name = "SGD - Distributed Heterogeneous";
static const char* const desc = "SGD on Distributed Galois.";
static const char* const url = 0;

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

namespace cll = llvm::cl;

static cll::opt<unsigned int> maxIterations("maxIterations", 
    cll::desc("Maximum iterations: Default 10000"), 
    cll::init(10000));
static cll::opt<unsigned int> src_node("srcNodeId", 
    cll::desc("ID of the source node"), 
    cll::init(0));
static cll::opt<bool> verify("verify", 
    cll::desc("Verify ranks by printing to file"), 
    cll::init(false));
static cll::opt<bool> bipartite("bipartite", 
    cll::desc("Is graph bipartite? if yes, it expects first N nodes to have "
              "edges."), 
    cll::init(true));

#ifdef __GALOIS_HET_CUDA__
static cll::opt<int> gpudevice("gpu", 
                                cll::desc("Select GPU to run on, "
                                          "default is to choose automatically"), 
                                cll::init(-1));
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), 
                  clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), 
                  clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), 
                  clEnumValEnd),
      cll::init(CPU));
static cll::opt<unsigned> scalegpu("scalegpu", 
      cll::desc("Scale GPU workload w.r.t. CPU, default is proportionally "
                "equal workload to CPU and GPU (1)"), 
      cll::init(1));
static cll::opt<unsigned> scalecpu("scalecpu", 
      cll::desc("Scale CPU workload w.r.t. GPU, default is proportionally "
                "equal workload to CPU and GPU (1)"), 
      cll::init(1));
static cll::opt<int> num_nodes("num_nodes", 
      cll::desc("Num of physical nodes with devices (default = num of hosts): " 
                "detect GPU to use for each host automatically"), 
      cll::init(-1));
static cll::opt<std::string> personality_set("pset", 
      cll::desc("String specifying personality for hosts on each physical "
                "node. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), 
      cll::init("c"));
#endif

/******************************************************************************/
/* Graph structure declarations + helper functions + other initialization */
/******************************************************************************/

#define LATENT_VECTOR_SIZE 20
static const double LEARNING_RATE = 0.001; // GAMMA, Purdue: 0.01 Intel: 0.001
static const double DECAY_RATE = 0.9; // STEP_DEC, Purdue: 0.1 Intel: 0.9
static const double LAMBDA = 0.001; // Purdue: 1.0 Intel: 0.001
static const double MINVAL = -1e+100;
static const double MAXVAL = 1e+100;

const unsigned int infinity = std::numeric_limits<unsigned int>::max() / 4;
unsigned iteration;

struct NodeData {
  std::array<double, LATENT_VECTOR_SIZE> latent_vector;
  unsigned int updates;
  unsigned int edge_offset;
};

#include "gen_sync.hh"

typedef hGraph<NodeData, double> Graph;
typedef typename Graph::GraphNode GNode;

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
    galois::do_all(allNodes.begin(), allNodes.end(), 
                   InitializeGraph {&_graph}, galois::loopname("Init"));

    // due to latent_vector being generated randomly, it should be sync'd
    // to 1 consistent version across all hosts
    _graph.sync<writeSource, readAny, Reduce_set_latent_vector, 
                Broadcast_latent_vector>("InitializeGraph");
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.updates = 0;
    sdata.edge_offset = 0;

    for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
      sdata.latent_vector[i] = genRand(); // randomly create latent vector 

      #ifndef NDEBUG
      if(!std::isnormal(sdata.latent_vector[i]))
        galois::gDebug("GEN for ", i, " ",  sdata.latent_vector[i]);
      #endif
    }
  }
};

struct SGD_doPartialGradientUpdate {
  Graph* graph;
  double step_size;

  SGD_doPartialGradientUpdate(Graph* _graph, double _step_size) : 
      graph(_graph), step_size(_step_size) {}

  void static go(Graph& _graph, double _step_size) {
    // TODO GPU hook?
    // galois::do_all(_graph.begin(), _graph.end(), SGD_doPartialGradientUpdate { &_graph, _step_size }, galois::loopname("SGD_doPartialGradientUpdate"));
    galois::for_each(_graph.begin(), _graph.end(), 
                     SGD_doPartialGradientUpdate(&_graph, _step_size));

    // sync all latent vectors
    _graph.sync<writeAny, readAny, Reduce_pair_wise_avg_array_latent_vector, 
                Broadcast_latent_vector>("SGD");
  }

  template<typename Context>
  void operator()(GNode src, Context& cnx) const {
  //void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    auto& movie_node = sdata.latent_vector;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src);
         jj != ej; 
         ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& ddata = graph->getData(dst);

      auto& user_node = ddata.latent_vector;
      auto& sdata_up = sdata.updates;

      double edge_rating = graph->getEdgeData(dst);

      // doGradientUpdate
      double old_dp = galois::innerProduct(user_node.begin(), user_node.end(), 
                                           movie_node.begin(), 0.0);

      double cur_error = edge_rating - old_dp;

      assert(cur_error < 1000 && cur_error > -1000);

      // update both vectors based on error derived from 2 previous vectors
      for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
        double prevUser = user_node[i];
        double prevMovie = movie_node[i];

        user_node[i] += step_size * (cur_error * prevMovie - LAMBDA * prevUser);
        assert(std::isnormal(user_node[i]));

        movie_node[i] += step_size * (cur_error * prevUser - LAMBDA * prevMovie);
        assert(std::isnormal(movie_node[i]));
      }
    }
  }
};

struct SGD {
  Graph* graph;
  double step_size;
  static galois::DGAccumulator<double> DGAccumulator_accum;

  SGD(Graph* _graph, double _step_size) : 
      graph(_graph),step_size(_step_size) {}

  void static go(Graph& _graph) {
    iteration = 0;
    double rms_normalized = 0.0;

    do {
      auto step_size = getstep_size(iteration);

      SGD_doPartialGradientUpdate::go(_graph,step_size);

      DGAccumulator_accum.reset();
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          std::string impl_str("CUDA_DO_ALL_IMPL_SGD_" + (_graph.get_run_identifier()));
          galois::StatTimer StatTimer_cuda(impl_str.c_str());
          StatTimer_cuda.start();
          int __retval = 0;
          SGD_all_cuda(__retval, cuda_ctx);
          DGAccumulator_accum += __retval;
          StatTimer_cuda.stop();
        } else if (personality == CPU)
      #endif

      galois::do_all(_graph.begin(), _graph.end(), 
                     SGD { &_graph, step_size }, galois::loopname("SGD"));
      ++iteration;

      // calculate root mean squared error
      rms_normalized = std::sqrt(DGAccumulator_accum.reduce() /
                                 _graph.get_totalEdges());
      galois::gDebug("RMS Normalized : ", rms_normalized;
    } while((iteration < maxIterations) && (rms_normalized > 10));
    // loop until root mean squared error is low enough

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat("(NULL)", "NUM_ITERATIONS_" + 
          std::to_string(_graph.get_run_num()), (unsigned long)iteration, 0);
    }
  }

  void operator()(GNode src) const {
    NodeData& sdata= graph->getData(src);
    auto& movie_node = sdata.latent_vector;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); 
         jj != ej; 
         ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& ddata = graph->getData(dst);
      auto& user_node = ddata.latent_vector;
      double edge_rating = graph->getEdgeData(dst);

      // get error of prediction based on current latent vectors
      double cur_error2 = edge_rating - calcPrediction(sdata, ddata);
      DGAccumulator_accum += (cur_error2 * cur_error2);
    }
  }
};
galois::DGAccumulator<double> SGD::DGAccumulator_accum;

/******************************************************************************/
/* Main */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    // TODO galois::System G
    // TODO
    LonestarStart(argc, argv, name, desc, url);

    galois::StatManager statManager;
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT"), 
                      StatTimer_total("TIMER_TOTAL"), 
                      StatTimer_hg_init("TIMER_HG_INIT");

    StatTimer_total.start();

    std::vector<unsigned> scalefactor;
#ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = galois::runtime::getHostID();
    int gpu_device = gpudevice;
    //Parse arg string when running on multiple hosts and update/override personality
    //with corresponding value.
    if (personality_set.length() == galois::runtime::NetworkInterface::Num) {
      switch (personality_set.c_str()[my_host_id]) {
      case 'g':
        personality = GPU_CUDA;
        break;
      case 'o':
        assert(0);
        personality = GPU_OPENCL;
        break;
      case 'c':
      default:
        personality = CPU;
        break;
      }
#ifdef __GALOIS_SINGLE_HOST_MULTIPLE_GPUS__
      if (gpu_device == -1) {
        gpu_device = 0;
        for (unsigned i = 0; i < my_host_id; ++i) {
          if (personality_set.c_str()[i] != 'c') ++gpu_device;
        }
      }
#endif
      for (unsigned i=0; i<personality_set.length(); ++i) {
        if (personality_set.c_str()[i] == 'c') 
          scalefactor.push_back(scalecpu);
        else
          scalefactor.push_back(scalegpu);
      }
    }
#endif


   // TODO use new graph interface 
   // TODO support bipartite in new graph interface
   StatTimer_hg_init.start();
   Graph* hg;
   if (enableVCut) {
      if(vertexcut == CART_VCUT)
        hg = new Graph_cartesianCut(inputFile,partFolder, net.ID, net.Num, scalefactor, transpose);
      else if(vertexcut == PL_VCUT)
        hg = new Graph_vertexCut(inputFile,partFolder, net.ID, net.Num, scalefactor, transpose, VCutThreshold, bipartite);
   } else {
    hg = new Graph_edgeCut(inputFile,partFolder, net.ID, net.Num, scalefactor, transpose, bipartite);
   }

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = (*hg).getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      //galois::opencl::cl_env.init(cldevice.Value);
    }
#endif
    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
    StatTimer_init.start();
      InitializeGraph::go((*hg));
    StatTimer_init.stop();

    for(auto run = 0; run < numRuns; ++run){
      std::cout << "[" << net.ID << "] SGD::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      galois::StatTimer StatTimer_main(timer_str.c_str());

      (*hg).reset_num_iter(run);

      StatTimer_main.start();
        SGD::go((*hg));
      StatTimer_main.stop();

      if((run + 1) != numRuns){
        galois::runtime::getHostBarrier().wait();
        (*hg).reset_num_iter(run);
        InitializeGraph::go((*hg));
      }
    }

   StatTimer_total.stop();

    // Verify
    if (verify) {
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for (auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          for (auto i = 0; i < LATENT_VECTOR_SIZE; ++i)
            galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
                (*hg).getData(*ii).latent_vector[i]);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA)  {
        for (auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
              get_node_dist_current_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }

    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
