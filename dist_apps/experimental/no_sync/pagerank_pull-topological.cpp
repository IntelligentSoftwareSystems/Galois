/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/gstl.h"

#include "galois/runtime/CompilerHelperFunctions.h"
#include "galois/runtime/Tracer.h"

#include "galois/runtime/dGraph_edgeCut.h"
#include "galois/runtime/dGraph_vertexCut.h"

#include "galois/DReducible.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/cuda/cuda_device.h"
#include "gen_cuda.h"
struct CUDA_Context* cuda_ctx;
#endif

static const char* const name =
    "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "PageRank Pull version on Distributed Galois.";
static const char* const url  = 0;

#ifdef __GALOIS_HET_CUDA__
enum Personality { CPU, GPU_CUDA, GPU_OPENCL };
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

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> partFolder("partFolder",
                                        cll::desc("path to partitionFolder"),
                                        cll::init(""));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(0.000001));
static cll::opt<unsigned int>
    maxIterations("maxIterations",
                  cll::desc("Maximum iterations: Default 10000"),
                  cll::init(10000));
static cll::opt<bool>
    verify("verify", cll::desc("Verify ranks by printing to the output stream"),
           cll::init(false));

static cll::opt<bool>
    enableVCut("enableVertexCut",
               cll::desc("Use vertex cut for graph partitioning."),
               cll::init(false));

#ifdef __GALOIS_HET_CUDA__
static cll::opt<int> gpudevice(
    "gpu",
    cll::desc("Select GPU to run on, default is to choose automatically"),
    cll::init(-1));
static cll::opt<Personality>
    personality("personality", cll::desc("Personality"),
                cll::values(clEnumValN(CPU, "cpu", "Galois CPU"),
                            clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"),
                            clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"),
                            clEnumValEnd),
                cll::init(CPU));
static cll::opt<std::string>
    personality_set("pset",
                    cll::desc("String specifying personality for each host. "
                              "'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"),
                    cll::init(""));
static cll::opt<unsigned>
    scalegpu("scalegpu",
             cll::desc("Scale GPU workload w.r.t. CPU, default is "
                       "proportionally equal workload to CPU and GPU (1)"),
             cll::init(1));
static cll::opt<unsigned>
    scalecpu("scalecpu",
             cll::desc("Scale CPU workload w.r.t. GPU, default is "
                       "proportionally equal workload to CPU and GPU (1)"),
             cll::init(1));
static cll::opt<int> num_nodes(
    "num_nodes",
    cll::desc("Num of physical nodes with devices (default = num of hosts): "
              "detect GPU to use for each host automatically"),
    cll::init(-1));
#endif

static const float alpha = (1.0 - 0.85);
struct PR_NodeData {
  float value;
  std::atomic<int> nout;
};

typedef galois::graphs::DistGraph<PR_NodeData, void> Graph;
typedef galois::graphs::DistGraphEdgeCut<PR_NodeData, void> Graph_edgeCut;
typedef galois::graphs::DistGraphHybridCut<PR_NodeData, void> Graph_vertexCut;

typedef typename Graph::GraphNode GNode;

unsigned iteration;

struct ResetGraph {
  Graph* graph;

  ResetGraph(Graph* _graph) : graph(_graph) {}
  void static go(Graph& _graph) {
    galois::do_all(_graph.begin(), _graph.end(), ResetGraph{&_graph},
                   galois::loopname("ResetGraph"),
                   galois::numrun(_graph.get_run_identifier()));
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value        = 0;
    sdata.nout         = 0;
  }
};

struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}
  void static go(Graph& _graph) {
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{&_graph},
                   galois::loopname("InitializeGraph"),
                   galois::numrun(_graph.get_run_identifier()));
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value        = alpha;
    for (auto nbr = graph->edge_begin(src), ee = graph->edge_end(src);
         nbr != ee; ++nbr) {
      GNode dst          = graph->getEdgeDst(nbr);
      PR_NodeData& ddata = graph->getData(dst);
      galois::atomicAdd(ddata.nout, 1);
    }
  }
};

struct PageRank {
  Graph* graph;

  PageRank(Graph* _graph) : graph(_graph) {}
  void static go(Graph& _graph) {
    iteration = 0;
    do {
      _graph.set_num_round(iteration);
      DGAccumulator_accum.reset();
      galois::do_all(_graph.begin(), _graph.end(), PageRank{&_graph},
                     galois::loopname("PageRank"),
                     galois::numrun(_graph.get_run_identifier()));
      ++iteration;
    } while ((iteration < maxIterations) && DGAccumulator_accum.reduce());
    galois::runtime::reportStat(
        "(NULL)", "NumIterations_" + std::to_string(_graph.get_run_num()),
        (unsigned long)iteration, 0);
  }

  static galois::DGAccumulator<int> DGAccumulator_accum;
  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    float sum          = 0;
    for (auto nbr = graph->edge_begin(src), ee = graph->edge_end(src);
         nbr != ee; ++nbr) {
      GNode dst          = graph->getEdgeDst(nbr);
      PR_NodeData& ddata = graph->getData(dst);
      unsigned dnout     = ddata.nout;
      if (dnout > 0) {
        sum += ddata.value / dnout;
      }
    }

    float pr_value = sum * (1.0 - alpha) + alpha;
    float diff     = pr_value - sdata.value;

    if (diff > tolerance) {
      sdata.value = pr_value;
      DGAccumulator_accum += 1;
    }
  }
};
galois::DGAccumulator<int> PageRank::DGAccumulator_accum;

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    galois::runtime::reportStat("(NULL)", "Max Iterations",
                                (unsigned long)maxIterations, 0);
    std::ostringstream ss;
    ss << tolerance;
    galois::runtime::reportStat("(NULL)", "Tolerance", ss.str(), 0);
    galois::StatManager statManager;
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT"),
        StatTimer_total("TimerTotal"), StatTimer_hg_init("TIMER_HG_INIT");

    StatTimer_total.start();

    std::vector<unsigned> scalefactor;
#ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = galois::runtime::getHostID();
    int gpu_device            = gpudevice;
    // Parse arg string when running on multiple hosts and update/override
    // personality with corresponding value.
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
      if ((personality == GPU_CUDA) && (gpu_device == -1)) {
        gpu_device = get_gpu_device_id(personality_set, num_nodes);
      }
      for (unsigned i = 0; i < personality_set.length(); ++i) {
        if (personality_set.c_str()[i] == 'c')
          scalefactor.push_back(scalecpu);
        else
          scalefactor.push_back(scalegpu);
      }
    }
#endif

    assert(!enableVCut); // does not support vertex-cuts

    StatTimer_hg_init.start();
    Graph* hg;
    if (enableVCut) {
      hg = new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num,
                               scalefactor);
    } else {
      hg = new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                             scalefactor);
    }

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = (*hg).getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      // galois::opencl::cl_env.init(cldevice.Value);
    }
#endif
    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
    StatTimer_init.start();
    InitializeGraph::go((*hg));
    StatTimer_init.stop();

    for (auto run = 0; run < numRuns; ++run) {
      std::cout << "[" << net.ID << "] PageRank::go run " << run << " called\n";
      std::string timer_str("Timer_" + std::to_string(run));
      galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
      PageRank::go((*hg));
      StatTimer_main.stop();

      if ((run + 1) != numRuns) {
        galois::runtime::getHostBarrier().wait();
        (*hg).reset_num_iter(run + 1);
        ResetGraph::go((*hg));
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
          if ((*hg).isOwned((*hg).getGID(*ii)))
            galois::runtime::printOutput("% %\n", (*hg).getGID(*ii),
                                         (*hg).getData(*ii).value);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA) {
        for (auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if ((*hg).isOwned((*hg).getGID(*ii)))
            galois::runtime::printOutput("% %\n", (*hg).getGID(*ii),
                                         get_node_value_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }

    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
