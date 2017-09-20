/** Residual based Page Rank -*- C++ -*-
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
 * Compute pageRank Pull version using residual on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/gstl.h"

#include "galois/runtime/CompilerHelperFunctions.h"
#include "galois/runtime/Tracer.h"

#include "galois/Dist/vGraph.h"
#include "galois/DistAccumulator.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/Cuda/cuda_mtypes.h"
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;
#endif

static const char* const name = "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "PageRank Pull version on Distributed Galois.";
static const char* const url = 0;

#ifdef __GALOIS_HET_CUDA__
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
   assert(false&& "Invalid personality");
   return "";
}
#endif

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> partFolder("partFolder", cll::desc("path to partitionFolder"), cll::init(""));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.01));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to the output stream"), cll::init(false));
#ifdef __GALOIS_HET_CUDA__
static cll::opt<int> gpudevice("gpu", cll::desc("Select GPU to run on, default is to choose automatically"), cll::init(-1));
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), clEnumValEnd),
      cll::init(CPU));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init(""));
#endif


static const float alpha = (1.0 - 0.85);
struct PR_NodeData {
  float value;
  float sum;
  std::atomic<int> nout;
};

typedef vGraph<PR_NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph) {
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ &_graph }, galois::loopname("Init"));
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value = 1.0 - alpha;
    sdata.sum = 0;
    for(auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src); ++nbr){
      GNode dst = graph->getEdgeDst(nbr);
      PR_NodeData& ddata = graph->getData(dst);
      galois::atomicAdd(ddata.nout, 1);
    }
  }
};

struct PageRank_pull_partial {
  Graph* graph;

  PageRank_pull_partial(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph) {
    galois::do_all(_graph.begin(), _graph.end(), PageRank_pull_partial { &_graph }, galois::loopname("pageRank partial"));
  }

  void operator()(GNode src)const {
    PR_NodeData& sdata = graph->getData(src);
    for(auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src); ++nbr){
      GNode dst = graph->getEdgeDst(nbr);
      PR_NodeData& ddata = graph->getData(dst);
      unsigned dnout = ddata.nout;
      if (dnout > 0) {
        sdata.sum += ddata.value/dnout;
      }
    }
  }
};

struct PageRank_pull {
  Graph* graph;

  PageRank_pull(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph) {
    do{
      DGAccumulator_accum.reset();
      PageRank_pull_partial::go(_graph);
      galois::do_all(_graph.begin(), _graph.end(), PageRank_pull { &_graph }, galois::loopname("pageRank"));
    } while (DGAccumulator_accum.reduce());
  }

  static galois::DGAccumulator<int> DGAccumulator_accum;
  void operator()(GNode src)const {
    PR_NodeData& sdata = graph->getData(src);
    float pr_value = sdata.sum*(1.0 - alpha) + alpha;
    float diff = std::fabs(pr_value - sdata.value);
    sdata.sum = 0;

    if(diff > tolerance){
      sdata.value = pr_value; 
      DGAccumulator_accum+= 1;
    }
  }
};
galois::DGAccumulator<int>  PageRank_pull::DGAccumulator_accum;

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_vGraph_init, T_init, T_pageRank;

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
    }
#endif

    T_total.start();

    T_vGraph_init.start();
    Graph hg(inputFile, partFolder, net.ID, net.Num);
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = hg.getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m);
    } else if (personality == GPU_OPENCL) {
      //galois::OpenCL::cl_env.init(cldevice.Value);
    }
#endif
    T_vGraph_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";

    T_init.start();
    InitializeGraph::go(hg);
    T_init.stop();

    // Verify
    /*if(verify){
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          galois::runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).nout);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          galois::runtime::printOutput("% %\n", hg.getGID(*ii), get_node_nout_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }*/

    std::cout << "[" << net.ID << "] PageRank_pull::go called\n";
    T_pageRank.start();
    PageRank_pull::go(hg);
    T_pageRank.stop();

    T_total.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " vGraph : " << T_vGraph_init.get() << " Init : " << T_init.get() << " PageRank_pull : " << T_pageRank.get() << "(msec)\n\n";

    // Verify
    if(verify){
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          galois::runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).value);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          galois::runtime::printOutput("% %\n", hg.getGID(*ii), get_node_value_cuda(cuda_ctx, *ii));
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
