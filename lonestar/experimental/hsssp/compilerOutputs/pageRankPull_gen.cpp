/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/Dist/DistGraph.h"
#include "galois/DistAccumulator.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/Cuda/cuda_mtypes.h"
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
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(4));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(0.01));
static cll::opt<bool>
    verify("verify", cll::desc("Verify ranks by printing to the output stream"),
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
#endif

static const float alpha = (1.0 - 0.85);
struct PR_NodeData {
  float value;
  std::atomic<int> nout;
};

typedef DistGraph<PR_NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  const float& local_alpha;
  Graph* graph;

  InitializeGraph(const float& _alpha, Graph* _graph)
      : local_alpha(_alpha), graph(_graph) {}
  void static go(Graph& _graph) {
    struct SyncerPull_0 {
      static float extract(uint32_t node_id, const struct PR_NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          return get_node_value_cuda(cuda_ctx, node_id);
        assert(personality == CPU);
#endif
        return node.value;
      }
      static void setVal(uint32_t node_id, struct PR_NodeData& node, float y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_value_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
          node.value = y;
      }
      typedef float ValTy;
    };
    struct SyncerPull_1 {
      static int extract(uint32_t node_id, const struct PR_NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          return get_node_nout_cuda(cuda_ctx, node_id);
        assert(personality == CPU);
#endif
        return node.nout;
      }
      static void setVal(uint32_t node_id, struct PR_NodeData& node, int y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_nout_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
          node.nout = y;
      }
      typedef int ValTy;
    };
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      InitializeGraph_cuda(alpha, cuda_ctx);
    } else if (personality == CPU)
#endif
      galois::do_all(
          _graph.begin(), _graph.end(), InitializeGraph{alpha, &_graph},
          galois::loopname("Init"),
          galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &",
                            "struct PR_NodeData &", "value", "float"));
    _graph.sync_pull<SyncerPull_0>();
    _graph.sync_pull<SyncerPull_1>();
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value        = 1.0 - local_alpha;
    sdata.nout         = 0;
  }
};

struct PrecomputeGraph {
  Graph* graph;

  void static go(Graph& _graph) {
    struct Syncer_0 {
      static int extract(uint32_t node_id, const struct PR_NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          return get_node_nout_cuda(cuda_ctx, node_id);
        assert(personality == CPU);
#endif
        return node.nout;
      }
      static void reduce(uint32_t node_id, struct PR_NodeData& node, int y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          add_node_nout_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
        {
          galois::atomicAdd(node.nout, y);
        }
      }
      static void reset(uint32_t node_id, struct PR_NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_nout_cuda(cuda_ctx, node_id, 0);
        else if (personality == CPU)
#endif
        {
          node.nout = 0;
        }
      }
      typedef int ValTy;
    };
    struct SyncerPull_0 {
      static int extract(uint32_t node_id, const struct PR_NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          return get_node_nout_cuda(cuda_ctx, node_id);
        assert(personality == CPU);
#endif
        return node.nout;
      }
      static void setVal(uint32_t node_id, struct PR_NodeData& node, int y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_nout_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
          node.nout = y;
      }
      typedef int ValTy;
    };
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      PrecomputeGraph_cuda(cuda_ctx);
    } else if (personality == CPU)
#endif
      galois::do_all(
          _graph.begin(), _graph.end(), PrecomputeGraph{&_graph},
          galois::loopname("Precompute"),
          galois::write_set("sync_push", "this->graph", "struct PR_NodeData &",
                            "struct PR_NodeData &", "nout", "int",
                            "{ galois::atomicAdd(node.nout, y);}",
                            "{node.nout = 0 ; }"),
          galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &",
                            "struct PR_NodeData &", "nout", "int"));
    _graph.sync_push<Syncer_0>();

    _graph.sync_pull<SyncerPull_0>();
  }

  void operator()(GNode src) const {
    for (auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src);
         ++nbr) {
      GNode dst          = graph->getEdgeDst(nbr);
      PR_NodeData& ddata = graph->getData(dst);
      galois::atomicAdd(ddata.nout, 1);
    }
  }
};

struct PageRank_pull {
  const float& local_alpha;
  const float& local_tolerance;
  Graph* graph;

  PageRank_pull(const float& _tolerance, const float& _alpha, Graph* _graph)
      : local_tolerance(_tolerance), local_alpha(_alpha), graph(_graph) {}
  void static go(Graph& _graph) {

    do {
      DGAccumulator_accum.reset();

      struct SyncerPull_0 {
        static float extract(uint32_t node_id, const struct PR_NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
          if (personality == GPU_CUDA)
            return get_node_value_cuda(cuda_ctx, node_id);
          assert(personality == CPU);
#endif
          return node.value;
        }
        static void setVal(uint32_t node_id, struct PR_NodeData& node,
                           float y) {
#ifdef __GALOIS_HET_CUDA__
          if (personality == GPU_CUDA)
            set_node_value_cuda(cuda_ctx, node_id, y);
          else if (personality == CPU)
#endif
            node.value = y;
        }
        typedef float ValTy;
      };
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        int __retval = 0;
        PageRank_pull_cuda(__retval, alpha, tolerance, cuda_ctx);
        DGAccumulator_accum += __retval;
      } else if (personality == CPU)
#endif
        galois::do_all(_graph.begin(), _graph.end(),
                       PageRank_pull{tolerance, alpha, &_graph},
                       galois::loopname("pageRank"),
                       galois::write_set(
                           "sync_pull", "this->graph", "struct PR_NodeData &",
                           "struct PR_NodeData &", "value", "float"));
      _graph.sync_pull<SyncerPull_0>();

    } while (DGAccumulator_accum.reduce());
  }

  static galois::DGAccumulator<int> DGAccumulator_accum;
  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    float sum          = 0;
    for (auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src);
         ++nbr) {
      GNode dst          = graph->getEdgeDst(nbr);
      PR_NodeData& ddata = graph->getData(dst);
      unsigned dnout     = ddata.nout;
      if (dnout > 0) {
        sum += ddata.value / dnout;
      }
    }

    float pr_value = sum * (1.0 - local_alpha) + local_alpha;
    float diff     = std::fabs(pr_value - sdata.value);

    if (diff > local_tolerance) {
      sdata.value = pr_value;
      DGAccumulator_accum += 1;
    }
  }
};
galois::DGAccumulator<int> PageRank_pull::DGAccumulator_accum;

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init,
        T_pageRank;

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
#ifdef __GALOIS_SINGLE_HOST_MULTIPLE_GPUS__
      if (gpu_device == -1) {
        gpu_device = 0;
        for (unsigned i = 0; i < my_host_id; ++i) {
          if (personality_set.c_str()[i] != 'c')
            ++gpu_device;
        }
      }
#endif
    }
    std::vector<unsigned> scalefactor;
    for (unsigned i = 0; i < personality_set.length(); ++i) {
      if (personality_set.c_str()[i] == 'c')
        scalefactor.push_back(scalecpu);
      else
        scalefactor.push_back(scalegpu);
    }
#endif

    T_total.start();

    T_offlineGraph_init.start();
    OfflineGraph g(inputFile);
    T_offlineGraph_init.stop();
    std::cout << g.size() << " " << g.sizeEdges() << "\n";

    T_DistGraph_init.start();
#ifndef __GALOIS_HET_CUDA__
    Graph hg(inputFile, net.ID, net.Num);
#else
    Graph hg(inputFile, net.ID, net.Num, scalefactor);
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = hg.getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m);
    } else if (personality == GPU_OPENCL) {
      // galois::opencl::cl_env.init(cldevice.Value);
    }
#endif
    T_DistGraph_init.stop();

    std::cout << "InitializeGraph::go called\n";

    T_init.start();
    InitializeGraph::go(hg);
    PrecomputeGraph::go(hg);
    T_init.stop();

    // Verify
    /*if(verify){
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) {
#endif
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          galois::runtime::printOutput("% %\n", hg.getGID(*ii),
hg.getData(*ii).nout);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          galois::runtime::printOutput("% %\n", hg.getGID(*ii),
get_node_nout_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }*/

    std::cout << "PageRank_pull::go called\n";
    T_pageRank.start();
    PageRank_pull::go(hg);
    T_pageRank.stop();

    T_total.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get() << " PageRank_pull ("
              << maxIterations << ") : " << T_pageRank.get() << "(msec)\n\n";

    // Verify
    if (verify) {
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) {
#endif
        for (auto ii = hg.begin(); ii != hg.end(); ++ii) {
          galois::runtime::printOutput("% %\n", hg.getGID(*ii),
                                       hg.getData(*ii).value);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA) {
        for (auto ii = hg.begin(); ii != hg.end(); ++ii) {
          galois::runtime::printOutput("% %\n", hg.getGID(*ii),
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
