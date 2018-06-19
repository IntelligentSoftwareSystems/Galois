/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/graphs/OfflineGraph.h"
#include "galois/runtime/DistGraph.h"
#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

static const char* const name = "BFS - Distributed Heterogeneous";
static const char* const desc =
    "Bellman-Ford BFS on Distributed Galois with 2 loop transform.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int>
    maxIterations("maxIterations",
                  cll::desc("Maximum iterations: Default 1024"),
                  cll::init(1024));
static cll::opt<unsigned int>
    src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool>
    verify("verify",
           cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"),
           cll::init(false));

struct NodeData {
  std::atomic<unsigned long long> dist_current;
  unsigned long long dist_old;
};

typedef DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  Graph* graph;
  llvm::cl::opt<unsigned int>& local_src_node;

  InitializeGraph(llvm::cl::opt<unsigned int>& _src_node, Graph* _graph)
      : local_src_node(_src_node), graph(_graph) {}
  void static go(Graph& _graph) {
    struct SyncerPull_0 {
      static unsigned long long extract(uint32_t node_id,
                                        const struct NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          return get_node_dist_current_cuda(cuda_ctx, node_id);
        assert(personality == CPU);
#endif
        return node.dist_current;
      }
      static void setVal(uint32_t node_id, struct NodeData& node,
                         unsigned long long y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_dist_current_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
          node.dist_current = y;
      }
      typedef unsigned long long ValTy;
    };
    struct SyncerPull_1 {
      static unsigned long long extract(uint32_t node_id,
                                        const struct NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          return get_node_dist_old_cuda(cuda_ctx, node_id);
        assert(personality == CPU);
#endif
        return node.dist_old;
      }
      static void setVal(uint32_t node_id, struct NodeData& node,
                         unsigned long long y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_dist_old_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
          node.dist_old = y;
      }
      typedef unsigned long long ValTy;
    };

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      InitializeGraph_cuda(cuda_ctx);
    } else if (personality == CPU)
#endif
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        InitializeGraph_cuda(cuda_ctx);
      } else if (personality == CPU)
#endif
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          InitializeGraph_cuda(src_node, cuda_ctx);
        } else if (personality == CPU)
#endif
          galois::do_all(_graph.begin(), _graph.end(),
                         InitializeGraph{src_node, &_graph},
                         galois::loopname("InitGraph"));

    _graph.sync_pull<SyncerPull_0>("");
    _graph.sync_pull<SyncerPull_1>("");
  }

  void operator()(GNode src) const {
    NodeData& sdata    = graph->getData(src);
    sdata.dist_current = std::numeric_limits<unsigned long long>::max() / 4;
    sdata.dist_old     = std::numeric_limits<unsigned long long>::max() / 4;
    auto& net          = galois::runtime::getSystemNetworkInterface();
    if ((net.ID == 0) && (src == local_src_node)) {
      sdata.dist_current = 0;
      sdata.dist_old     = 0;
    }
  }
};

struct FirstItr_BFS {
  Graph* graph;

  FirstItr_BFS(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {

    struct Syncer_0 {
      static unsigned long long extract(uint32_t node_id,
                                        const struct NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          return get_node_dist_current_cuda(cuda_ctx, node_id);
        assert(personality == CPU);
#endif
        return node.dist_current;
      }
      static void reduce(uint32_t node_id, struct NodeData& node,
                         unsigned long long y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          add_node_dist_current_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
        {
          galois::atomicMin(node.dist_current, y);
        }
      }
      static void reset(uint32_t node_id, struct NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA)
          set_node_dist_current_cuda(cuda_ctx, node_id, 0);
        else if (personality == CPU)
#endif
        {
          node.dist_current =
              std::numeric_limits<unsigned long long>::max() / 4;
        }
      }
      typedef unsigned long long ValTy;
    };
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      FirstItr_BFS_cuda(cuda_ctx);
    } else if (personality == CPU)
#endif
      galois::do_all(
          _graph.begin(), _graph.end(), FirstItr_BFS{&_graph},
          galois::loopname("BFS"),
          galois::write_set("sync_push", "this->graph", "struct NodeData &",
                            "struct NodeData &", "dist_current",
                            "unsigned long long",
                            "{ galois::atomicMin(node.dist_current, y);}",
                            "{node.dist_current = std::numeric_limits<unsigned "
                            "long long>::max()/4; }"));
    _graph.sync_push<Syncer_0>("");
  }

  void operator()(GNode src) { /*, galois::UserContext<GNode>& ctx) const {*/
    NodeData& snode = graph->getData(src);
    auto& sdist     = snode.dist_current;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); jj != ej;
         ++jj) {
      GNode dst                   = graph->getEdgeDst(jj);
      auto& dnode                 = graph->getData(dst);
      unsigned long long new_dist = 1 + sdist;
      galois::atomicMin(dnode.dist_current, new_dist);
    }
  }
};

struct BFS {
  Graph* graph;
  static galois::DGAccumulator<int> DGAccumulator_accum;

  BFS(Graph* _graph) : graph(_graph) {}
  void static go(Graph& _graph) {
    FirstItr_BFS::go(_graph);

    unsigned int iterations = 0;
    do {
      DGAccumulator_accum.reset();

      struct Syncer_0 {
        static unsigned long long extract(uint32_t node_id,
                                          const struct NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
          if (personality == GPU_CUDA)
            return get_node_dist_current_cuda(cuda_ctx, node_id);
          assert(personality == CPU);
#endif
          return node.dist_current;
        }
        static void reduce(uint32_t node_id, struct NodeData& node,
                           unsigned long long y) {
#ifdef __GALOIS_HET_CUDA__
          if (personality == GPU_CUDA)
            add_node_dist_current_cuda(cuda_ctx, node_id, y);
          else if (personality == CPU)
#endif
          {
            galois::atomicMin(node.dist_current, y);
          }
        }
        static void reset(uint32_t node_id, struct NodeData& node) {
#ifdef __GALOIS_HET_CUDA__
          if (personality == GPU_CUDA)
            set_node_dist_current_cuda(cuda_ctx, node_id, 0);
          else if (personality == CPU)
#endif
          {
            node.dist_current =
                std::numeric_limits<unsigned long long>::max() / 4;
          }
        }
        typedef unsigned long long ValTy;
      };
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        int __retval = 0;
        BFS_cuda(__retval, cuda_ctx);
        DGAccumulator_accum += __retval;
      } else if (personality == CPU)
#endif
        galois::do_all(
            _graph.begin(), _graph.end(), BFS{&_graph}, galois::loopname("BFS"),
            galois::write_set(
                "sync_push", "this->graph", "struct NodeData &",
                "struct NodeData &", "dist_current", "unsigned long long",
                "{ galois::atomicMin(node.dist_current, y);}",
                "{node.dist_current = std::numeric_limits<unsigned long "
                "long>::max()/4; }"));
      _graph.sync_push<Syncer_0>("");
      ++iterations;

    } while (DGAccumulator_accum.reduce());

    std::cout << "Iterations XXXXXXXXXXXXXX : " << iterations << "\n";
  }

  void operator()(GNode src) { /*, galois::UserContext<GNode>& ctx) const {*/
    NodeData& snode = graph->getData(src);
    auto& sdist     = snode.dist_current;

    if (snode.dist_old > sdist) {
      snode.dist_old = sdist; // snode.dist_current;
      DGAccumulator_accum += 1;
      for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src);
           jj != ej; ++jj) {
        GNode dst                   = graph->getEdgeDst(jj);
        auto& dnode                 = graph->getData(dst);
        unsigned long long new_dist = 1 + sdist;
        galois::atomicMin(dnode.dist_current, new_dist);
      }
    }
  }
};
galois::DGAccumulator<int> BFS::DGAccumulator_accum;

/********Set source Node ************/
void setSource(Graph& _graph) {
  auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.ID == 0) {
    auto& nd        = _graph.getData(src_node);
    nd.dist_current = 0;
  }
}

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init,
        T_BFS1, T_BFS2, T_BFS3;

    T_total.start();

    T_DistGraph_init.start();
    Graph hg(inputFile, net.ID, net.Num);
    T_DistGraph_init.stop();

    std::cout << "InitializeGraph::go called\n";
    T_init.start();
    InitializeGraph::go(hg);
    T_init.stop();

    // Verify
    /*
        if(verify){
          if(net.ID == 0) {
            for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
              std::cout << "[" << *ii << "]  " << hg.getData(*ii).dist_current
       << "\n";
            }
          }
        }
    */

    std::cout << "BFS::go run1 called  on " << net.ID << "\n";
    T_BFS1.start();
    BFS::go(hg);
    T_BFS1.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get() << " BFS1 : " << T_BFS1.get()
              << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "BFS::go run2 called  on " << net.ID << "\n";
    T_BFS2.start();
    BFS::go(hg);
    T_BFS2.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get() << " BFS2 : " << T_BFS2.get()
              << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "BFS::go run3 called  on " << net.ID << "\n";
    T_BFS3.start();
    BFS::go(hg);
    T_BFS3.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get() << " BFS3 : " << T_BFS3.get()
              << " (msec)\n\n";

    T_total.stop();

    auto mean_time = (T_BFS1.get() + T_BFS2.get() + T_BFS3.get()) / 3;

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get() << " BFS1 : " << T_BFS1.get()
              << " BFS2 : " << T_BFS2.get() << " BFS3 : " << T_BFS3.get()
              << " BFS mean time (3 runs ) (" << maxIterations
              << ") : " << mean_time << "(msec)\n\n";

    if (verify) {
      for (auto ii = hg.begin(); ii != hg.end(); ++ii) {
        galois::runtime::printOutput("% %\n", hg.getGID(*ii),
                                     hg.getData(*ii).dist_current);
      }
    }
    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
