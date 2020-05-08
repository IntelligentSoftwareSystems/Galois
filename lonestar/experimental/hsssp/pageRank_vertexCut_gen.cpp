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

#include "galois/graphs/OfflineGraph.h"
#include "galois/Dist/vGraph.h"
#include "galois/DistAccumulator.h"

static const char* const name =
    "PageRank - Compiler Generated Distributed Heterogeneous using Vertex Cut.";
static const char* const desc =
    "Residual PageRank on Distributed Galois using vertex Cut.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(1000));
static cll::opt<unsigned int>
    src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(0.01));
static cll::opt<bool>
    verify("verify",
           cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"),
           cll::init(false));

static cll::opt<std::string> partFolder("partFolder",
                                        cll::desc("path to partitionFolder"),
                                        cll::init(""));

static const float alpha = (1.0 - 0.85);
struct PR_NodeData {
  float value;
  unsigned int nout;
  std::atomic<float> residual;
};

typedef vGraph<PR_NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  Graph* graph;

  struct Syncer_1 {
    static float extract(GNode gid, const struct PR_NodeData& node) {
      return node.nout;
    }
    static void reduce(GNode gid, struct PR_NodeData& node, unsigned y) {
      node.nout += y;
    }
    static void reset(GNode gid, struct PR_NodeData& node) { node.nout = 0; }
    typedef float ValTy;
  };
  struct SyncerPull_1 {
    static unsigned int extract(GNode gid, const struct PR_NodeData& node) {
      return node.nout;
    }
    static void setVal(GNode gid, struct PR_NodeData& node, unsigned int y) {
      node.nout = y;
    }
    typedef unsigned int ValTy;
  };
  void static go(Graph& _graph) {

    galois::do_all(
        _graph.begin(), _graph.end(), InitializeGraph{&_graph},
        galois::loopname("Init"),
        galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &",
                          "struct PR_NodeData &", "value", "float"),
        galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &",
                          "struct PR_NodeData &", "nout", "unsigned int"),
        galois::write_set("sync_push", "this->graph", "struct PR_NodeData &",
                          "struct PR_NodeData &", "residual", "float",
                          "{ galois::atomicAdd(node.residual, y);}", "0"));
    _graph.sync_push<Syncer_1>();
    _graph.sync_pull<SyncerPull_1>();
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.nout = std::distance(graph->edge_begin(src), graph->edge_end(src));

    /**Synchronize nout since it's used after this: Operator splitting.**/
    // graph->sync_push<Syncer_1>();
    // graph->sync_pull<SyncerPull_1>();
  }
};

/** To initialize residual for each node **/
struct InitializeGraph_residual {
  Graph* graph;

  struct Syncer_0 {
    static float extract(GNode gid, const struct PR_NodeData& node) {
      return node.residual;
    }
    static void reduce(GNode gid, struct PR_NodeData& node, float y) {
      galois::atomicAdd(node.residual, y);
    }
    static void reset(GNode gid, struct PR_NodeData& node) {
      node.residual = 0;
    }
    typedef float ValTy;
  };
  struct SyncerPull_0 {
    static float extract(GNode gid, const struct PR_NodeData& node) {
      return node.residual;
    }
    static void setVal(GNode gid, struct PR_NodeData& node, float y) {
      node.residual = y;
    }
    typedef float ValTy;
  };
  void static go(Graph& _graph) {

    galois::do_all(
        _graph.begin(), _graph.end(), InitializeGraph_residual{&_graph},
        galois::loopname("Init"),
        galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &",
                          "struct PR_NodeData &", "value", "float"),
        galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &",
                          "struct PR_NodeData &", "nout", "unsigned int"),
        galois::write_set("sync_push", "this->graph", "struct PR_NodeData &",
                          "struct PR_NodeData &", "residual", "float",
                          "{ galois::atomicAdd(node.residual, y);}", "0"));
    _graph.sync_push<Syncer_0>();
    _graph.sync_pull<SyncerPull_0>();
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value        = 1.0 - alpha;
    if (sdata.nout > 0) {
      float delta = sdata.value * alpha / sdata.nout;
      for (auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src);
           ++nbr) {
        GNode dst          = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        galois::atomicAdd(ddata.residual, delta);
      }
    }
  }
};

struct PageRank {
  Graph* graph;

  void static go(Graph& _graph) {
    struct Syncer_0 {
      static float extract(GNode gid, const struct PR_NodeData& node) {
        return node.residual;
      }
      static void reduce(GNode gid, struct PR_NodeData& node, float y) {
        galois::atomicAdd(node.residual, y);
      }
      static void reset(GNode gid, struct PR_NodeData& node) {
        node.residual = 0;
      }
      typedef float ValTy;
    };
    struct SyncerPull_0 {
      static float extract(GNode gid, const struct PR_NodeData& node) {
        return node.residual;
      }
      static void setVal(GNode gid, struct PR_NodeData& node, float y) {
        node.residual = y;
      }
      typedef float ValTy;
    };

    do {
      DGAccumulator_accum.reset();

      galois::do_all(
          _graph.begin(), _graph.end(), PageRank{&_graph},
          galois::loopname("pageRank"),
          galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &",
                            "struct PR_NodeData &", "value", "float"),
          galois::write_set("sync_push", "this->graph", "struct PR_NodeData &",
                            "struct PR_NodeData &", "residual", "float",
                            "{ galois::atomicAdd(node.residual, y);}", "0"));
      _graph.sync_push<Syncer_0>();
      _graph.sync_pull<SyncerPull_0>();

    } while (DGAccumulator_accum.reduce());
  }

  static galois::DGAccumulator<int> DGAccumulator_accum;
  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    float residual_old = sdata.residual.exchange(0.0);
    sdata.value += residual_old;
    if (sdata.nout > 0) {
      float delta = residual_old * alpha / sdata.nout;
      for (auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src);
           ++nbr) {
        GNode dst             = graph->getEdgeDst(nbr);
        PR_NodeData& ddata    = graph->getData(dst);
        auto dst_residual_old = galois::atomicAdd(ddata.residual, delta);
        if ((dst_residual_old <= tolerance) &&
            ((dst_residual_old + delta) >= tolerance)) {
          DGAccumulator_accum += 1;
        }
      }
    }
  }
};
galois::DGAccumulator<int> PageRank::DGAccumulator_accum;

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init,
        T_pageRank1, T_pageRank2, T_pageRank3;

    std::cout << "[ " << net.ID << " ] InputFile : " << inputFile << "\n";
    std::cout << "[ " << net.ID << " ] PartFile : " << partFolder << "\n";

    T_total.start();

    T_DistGraph_init.start();
    Graph hg(inputFile, partFolder, net.ID, net.Num);
    T_DistGraph_init.stop();

    std::cout << "InitializeGraph::go called\n";

    T_init.start();
    InitializeGraph::go(hg);
    InitializeGraph_residual::go(hg);
    T_init.stop();
    galois::runtime::getHostBarrier().wait();

    // Verify

    /*
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          //std::cout << "[" << hg.getGID(*ii) << "]  " << hg.getData(*ii).nout
    << " val : " <<hg.getData(*ii).residual <<   "\n";
          galois::runtime::printOutput("[%] % , res : %\n", hg.getGID(*ii),
    hg.getData(*ii).nout , hg.getData(*ii).residual);
        }
      }
    }
    */

    std::cout << "PageRank::go run1 called  on " << net.ID << "\n";
    T_pageRank1.start();
    PageRank::go(hg);
    T_pageRank1.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get()
              << " PageRank1 : " << T_pageRank1.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph_residual::go(hg);

    std::cout << "PageRank::go run2 called  on " << net.ID << "\n";
    T_pageRank2.start();
    PageRank::go(hg);
    T_pageRank2.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get()
              << " PageRank2 : " << T_pageRank2.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph_residual::go(hg);

    std::cout << "PageRank::go run3 called  on " << net.ID << "\n";
    T_pageRank3.start();
    PageRank::go(hg);
    T_pageRank3.stop();

    T_total.stop();
    auto mean_time =
        (T_pageRank1.get() + T_pageRank2.get() + T_pageRank3.get()) / 3;

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get()
              << " PageRank1 : " << T_pageRank1.get()
              << " PageRank2 : " << T_pageRank2.get()
              << " PageRank3 : " << T_pageRank3.get()
              << " PageRank mean time (3 runs ) (" << maxIterations
              << ") : " << mean_time << "(msec)\n\n";

    // Verify
    if (verify) {
      // if(net.ID == 0) {
      for (auto ii = hg.begin(); ii != hg.end(); ++ii) {
        // std::cout << "[" << hg.getGID(*ii) << "]  " << hg.getData(*ii).value
        // << "\n";
        galois::runtime::printOutput("% %\n", hg.getGID(*ii),
                                     hg.getData(*ii).value);
      }
      //}
    }
    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
