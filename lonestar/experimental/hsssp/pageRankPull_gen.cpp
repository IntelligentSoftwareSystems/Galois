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
#include "galois/Dist/DistGraph.h"
#include "galois/DistAccumulator.h"

static const char* const name =
    "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "PageRank Pull version on Distributed Galois.";
static const char* const url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file transpose graph>"),
              cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(4));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(0.01));
static cll::opt<bool>
    verify("verify", cll::desc("Verify ranks by printing to the output stream"),
           cll::init(false));

static const float alpha = 0.85; //(1.0 - 0.85);
// static const float  tolerance = 0.1;
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
    galois::do_all(_graph.begin(), _graph.end(),
                   InitializeGraph{alpha, &_graph}, galois::loopname("Init"));
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value        = 1.0 - local_alpha;
    galois::atomicAdd(sdata.nout, 0);
    for (auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src);
         ++nbr) {
      GNode dst          = graph->getEdgeDst(nbr);
      PR_NodeData& ddata = graph->getData(dst);
      galois::atomicAdd(ddata.nout, 1);
    }
  }
};

struct InitializeGraph_value {
  Graph* graph;

  void static go(Graph& _graph) {

    struct SyncerPull_0 {
      static float extract(GNode src, const struct PR_NodeData& node) {
        return node.value;
      }
      static void setVal(GNode src, struct PR_NodeData& node, float y) {
        node.value = y;
      }
      typedef float ValTy;
    };

    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph_value{&_graph},
                   galois::loopname("Init"));
    _graph.sync_pull<SyncerPull_0>();
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value        = 1.0 - alpha;
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
      galois::do_all(_graph.begin(), _graph.end(),
                     PageRank_pull{tolerance, alpha, &_graph},
                     galois::loopname("pageRank"));
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
      if (ddata.nout > 0) {
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
        T_pageRank1, T_pageRank2, T_pageRank3;

    std::cout << "[ " << net.ID << " ] InputFile : " << inputFile << "\n";

    T_total.start();

    T_offlineGraph_init.start();
    OfflineGraph g(inputFile);
    T_offlineGraph_init.stop();
    std::cout << g.size() << " " << g.sizeEdges() << "\n";

    T_DistGraph_init.start();
    Graph hg(inputFile, net.ID, net.Num);
    T_DistGraph_init.stop();

    std::cout << "InitializeGraph::go called\n";

    T_init.start();
    InitializeGraph::go(hg);
    T_init.stop();
    galois::runtime::getHostBarrier().wait();

    // Verify
#if 0
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).nout << "\n";
        }
      }
    }
#endif

    std::cout << "PageRank::go run1 called  on " << net.ID << "\n";
    T_pageRank1.start();
    PageRank_pull::go(hg);
    T_pageRank1.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get()
              << " PageRank1 : " << T_pageRank1.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph_value::go(hg);

    std::cout << "PageRank::go run2 called  on " << net.ID << "\n";
    T_pageRank2.start();
    PageRank_pull::go(hg);
    T_pageRank2.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " offlineGraph : " << T_offlineGraph_init.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get()
              << " PageRank2 : " << T_pageRank2.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph_value::go(hg);

    std::cout << "PageRank::go run3 called  on " << net.ID << "\n";
    T_pageRank3.start();
    PageRank_pull::go(hg);
    T_pageRank3.stop();

    // Verify
    if (verify) {
      for (auto ii = hg.begin(); ii != hg.end(); ++ii) {
        galois::runtime::printOutput("% %\n", hg.getGID(*ii),
                                     hg.getData(*ii).value);
        // std::cout << "[" << *ii << "]  " << hg.getData(*ii).value << "\n";
      }
    }

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

    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
