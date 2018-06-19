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
#include "Lonestar/BoilerPlate.h"
#include "galois/gstl.h"

#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/graphs/OfflineGraph.h"
#include "galois/Dist/DistGraph.h"

static const char* const name =
    "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "Residual PageRank on Distributed Galois.";
static const char* const url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(4));
static cll::opt<unsigned int>
    src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool>
    verify("verify",
           cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"),
           cll::init(false));

static const float alpha = (1.0 - 0.85);
struct PR_NodeData {
  float value;
  std::atomic<float> residual;
  unsigned int nout;
};

typedef DistGraph<PR_NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}
  void static go(Graph& _graph) {
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{&_graph},
                   galois::loopname("Init"));
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value        = 1.0 - alpha;
    sdata.nout = std::distance(graph->edge_begin(src), graph->edge_end(src));

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

  PageRank(Graph* _graph) : graph(_graph) {}
  void static go(Graph& _graph) {
    galois::do_all(_graph.begin(), _graph.end(), PageRank{&_graph},
                   galois::loopname("pageRank"));
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    float residual_old = sdata.residual.exchange(0.0);
    sdata.value += residual_old;
    // sdata.residual = residual_old;
    if (sdata.nout > 0) {
      float delta = residual_old * alpha / sdata.nout;
      for (auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src);
           ++nbr) {
        GNode dst          = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        galois::atomicAdd(ddata.residual, delta);
      }
    }
  }
};

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init,
        T_pageRank;

    T_total.start();

    T_DistGraph_init.start();
    Graph hg(inputFile, net.ID, net.Num);
    T_DistGraph_init.stop();

    std::cout << "InitializeGraph::go called\n";

    T_init.start();
    InitializeGraph::go(hg);
    T_init.stop();

    // Verify
    if (verify) {
      if (net.ID == 0) {
        for (auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).value << "\n";
        }
      }
    }

    std::cout << "PageRank::go called\n";
    T_pageRank.start();
    for (int i = 0; i < maxIterations; ++i) {
      std::cout << " Iteration : " << i << "\n";
      PageRank::go(hg);
    }
    T_pageRank.stop();

    // Verify
    if (verify) {
      if (net.ID == 0) {
        for (auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).value << "\n";
        }
      }
    }

    T_total.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get() << " PageRank (" << maxIterations
              << ") : " << T_pageRank.get() << "(msec)\n\n";

    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
