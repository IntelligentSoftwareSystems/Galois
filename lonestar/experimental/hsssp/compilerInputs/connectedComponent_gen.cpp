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
#include "DistGraph.h"

static const char* const name = "Connected Component Label Propagation - "
                                "Compiler Generated Distributed Heterogeneous";
static const char* const desc =
    "Connected Component Propagation on Distributed Galois.";
static const char* const url = 0;

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

struct CC_NodeData {
  std::atomic<uint32_t> id;
  std::atomic<uint32_t> comp;
};

typedef DistGraph<CC_NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}
  void static go(Graph& _graph) {
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{&_graph},
                   galois::loopname("Init"));
  }

  void operator()(GNode src) const {
    CC_NodeData& sdata = graph->getData(src);
    sdata.id           = graph->getGID(src);
    sdata.comp         = sdata.id.load();
  }
};

struct LabelPropAlgo {
  Graph* graph;

  LabelPropAlgo(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    galois::do_all(_graph.begin(), _graph.end(), LabelPropAlgo{&_graph},
                   galois::loopname("LabelPropAlgo"));
  }

  void operator()(GNode src) const {
    CC_NodeData& sdata = graph->getData(src);
    auto& s_comp       = sdata.comp;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); jj != ej;
         ++jj) {
      GNode dst   = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      //      auto& d_comp = dnode.comp;
      //      uint32_t old_comp = d_comp;
      uint32_t new_comp = s_comp;
      galois::atomicMin(dnode.comp, new_comp);
      /*      while(d_comp > new_comp) {
             d_comp.compare_exchange_strong(old_comp, new_comp);
            }*/
      // while(old_comp > new_comp && !d_comp.compare_exchange_strong(old_comp,
      // new_comp)){}
    }
  }
};

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_DistGraph_init, T_init, T_labelProp;

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
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).comp << "\n";
        }
      }
    }

    std::cout << "CC::go called\n";
    T_labelProp.start();
    for (int i = 0; i < maxIterations; ++i) {
      std::cout << " Iteration : " << i << "\n";
      LabelPropAlgo::go(hg);
    }
    T_labelProp.stop();

    // Verify
    if (verify) {
      if (net.ID == 0) {
        for (auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).comp << "\n";
        }
      }
    }

    T_total.stop();

    std::cout << "[" << net.ID << "]"
              << " Total Time : " << T_total.get()
              << " DistGraph : " << T_DistGraph_init.get()
              << " Init : " << T_init.get() << " PageRank (" << maxIterations
              << ") : " << T_labelProp.get() << "(msec)\n\n";

    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
