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

#ifndef APPS_CONNECTEDCOMPONENTS_LIGRAALGO_H
#define APPS_CONNECTEDCOMPONENTS_LIGRAALGO_H

#include "galois/DomainSpecificExecutors.h"
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/GraphNodeBag.h"
#include "llvm/Support/CommandLine.h"

#include <boost/mpl/if.hpp>

extern llvm::cl::opt<unsigned int> memoryLimit;

template <typename Graph>
void readInOutGraph(Graph& graph);

template <bool UseGraphChi>
struct LigraAlgo : public galois::ligraGraphChi::ChooseExecutor<UseGraphChi> {
  struct LNode {
    typedef unsigned int component_type;
    unsigned int id;
    unsigned int comp;
    unsigned int oldComp;

    component_type component() { return comp; }
    bool isRep() { return id == comp; }
  };

  typedef typename galois::graphs::LC_CSR_Graph<LNode, void>::
      template with_no_lockable<true>::type ::template with_numa_alloc<
          true>::type InnerGraph;
  typedef typename boost::mpl::if_c<
      UseGraphChi, galois::graphs::OCImmutableEdgeGraph<LNode, void>,
      galois::graphs::LC_InOut_Graph<InnerGraph>>::type Graph;
  typedef typename Graph::GraphNode GNode;

  template <typename Bag>
  struct Initialize {
    Graph& graph;
    Bag& bag;

    Initialize(Graph& g, Bag& b) : graph(g), bag(b) {}
    void operator()(GNode n) const {
      LNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      data.comp   = data.id;
      bag.push(n, graph.sizeEdges() / graph.size());
    }
  };

  struct Copy {
    Graph& graph;

    Copy(Graph& g) : graph(g) {}
    void operator()(size_t n, galois::UserContext<size_t>&) { (*this)(n); }
    void operator()(size_t id) {
      GNode n      = graph.nodeFromId(id);
      LNode& data  = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      data.oldComp = data.comp;
    }
  };

  struct EdgeOperator {
    template <typename GTy>
    bool cond(GTy& graph, typename GTy::GraphNode) {
      return true;
    }

    template <typename GTy>
    bool operator()(GTy& graph, typename GTy::GraphNode src,
                    typename GTy::GraphNode dst,
                    typename GTy::edge_data_reference) {
      LNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      typename LNode::component_type orig = ddata.comp;
      while (true) {
        typename LNode::component_type old = ddata.comp;
        if (old <= sdata.comp)
          return false;
        if (__sync_bool_compare_and_swap(&ddata.comp, old, sdata.comp)) {
          return orig == ddata.oldComp;
        }
      }
      return false;
    }
  };

  template <typename G>
  void readGraph(G& graph) {
    readInOutGraph(graph);
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  void operator()(Graph& graph) {
    typedef galois::worklists::PerSocketChunkFIFO<256> WL;
    typedef galois::graphsNodeBagPair<> BagPair;
    BagPair bags(graph.size());

    galois::do_all(graph,
                   Initialize<typename BagPair::bag_type>(graph, bags.next()));
    while (!bags.next().empty()) {
      bags.swap();
      galois::for_each(bags.cur(), Copy(graph), galois::wl<WL>());
      this->outEdgeMap(memoryLimit, graph, EdgeOperator(), bags.cur(),
                       bags.next(), false);
    }
  }
};

#endif
