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

#ifndef APPS_CONNECTEDCOMPONENTS_GRAPHCHIALGO_H
#define APPS_CONNECTEDCOMPONENTS_GRAPHCHIALGO_H

#include "galois/DomainSpecificExecutors.h"
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/GraphNodeBag.h"
#include "llvm/Support/CommandLine.h"

#include <boost/mpl/if.hpp>

extern llvm::cl::opt<unsigned int> memoryLimit;

template <typename Graph>
void readInOutGraph(Graph& graph);

struct GraphChiAlgo : public galois::ligraGraphChi::ChooseExecutor<true> {
  struct LNode {
    typedef unsigned int component_type;
    unsigned int id;
    unsigned int comp;

    component_type component() { return comp; }
    bool isRep() { return id == comp; }
  };

  typedef galois::graphs::OCImmutableEdgeGraph<LNode, void> Graph;
  typedef Graph::GraphNode GNode;
  typedef galois::graphsNodeBagPair<> BagPair;

  template <typename G>
  void readGraph(G& graph) {
    readInOutGraph(graph);
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  struct Initialize {
    Graph& graph;

    Initialize(Graph& g) : graph(g) {}
    void operator()(GNode n) const {
      LNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      data.comp   = data.id;
    }
  };

  struct Process {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;

    typedef BagPair::bag_type bag_type;
    bag_type& next;

    Process(bag_type& n) : next(n) {}

    //! Add the next edge between components to the worklist
    template <typename GTy>
    void operator()(GTy& graph, const GNode& src) const {
      LNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      typename LNode::component_type m = sdata.comp;

      for (typename GTy::edge_iterator
               ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
               ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
           ii != ei; ++ii) {
        GNode dst    = graph.getEdgeDst(ii);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        m            = std::min(m, ddata.comp);
      }

      for (typename GTy::in_edge_iterator
               ii = graph.in_edge_begin(src, galois::MethodFlag::UNPROTECTED),
               ei = graph.in_edge_end(src, galois::MethodFlag::UNPROTECTED);
           ii != ei; ++ii) {
        GNode dst    = graph.getInEdgeDst(ii);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        m            = std::min(m, ddata.comp);
      }

      if (m != sdata.comp) {
        sdata.comp = m;
        for (typename GTy::edge_iterator
                 ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
                 ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
             ii != ei; ++ii) {
          GNode dst    = graph.getEdgeDst(ii);
          LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
          if (m < ddata.comp) {
            next.push(graph.idFromNode(dst), 1);
          }
        }
        for (typename GTy::in_edge_iterator
                 ii = graph.in_edge_begin(src, galois::MethodFlag::UNPROTECTED),
                 ei = graph.in_edge_end(src, galois::MethodFlag::UNPROTECTED);
             ii != ei; ++ii) {
          GNode dst    = graph.getInEdgeDst(ii);
          LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
          if (m < ddata.comp) {
            next.push(graph.idFromNode(dst), 1);
          }
        }
      }
    }
  };

  void operator()(Graph& graph) {
    BagPair bags(graph.size());

    galois::do_all(graph, Initialize(graph));
    galois::graphsChi::vertexMap(graph, Process(bags.next()), memoryLimit);
    while (!bags.next().empty()) {
      bags.swap();
      galois::graphsChi::vertexMap(graph, Process(bags.next()), bags.cur(),
                                   memoryLimit);
    }
  }
};

#endif
