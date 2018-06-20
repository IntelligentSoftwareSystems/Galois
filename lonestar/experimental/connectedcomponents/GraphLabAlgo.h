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

#ifndef APPS_CONNECTEDCOMPONENTS_GRAPHLABALGO_H
#define APPS_CONNECTEDCOMPONENTS_GRAPHLABALGO_H

#include "galois/DomainSpecificExecutors.h"
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/GraphNodeBag.h"

#include <boost/mpl/if.hpp>

template <typename Graph>
void readInOutGraph(Graph& graph);

struct GraphLabAlgo {
  struct LNode {
    typedef size_t component_type;
    unsigned int id;
    component_type labelid;

    component_type component() { return labelid; }
    bool isRep() { return id == labelid; }
  };

  typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<
      true>::type ::with_numa_alloc<true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  struct Initialize {
    Graph& graph;

    Initialize(Graph& g) : graph(g) {}
    void operator()(GNode n) const {
      LNode& data  = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      data.labelid = data.id;
    }
  };

  struct Program {
    typedef size_t gather_type;

    struct message_type {
      size_t value;
      message_type() : value(std::numeric_limits<size_t>::max()) {}
      explicit message_type(size_t v) : value(v) {}
      message_type& operator+=(const message_type& other) {
        value = std::min<size_t>(value, other.value);
        return *this;
      }
    };

    typedef int tt_needs_scatter_out_edges;
    typedef int tt_needs_scatter_in_edges;

  private:
    size_t received_labelid;
    bool perform_scatter;

  public:
    Program()
        : received_labelid(std::numeric_limits<size_t>::max()),
          perform_scatter(false) {}

    void init(Graph& graph, GNode node, const message_type& msg) {
      received_labelid = msg.value;
    }

    void apply(Graph& graph, GNode node, const gather_type&) {
      if (received_labelid == std::numeric_limits<size_t>::max()) {
        perform_scatter = true;
      } else if (graph.getData(node, galois::MethodFlag::UNPROTECTED).labelid >
                 received_labelid) {
        perform_scatter = true;
        graph.getData(node, galois::MethodFlag::UNPROTECTED).labelid =
            received_labelid;
      }
    }

    bool needsScatter(Graph& graph, GNode node) { return perform_scatter; }

    void gather(Graph& graph, GNode node, GNode src, GNode dst, gather_type&,
                typename Graph::edge_data_reference) {}

    void scatter(Graph& graph, GNode node, GNode src, GNode dst,
                 galois::graphsLab::Context<Graph, Program>& ctx,
                 typename Graph::edge_data_reference) {
      LNode& data = graph.getData(node, galois::MethodFlag::UNPROTECTED);

      if (node == src &&
          graph.getData(dst, galois::MethodFlag::UNPROTECTED).labelid >
              data.labelid) {
        ctx.push(dst, message_type(data.labelid));
      } else if (node == dst &&
                 graph.getData(src, galois::MethodFlag::UNPROTECTED).labelid >
                     data.labelid) {
        ctx.push(src, message_type(data.labelid));
      }
    }
  };

  template <typename G>
  void readGraph(G& graph) {
    readInOutGraph(graph);
  }

  void operator()(Graph& graph) {
    galois::do_all(graph, Initialize(graph));

    galois::graphsLab::SyncEngine<Graph, Program> engine(graph, Program());
    engine.execute();
  }
};

#endif
