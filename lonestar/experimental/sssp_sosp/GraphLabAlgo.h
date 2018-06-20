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

#ifndef APPS_SSSP_GRAPHLABALGO_H
#define APPS_SSSP_GRAPHLABALGO_H

#include "galois/DomainSpecificExecutors.h"
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/GraphNodeBag.h"

#include <boost/mpl/if.hpp>

#include "SSSP.h"

struct GraphLabAlgo {
  typedef galois::graphs::LC_CSR_Graph<SNode, uint32_t>::with_no_lockable<
      true>::type ::with_numa_alloc<true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "GraphLab"; }

  void readGraph(Graph& graph) { readInOutGraph(graph); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      g.getData(n).dist = DIST_INFINITY;
    }
  };

  struct Program {
    Dist min_dist;
    bool changed;

    struct gather_type {};
    typedef int tt_needs_scatter_out_edges;

    struct message_type {
      Dist dist;
      message_type(Dist d = DIST_INFINITY) : dist(d) {}

      message_type& operator+=(const message_type& other) {
        dist = std::min(dist, other.dist);
        return *this;
      }
    };

    void init(Graph& graph, GNode node, const message_type& msg) {
      min_dist = msg.dist;
    }

    void apply(Graph& graph, GNode node, const gather_type&) {
      changed     = false;
      SNode& data = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      if (data.dist > min_dist) {
        changed   = true;
        data.dist = min_dist;
      }
    }

    bool needsScatter(Graph& graph, GNode node) { return changed; }

    void scatter(Graph& graph, GNode node, GNode src, GNode dst,
                 galois::graphsLab::Context<Graph, Program>& ctx,
                 Graph::edge_data_reference edgeValue) {
      SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      SNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      Dist newDist = sdata.dist + edgeValue;
      if (ddata.dist > newDist) {
        ctx.push(dst, message_type(newDist));
      }
    }

    void gather(Graph& graph, GNode node, GNode src, GNode dst, gather_type&,
                Graph::edge_data_reference) {}
  };

  void operator()(Graph& graph, const GNode& source) {
    galois::graphsLab::SyncEngine<Graph, Program> engine(graph, Program());
    engine.signal(source, Program::message_type(0));
    engine.execute();
  }
};

#endif
