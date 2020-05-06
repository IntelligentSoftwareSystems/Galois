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

// Basic Pagerank algorithm that uses a worklist (without priorities for now).

struct PagerankDelta {
  struct LNode {
    float value;
    unsigned int nout;
    LNode() : value(1.0), nout(0) {}
    float getPageRank() { return value; }
  };

  typedef
      typename galois::graphs::LC_CSR_Graph<LNode,
                                            void>::with_numa_alloc<true>::type
          //    ::with_no_lockable<true>::type
          InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;

  std::string name() const { return "PagerankDelta"; }

  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.value  = 1.0;
      int outs = std::distance(g.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                               g.edge_end(n, galois::MethodFlag::UNPROTECTED));
      data.nout = outs;
    }
  };

  struct Process {
    PagerankDelta* self;
    Graph& graph;

    Process(PagerankDelta* s, Graph& g) : self(s), graph(g) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) {
      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, galois::MethodFlag::UNPROTECTED),
                ej = graph.in_edge_end(src, galois::MethodFlag::UNPROTECTED);
           jj != ej; ++jj) {
        GNode dst    = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::WRITE);
        sum += ddata.value / ddata.nout;
      }
      float value  = (1.0 - alpha) * sum + alpha;
      LNode& sdata = graph.getData(src, galois::MethodFlag::WRITE);
      float diff   = std::fabs(value - sdata.value);
      if (diff > tolerance) {
        sdata.value = value;
        for (auto jj = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
                  ej = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
             jj != ej; ++jj) {
          GNode dst = graph.getEdgeDst(jj);
          ctx.push(dst);
        }
      }
    }
  };

  void operator()(Graph& graph) {
    typedef galois::worklists::PerSocketChunkFIFO<512> WL;
    galois::for_each(graph, Process(this, graph), galois::wl<WL>());
  }
};
