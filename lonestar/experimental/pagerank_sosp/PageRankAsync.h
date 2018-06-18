/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

struct AsyncSet {
  struct LNode {
    PRTy value;
    std::atomic<int> flag;
    void init() {
      value = 1.0 - alpha;
      flag  = 0;
    }
    PRTy getPageRank(int x = 0) { return value; }
    friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
      os << "{PR " << n.value << ", flag " << n.flag << "}";
      return os;
    }
  };

  typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_numa_alloc<true>::type
      InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "AsyncSet"; }

  void readGraph(Graph& graph, std::string filename,
                 std::string transposeGraphName) {
    check_types<Graph, InnerGraph>();
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Process {
    Graph& graph;
    PRTy tolerance;

    Process(Graph& g, PRTy t) : graph(g), tolerance(t) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
      LNode& sdata                = graph.getData(src);
      sdata.flag                  = 0;
      galois::MethodFlag lockflag = galois::MethodFlag::UNPROTECTED;

      PRTy pr   = computePageRankInOut(graph, src, 0, lockflag);
      PRTy diff = std::fabs(pr - sdata.value);
      if (diff >= tolerance) {
        sdata.value = pr;
        // for each out-going neighbors
        for (auto jj = graph.edge_begin(src, lockflag),
                  ej = graph.edge_end(src, lockflag);
             jj != ej; ++jj) {
          GNode dst    = graph.getEdgeDst(jj);
          LNode& ddata = graph.getData(dst, lockflag);
          // if the node is not in the worklist, then push
          if (!ddata.flag.load(std::memory_order_relaxed))
            if (0 == ddata.flag.exchange(1, std::memory_order_relaxed))
              ctx.push(dst);
        }
      }
    }
  };

  void operator()(Graph& graph, PRTy tolerance, PRTy amp) {
    typedef galois::worklists::PerSocketChunkFIFO<16> WL;
    galois::for_each(graph, Process(graph, tolerance), galois::wl<WL>());
  }

  void verify(Graph& graph, PRTy tolerance) { verifyInOut(graph, tolerance); }
};
