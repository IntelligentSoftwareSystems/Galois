/** Page rank application -*- C++ -*-
* @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Joyce Whang <joyce@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

static llvm::cl::opt<bool> edgePri("edgePri", llvm::cl::desc("Use priority for edges-based"), llvm::cl::init(false));
struct AsyncEdge {
  struct LNode {
    PRTy value;
    std::atomic<PRTy> residual; 
    void init() { value = 1.0 - alpha; residual = 0.0; }
    PRTy getPageRank(int x = 0) { return value; }
    friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
      os << "{PR " << n.value << ", residual " << n.residual << "}";
      return os;
    }
  };

  typedef Galois::Graph::LC_CSR_Graph<LNode,void>::with_numa_alloc<true>::type InnerGraph;
  typedef Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return edgePri? "EdgePri" : "EdgeAsync"; }

  void readGraph(Graph& graph, std::string filename, std::string transposeGraphName) {
    check_types<Graph, InnerGraph>();
    if (transposeGraphName.size()) {
      Galois::Graph::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct sndPri {
    int operator()(const std::pair<GNode, int>& n) const {
      return n.second;
    }
  };

  struct Process {
    Graph& graph;
    PRTy tolerance;
    PRTy amp;

    Process(Graph& g, PRTy t, PRTy a): graph(g), tolerance(t), amp(a) { }

    int pri(PRTy r, int n) const {
      return (int)(r/n * amp);
    }

    void condSched(const GNode& node, LNode& lnode, PRTy delta, Galois::UserContext<GNode>& ctx) const {
      PRTy old = atomicAdd(lnode.residual, delta);
      if (std::fabs(old) <= tolerance && std::fabs(old + delta) >= tolerance)
        ctx.push(node);
    }

    void condSched(const GNode& node, LNode& lnode, PRTy delta, Galois::UserContext<std::pair<GNode, int> >& ctx) const {
      PRTy old = atomicAdd(lnode.residual, delta);
      int out = nout(graph, node, Galois::MethodFlag::NONE) + 1;
      if ((std::fabs(old) <= tolerance && std::fabs(old + delta) >= tolerance) || (pri(old, out) != pri(old+delta, out))) {
        //std::cerr << " " << pri(old+delta) << " ";
        ctx.push(std::make_pair(node, pri(old+delta, out)) );
      }
    }

    template<typename Context>
    void operator()(const std::pair<GNode, int>& data, Context& ctx) const {
      operator()(data.first, ctx);
    }

    template<typename Context>
    void operator()(const GNode& src, Context& ctx) const {
      LNode& sdata = graph.getData(src);      
      Galois::MethodFlag lockflag = Galois::MethodFlag::NONE;

      PRTy oldResidual = sdata.residual.exchange(0.0);
      if (std::fabs(oldResidual) > tolerance) {
        sdata.value = sdata.value + oldResidual;
        int src_nout = nout(graph,src, lockflag);
        PRTy delta = oldResidual*alpha/src_nout;
        // for each out-going neighbors
        for (auto jj = graph.edge_begin(src, lockflag), ej = graph.edge_end(src, lockflag); jj != ej; ++jj) {
          GNode dst = graph.getEdgeDst(jj);
          LNode& ddata = graph.getData(dst, lockflag);
          condSched(dst, ddata, delta, ctx);
        }
      } else { // might need to reschedule self
        condSched(src, sdata, oldResidual, ctx);
      }
    }
  };

  void operator()(Graph& graph, PRTy tolerance, PRTy amp) {
    if (!edgePri) {
      initResidual(graph);
      typedef Galois::WorkList::dChunkedFIFO<16> WL;
      Galois::for_each_local(graph, Process(graph, tolerance, amp), Galois::wl<WL>());
    } else {
      Galois::InsertBag<std::pair<GNode, int>> b;
      initResidual(graph, b, [amp] (Graph& graph, const GNode& node) {
          return (int)(graph.getData(node).residual * amp);
        });
      typedef Galois::WorkList::dChunkedFIFO<128> WL;
      typedef Galois::WorkList::OrderedByIntegerMetric<sndPri,WL> OBIM;
      Galois::for_each_local(b, Process(graph, tolerance, amp), Galois::wl<OBIM>());
    }
  }

  void verify(Graph& graph, PRTy tolerance) {
    verifyInOut(graph, tolerance);
  }
};

