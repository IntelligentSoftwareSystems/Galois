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

struct AsyncPri{
  struct LNode {
    PRTy value;
    std::atomic<PRTy> residual; // tracking residual
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

  std::string name() const { return "AsyncPri"; }

  void readGraph(Graph& graph, std::string filename, std::string transposeGraphName) {
    if (transposeGraphName.size()) {
      Galois::Graph::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct PRPri {
    Graph& graph;
    PRTy amp;
    static const bool outOnly = true;
    PRPri(Graph& g, PRTy a) : graph(g), amp(a) {}
    int operator()(const GNode& src, PRTy d) const {
      if (outOnly)
        d /= nout(graph, src, Galois::MethodFlag::NONE);
      else
        d /= ninout(graph, src, Galois::MethodFlag::NONE);
      return d*amp; //std::max((int)floor(d*amp), 0);
    }      
    int operator()(const GNode& src) const {
      PRTy d = graph.getData(src, Galois::MethodFlag::NONE).residual;
      return operator()(src, d);
    }
  };

  struct sndPri {
    int operator()(const std::pair<GNode, int>& n) const {
      return n.second;
    }
  };
  
  struct Process {
    Graph& graph;
    PRTy tolerance;
    PRPri pri;

    Process(Graph& g, PRTy t, PRTy a): graph(g), tolerance(t), pri(g,a) { }

    void operator()(const std::pair<GNode,int>& srcn, Galois::UserContext<std::pair<GNode,int>>& ctx) const {
      LNode& sdata = graph.getData(srcn.first);
      
      if(sdata.residual < tolerance || 
         pri(srcn.first) != srcn.second)
        return;

      Galois::MethodFlag lockflag = Galois::MethodFlag::NONE;

      PRTy oldResidual = sdata.residual.exchange(0.0);
      PRTy pr = computePageRankInOut(graph, srcn.first, 0, lockflag);
      PRTy diff = std::fabs(pr - sdata.value);
      sdata.value = pr;
      int src_nout = nout(graph,srcn.first, lockflag);
      PRTy delta = diff*alpha/src_nout;
      // for each out-going neighbors
      for (auto jj = graph.edge_begin(srcn.first, lockflag), 
             ej = graph.edge_end(srcn.first, lockflag);
           jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        LNode& ddata = graph.getData(dst, lockflag);
        PRTy old = atomicAdd(ddata.residual, delta);
        // if the node is not in the worklist and the residual is greater than tolerance
        if(old + delta >= tolerance &&
           (old <= tolerance || pri(dst, old) != pri(dst,old+delta)))
          ctx.push(std::make_pair(dst, pri(dst, old+delta)));
      }
    }
  };

  void operator()(Graph& graph, PRTy tolerance, PRTy amp) {
    initResidual(graph);
    typedef Galois::WorkList::dChunkedFIFO<16> WL;
    typedef Galois::WorkList::OrderedByIntegerMetric<sndPri,WL>::with_block_period<16>::type OBIM;
    Galois::InsertBag<std::pair<GNode, int> > bag;
    PRPri pri(graph, amp);
    Galois::do_all_local(graph, [&graph, &bag, &pri] (const GNode& node) {
        bag.push(std::make_pair(node, pri(node)));
      });
    Galois::for_each_local(bag, Process(graph, tolerance, amp), Galois::wl<OBIM>());
  }

  void verify(Graph& graph, PRTy tolerance) {    
    verifyInOut(graph, tolerance);
  }
};

