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

struct AsyncRsd {
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

  std::string name() const { return "AsyncRsd"; }

  void readGraph(Graph& graph, std::string filename, std::string transposeGraphName) {
    if (transposeGraphName.size()) {
      Galois::Graph::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct InitResidual {
    Graph& graph;
     
    InitResidual(Graph& g): graph(g) { }

    void operator()(const GNode& src) const {
      LNode& data = graph.getData(src);
      // for each in-coming neighbour, add residual
      PRTy sum = 0.0;
      for (auto jj = graph.in_edge_begin(src), ej = graph.in_edge_end(src); jj != ej; ++jj){
        GNode dst = graph.getInEdgeDst(jj);
	LNode& ddata = graph.getData(dst);
	sum += 1.0/nout(graph,dst, Galois::MethodFlag::NONE);  
      }
      data.residual = sum * alpha*(1.0-alpha);
    }
  }; 
  
  struct Process {
    Graph& graph;
    PRTy tolerance;

    Process(Graph& g, PRTy t): graph(g), tolerance(t) { }

    void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) const {
      LNode& sdata = graph.getData(src);
      
      Galois::MethodFlag lockflag = Galois::MethodFlag::NONE;

      PRTy oldResidual = sdata.residual;
      PRTy sum = computePageRankInOut(graph, src, 0, lockflag);
      PRTy value = alpha*sum + (1.0 - alpha);
      PRTy diff = std::fabs(value - sdata.value);

      if (diff > tolerance) {
        // the node is processed
        atomicAdd(sdata.residual, -oldResidual);
        sdata.value = value;
        int src_nout = nout(graph,src, lockflag);
        PRTy delta = diff*alpha/src_nout;
        // for each out-going neighbors
        for (auto jj = graph.edge_begin(src, lockflag), ej = graph.edge_end(src, lockflag);
             jj != ej; ++jj) {
          GNode dst = graph.getEdgeDst(jj);
          LNode& ddata = graph.getData(dst, lockflag);
          if (ddata.residual < tolerance) {
            PRTy old = atomicAdd(ddata.residual, delta);
            // if the node is not in the worklist and the residual is greater than tolerance
            if(old + delta >= tolerance)
              ctx.push(dst);
          }
        }
      }
    }
  };

  void operator()(Graph& graph, PRTy tolerance, PRTy amp) {
    Galois::do_all_local(graph, InitResidual(graph), Galois::loopname("InitResidual"));
    typedef Galois::WorkList::dChunkedFIFO<16> WL;
    Galois::for_each_local(graph, Process(graph, tolerance), Galois::wl<WL>());
  }

  void verify(Graph& graph, PRTy tolerance) {
    verifyInOut(graph, tolerance);
  }
};

