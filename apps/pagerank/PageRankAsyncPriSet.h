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

struct AsyncPriSet{
  struct LNode {
    PRTy value;
    std::atomic<PRTy> residual; // tracking residual
    std::atomic<bool> inWL; // tracking wl ocupancy
    void init() { value = 1.0 - alpha; residual = 0.0; inWL = true; }
    PRTy getPageRank(int x = 0) { return value; }
    friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
      os << "{PR " << n.value << ", residual " << n.residual << ", inWL " << n.inWL << "}";
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
    Galois::InsertBag<GNode>& nextWL;
    Galois::Runtime::PerThreadStorage<Galois::OnlineStat>& stats;
    PRTy limit;
    

    Process(Graph& g, PRTy t, Galois::InsertBag<GNode>& wl, Galois::Runtime::PerThreadStorage<Galois::OnlineStat>& s, PRTy l): graph(g), tolerance(t), nextWL(wl), stats(s), limit(l) { }

    //    void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) const {
    void operator()(const GNode& src) const {
      LNode& sdata = graph.getData(src);
      auto src_nout = nout(graph, src, Galois::MethodFlag::NONE);
      bool leaf = false;
      if (src_nout == 0) {
        leaf = true;
        src_nout = 1;
      }
      if ( sdata.residual / src_nout < limit) {
        nextWL.push(src);
        double R = sdata.residual;
        R /= src_nout;
        stats.getLocal()->insert(std::max(0.0, R));
        return;
      }

      Galois::MethodFlag lockflag = Galois::MethodFlag::NONE;

      // the node is processed
      sdata.inWL = false;
      PRTy oldResidual = sdata.residual;
      PRTy sum = computePageRankInOut(graph, src, 0, lockflag);
      PRTy value = alpha*sum + (1.0 - alpha);
      PRTy diff = std::fabs(value - sdata.value);

      if (diff > tolerance || leaf) {
        atomicAdd(sdata.residual, -oldResidual);
        sdata.value = value;

        // for each out-going neighbors
        for (auto jj = graph.edge_begin(src, lockflag), ej = graph.edge_end(src, lockflag);
             jj != ej; ++jj) {
          GNode dst = graph.getEdgeDst(jj);
	  LNode& ddata = graph.getData(dst, lockflag);
          PRTy delta = diff*alpha/src_nout;
          PRTy old = atomicAdd(ddata.residual, delta);
	  // if the node is not in the worklist and the residual is greater than tolerance
	  if (old + delta > tolerance && !ddata.inWL) {
            bool already = ddata.inWL.exchange(true);
            if (!already)
              nextWL.push(dst);
          }
        }
      }
      double R = sdata.residual;
      R /= src_nout;
      stats.getLocal()->insert(std::max(0.0, R)); // not strictly true
    }
  };

  void operator()(Graph& graph, PRTy tolerance, PRTy amp) {

    Galois::do_all_local(graph, InitResidual(graph), Galois::loopname("InitResidual"));

    Galois::InsertBag<GNode> curWL;
    Galois::InsertBag<GNode> nextWL;
    Galois::Runtime::PerThreadStorage<Galois::OnlineStat> stats;

    //First do all the nodes once
    Galois::do_all_local(graph, Process(graph, tolerance, nextWL, stats, 0.0), Galois::do_all_steal<true>());

    while (!nextWL.empty()) {
      curWL.swap(nextWL);
      nextWL.clear();

      double limit = 0.0;
      int count = 0;
      for (int i = 0; i < stats.size(); ++i) {
        if (stats.getRemote(i)->getCount()) {
          //std::cout << *stats.getRemote(i) << "\n";
          count += stats.getRemote(i)->getCount();
          limit += stats.getRemote(i)->getMean() * stats.getRemote(i)->getCount();
          stats.getRemote(i)->reset();
        }
      }
      limit /= count;
      if (count < 1000)
        limit = 0.0;
      std::cout << "Count is " << count << " next limit is " << limit << "\n";
      Galois::do_all_local(curWL, Process(graph, tolerance, nextWL, stats, limit), Galois::do_all_steal<true>());
    }
  }

  void verify(Graph& graph, PRTy tolerance) {    
    verifyInOut(graph, tolerance);
  }
};

