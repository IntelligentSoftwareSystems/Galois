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

#include "Galois/OnlineStats.h"

struct AsyncPriSet{
  struct LNode {
    PRTy value;
    std::atomic<PRTy> residual; 
    std::atomic<int> inWL; // tracking wl occupancy
    void init() { value = 1.0 - alpha; residual = 0.0; inWL = 1; }
    PRTy getPageRank(int x = 0) { return value; }
    friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
      os << "{PR " << n.value << ", residual " << n.residual << ", inWL " << n.inWL << "}";
      return os;
    }
  };

  typedef galois::graphs::LC_CSR_Graph<LNode,void>::with_numa_alloc<true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "AsyncPriSet"; }

  void readGraph(Graph& graph, std::string filename, std::string transposeGraphName) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Process {
    Graph& graph;
    PRTy tolerance;
    galois::InsertBag<GNode>& nextWL;
    galois::substrate::PerThreadStorage<galois::OnlineStat>& stats;
    PRTy limit;

    Process(Graph& g, PRTy t, galois::InsertBag<GNode>& wl, galois::substrate::PerThreadStorage<galois::OnlineStat>& s, PRTy l): graph(g), tolerance(t), nextWL(wl), stats(s), limit(l) { }

    //    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
    void operator()(const GNode& src) const {
      LNode& sdata = graph.getData(src);
      sdata.inWL = 0;

      auto resScale = outOnly ? nout(graph, src, galois::MethodFlag::UNPROTECTED) + 1 : ninout(graph, src, galois::MethodFlag::UNPROTECTED);
      if ( sdata.residual / resScale < limit) {
        double R = sdata.residual;
        if (R >= tolerance) {
          if (0 == sdata.inWL.exchange(1)) {
            nextWL.push(src);
            R /= resScale;
            stats.getLocal()->insert(std::max(0.0, R));
          }
        }
        return;
      }

      galois::MethodFlag lockflag = galois::MethodFlag::UNPROTECTED;

      // the node is processed
      PRTy oldResidual = sdata.residual.exchange(0.0);
      PRTy pr = computePageRankInOut(graph, src, 0, lockflag);
      PRTy diff = std::fabs(pr - sdata.value);
      sdata.value = pr;
      auto src_nout = nout(graph, src, galois::MethodFlag::UNPROTECTED);
      PRTy delta = diff*alpha/src_nout;
      // for each out-going neighbors
      for (auto jj = graph.edge_begin(src, lockflag), ej = graph.edge_end(src, lockflag);
           jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        LNode& ddata = graph.getData(dst, lockflag);
        PRTy old = atomicAdd(ddata.residual, delta);
        // if the node is not in the worklist and the residual is greater than tolerance
        if (old + delta >= tolerance && !ddata.inWL) {
          if (0 ==ddata.inWL.exchange(1)) {
            nextWL.push(dst);
            auto rs = outOnly ? nout(graph, dst, galois::MethodFlag::UNPROTECTED) + 1 : ninout(graph, dst, galois::MethodFlag::UNPROTECTED);
            stats.getLocal()->insert(old+delta / rs);
          }
        }
      }
    }
  };

  void operator()(Graph& graph, PRTy tolerance, PRTy amp) {
    initResidual(graph);

    galois::InsertBag<GNode> curWL;
    galois::InsertBag<GNode> nextWL;
    galois::substrate::PerThreadStorage<galois::OnlineStat> stats;

    //First do all the nodes once
    galois::do_all_local(graph, Process(graph, tolerance, nextWL, stats, 0.0), galois::do_all_steal<true>());

    while (!nextWL.empty()) {
      curWL.swap(nextWL);
      nextWL.clear();

      double limit = 0.0, max = 0.0, sdev = 0.0;
      int count = 0, nonzero = 0;
      for (int i = 0; i < stats.size(); ++i) {
        auto* s = stats.getRemote(i);
        if (s->getCount()) {
          //std::cout << *s << "\n";
          count += s->getCount();
          limit += s->getMean() * s->getCount();
          max = std::max(max, s->getMax());
          sdev += s->getStdDeviation();
          stats.getRemote(i)->reset();
          ++nonzero;
        }
      }
      limit /= count;
      limit /= 2;
      //      limit += sdev / nonzero;
      if (count < 5000)
        limit = 0.0;
      //std::cout << "Count is " << count << " next limit is " << limit << " max is " << max << "\n";
      galois::do_all_local(curWL, Process(graph, tolerance, nextWL, stats, limit), galois::do_all_steal<true>());
    }
  }

  void verify(Graph& graph, PRTy tolerance) {    
    verifyInOut(graph, tolerance);
  }
};

