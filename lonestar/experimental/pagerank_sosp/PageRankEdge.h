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

//#include "galois/WorkList/WorkListDebug.h"

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

  typedef galois::graphs::LC_CSR_Graph<LNode,void>::with_numa_alloc<true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return edgePri? "EdgePri" : "EdgeAsync"; }

  void readGraph(Graph& graph, std::string filename, std::string transposeGraphName) {
    check_types<Graph, InnerGraph>();
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName); 
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

  static int pri(PRTy r, int n, PRTy amp, PRTy tolerance) {
    auto base = r/(n*tolerance);
    int i = (int)base; //(int)sqrt(base*100+1); //(int)log(1 + base * -1 * amp);
    //i = __builtin_clz(i+1);
    if (i > 50)
      return -50;
    return -i;
  }


  struct Process {
    Graph& graph;
    PRTy tolerance;
    PRTy amp;

    Process(Graph& g, PRTy t, PRTy a): graph(g), tolerance(t), amp(a) { }

    void condSched(const GNode& node, LNode& lnode, PRTy delta, galois::UserContext<GNode>& ctx) const {
      PRTy old = atomicAdd(lnode.residual, delta);
      if (std::fabs(old) <= tolerance && std::fabs(old + delta) >= tolerance)
        ctx.push(node);
    }

    void condSched(const GNode& node, LNode& lnode, PRTy delta, galois::UserContext<std::pair<GNode, int> >& ctx) const {
      PRTy old = atomicAdd(lnode.residual, delta);
      int out = nout(graph, node, galois::MethodFlag::UNPROTECTED) + 1;
      auto oldp = pri(old, out, amp, tolerance);
      auto newp = pri(old+delta, out, amp, tolerance);
      if ((std::fabs(old) <= tolerance && std::fabs(old + delta) >= tolerance) || (oldp != newp)) {
        ctx.push(std::make_pair(node, pri(old+delta, out, amp, tolerance)) );
      }
    }

    template<typename Context>
    void operator()(const std::pair<GNode, int>& data, Context& ctx) const {
      GNode node = data.first;
      LNode& sdata = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      int out = nout(graph, node, galois::MethodFlag::UNPROTECTED) + 1;
      if (sdata.residual < tolerance ||
          pri(sdata.residual, out, amp, tolerance) < data.second)
        return;
      // if (data.second < -500) 
      //   std::cout << data.first << "," << data.second << "," << nout(graph, node, galois::MethodFlag::UNPROTECTED) << "," << ninout(graph, node, galois::MethodFlag::UNPROTECTED) << "," << sdata.value << "," << sdata.residual << "\n";

      operator()(data.first, ctx);
    }

    template<typename Context>
    void operator()(const GNode& src, Context& ctx) const {
      LNode& sdata = graph.getData(src);      
      galois::MethodFlag lockflag = galois::MethodFlag::UNPROTECTED;

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
    initResidual(graph);
    if (!edgePri) {
      typedef galois::worklists::dChunkedFIFO<256> WL;
      galois::for_each_local(graph, Process(graph, tolerance, amp), galois::wl<WL>());
    } else {
      typedef galois::worklists::dChunkedFIFO<32> WL;
      //typedef galois::worklists::AltChunkedFIFO<32> WL;
      typedef galois::worklists::OrderedByIntegerMetric<sndPri,WL>::with_block_period<8>::type OBIM;
      //typedef galois::worklists::WorkListTracker<sndPri,OBIM> DOBIM;
      auto fn = [&graph, amp, tolerance] (const GNode& node) {
        int out = nout(graph, node, galois::MethodFlag::UNPROTECTED) + 1;
        return std::make_pair(node, pri(graph.getData(node, galois::MethodFlag::UNPROTECTED).residual, out,amp, tolerance));
      };
      galois::for_each(boost::make_transform_iterator(graph.begin(), std::ref(fn)),
                       boost::make_transform_iterator(graph.end(), std::ref(fn)),
                       Process(graph, tolerance, amp), galois::wl<OBIM>());
    }
  }

  void verify(Graph& graph, PRTy tolerance) {
    verifyInOut(graph, tolerance);
  }
};


struct AsyncEdgePriSet {
  struct LNode {
    PRTy value;
    std::atomic<PRTy> residual; 
    std::atomic<int> inWL;
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

  std::string name() const { return "EdgePriSet"; }

  void readGraph(Graph& graph, std::string filename, std::string transposeGraphName) {
    check_types<Graph, InnerGraph>();
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
    galois::substrate::PerThreadStorage<int>& Pstats;
    PRTy limit;

    Process(Graph& g, PRTy t, galois::InsertBag<GNode>& wl, galois::substrate::PerThreadStorage<galois::OnlineStat>& s, galois::substrate::PerThreadStorage<int>& p, PRTy l): graph(g), tolerance(t), nextWL(wl), stats(s), Pstats(p), limit(l) { }

    void operator()(const GNode& src) const {
      LNode& sdata = graph.getData(src);
      sdata.inWL = 0;

      auto resScale = nout(graph, src, galois::MethodFlag::UNPROTECTED) + 1;
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

      //++*Pstats.getLocal();

      galois::MethodFlag lockflag = galois::MethodFlag::UNPROTECTED;

      PRTy oldResidual = sdata.residual.exchange(0.0);
      sdata.value = sdata.value + oldResidual;
      int src_nout = nout(graph,src, lockflag);
      PRTy delta = oldResidual*alpha/src_nout;
      // for each out-going neighbors
      for (auto jj = graph.edge_begin(src, lockflag), ej = graph.edge_end(src, lockflag); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        LNode& ddata = graph.getData(dst, lockflag);
        PRTy old = atomicAdd(ddata.residual, delta);
        // if the node is not in the worklist and the residual is greater than tolerance
        if (old + delta >= tolerance && !ddata.inWL) {
          if (0 ==ddata.inWL.exchange(1)) {
            nextWL.push(dst);
            auto rs = nout(graph, dst, galois::MethodFlag::UNPROTECTED) + 1;
            stats.getLocal()->insert(old+delta / rs);
          }
        }
      }
    }

    void condSched(const GNode& node, LNode& lnode, PRTy delta, galois::UserContext<GNode>& ctx) const {
      PRTy old = atomicAdd(lnode.residual, delta);
      if (std::fabs(old) <= tolerance && std::fabs(old + delta) >= tolerance)
        ctx.push(node);
    }
    
    template<typename Context>
    void operator()(const GNode& src, Context& ctx) const {
      LNode& sdata = graph.getData(src);      
      galois::MethodFlag lockflag = galois::MethodFlag::UNPROTECTED;
      
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
    initResidual(graph);

    galois::InsertBag<GNode> curWL;
    galois::InsertBag<GNode> nextWL;
    galois::substrate::PerThreadStorage<galois::OnlineStat> stats;
    galois::substrate::PerThreadStorage<int> Pstats;
    double oldlimit = 1000.0;
    int round = 0;
    unsigned long long totaldid = 0;

    //First do all the nodes once
    galois::do_all_local(graph, Process(graph, tolerance, nextWL, stats, Pstats, 0.0), galois::do_all_steal<true>());

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
          sdev += s->getStdDeviation() * s->getCount();
          stats.getRemote(i)->reset();
          ++nonzero;
        }
      }
      limit /= count;
      sdev /= count;
      // if (limit > oldlimit) {
      //   limit = oldlimit;
      // } else {
      //   oldlimit = limit;
      // }

      limit /= 2;
      //      limit += sdev / nonzero;

      // int total = 0;
      // for (int i = 0; i < Pstats.size(); ++i) {
      //   total += *Pstats.getRemote(i);
      //   *Pstats.getRemote(i) = 0;
      // }

      if (count < 50000) {
        //limit = 0.0;
        galois::for_each_local(curWL, Process(graph, tolerance, nextWL, stats, Pstats, limit));
        return;
      }
      // std::cout << round << " Count is " << count << " next limit is " << limit << " max is " << max << " std " << sdev << " did " << total << "\n";
      // totaldid += total;
      // ++round;
      galois::do_all_local(curWL, Process(graph, tolerance, nextWL, stats, Pstats, limit), galois::do_all_steal<true>());
    }
    std::cout << "Did " << totaldid << " (in rounds)\n";
  }

  void verify(Graph& graph, PRTy tolerance) {
    verifyInOut(graph, tolerance);
  }
};

