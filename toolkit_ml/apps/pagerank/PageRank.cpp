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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Joyce Whang <joyce@cs.utexas.edu>
 */

#include "galois/config.h"
#include "galois/Galois.h"
#include "galois/Accumulator.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/Graph/LCGraph.h"
#include "galois/Graph/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"

#ifdef GALOIS_USE_EXP
#include "galois/worklists/WorkListDebug.h"
#endif

#include GALOIS_CXX11_STD_HEADER(atomic)
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>

#include "PageRank.h"

namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin";
static const char* url = 0;

enum Algo {
  pull,
  pull2,
  serial,
  sync_pr, // Joyce
  prt_rsd, // Joyce
  async // Joyce
};

cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Input graph is symmetric"));
static cll::opt<std::string> outputPullFilename("outputPull", cll::desc("Precompute data for Pull algorithm to file"));
cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(100000000));
cll::opt<unsigned int> memoryLimit("memoryLimit",
    cll::desc("Memory limit for out-of-core algorithms (in MB)"), cll::init(~0U));
static cll::opt<int> amp("amp", cll::desc("amp for priority"), cll::init(-100));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::pull, "pull", "Use precomputed data perform pull-based algorithm"),
      clEnumValN(Algo::pull2, "pull2", "Use pull-based algorithm"),
      clEnumValN(Algo::serial, "serial", "Compute PageRank in serial"),
      clEnumValN(Algo::sync_pr, "sync_pr", "Synchronous version..."),
      clEnumValN(Algo::prt_rsd, "prt_rsd", "Prioritized (degree biased residual) version..."),
      clEnumValN(Algo::async, "async", "Asynchronous without priority version..."),
      clEnumValEnd), cll::init(Algo::pull));

struct SerialAlgo {
  typedef galois::graphs::LC_CSR_Graph<PNode,void>
    ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Serial"; }
  
  void readGraph(Graph& graph) { galois::graphs::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      g.getData(n).value = 1.0;
      g.getData(n).accum.write(0.0);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;
    unsigned int numNodes = graph.size();

    while (true) {
      float max_delta = std::numeric_limits<float>::min();
      unsigned int small_delta = 0;
      double sum_delta = 0;

      for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
        GNode src = *ii;
        PNode& sdata = graph.getData(src);
        int neighbors = std::distance(graph.edge_begin(src), graph.edge_end(src));
        for (auto jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
          GNode dst = graph.getEdgeDst(jj);
          PNode& ddata = graph.getData(dst);
          float delta =  sdata.value / neighbors;
          ddata.accum.write(ddata.accum.read() + delta);
        }
      }

      for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
        GNode src = *ii;
        PNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
        float value = (1.0 - alpha) * sdata.accum.read() + alpha;
        float diff = std::fabs(value - sdata.value);
        if (diff <= tolerance)
          ++small_delta;
        if (diff > max_delta)
          max_delta = diff;
        sum_delta += diff;
        sdata.value = value;
        sdata.accum.write(0);
      }

      iteration += 1;

      std::cout << "iteration: " << iteration
                << " sum delta: " << sum_delta
                << " max delta: " << max_delta
                << " small delta: " << small_delta
                << " (" << small_delta / (float) numNodes << ")"
                << "\n";

      if (max_delta <= tolerance || iteration >= maxIterations)
        break;
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }
  }
};

struct PullAlgo {
  struct LNode {
    float value[2];

    float getPageRank() { return value[1]; }
    float getPageRank(unsigned int it) { return value[it & 1]; }
    void setPageRank(unsigned it, float v) { value[(it+1) & 1] = v; }
  };
  typedef galois::graphs::LC_InlineEdge_Graph<LNode,float>
    ::with_compressed_node_ptr<true>::type
    ::with_no_lockable<true>::type
    ::with_numa_alloc<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Pull"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<size_t> small_delta;
  galois::GAccumulator<float> sum_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.value[0] = 1.0;
      data.value[1] = 1.0;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.value[1] = data.value[0];
    }
  };

  struct Process {
    PullAlgo* self;
    Graph& graph;
    unsigned int iteration;

    Process(PullAlgo* s, Graph& g, unsigned int i): self(s), graph(g), iteration(i) { }

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
      operator()(src);
    }

    void operator()(const GNode& src) const {
      LNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      double sum = 0;

      for (auto jj = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), ej = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        float w = graph.getEdgeData(jj);

        LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        sum += ddata.getPageRank(iteration) * w;
      }

      float value = sum * (1.0 - alpha) + alpha;
      float diff = std::fabs(value - sdata.getPageRank(iteration));
       
      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
      self->sum_delta.update(diff);
      sdata.setPageRank(iteration, value);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;
    
    while (true) {
      galois::for_each(graph, Process(this, graph, iteration));
      iteration += 1;

      float delta = max_delta.reduce();
      size_t sdelta = small_delta.reduce();

      std::cout << "iteration: " << iteration
                << " sum delta: " << sum_delta.reduce()
                << " max delta: " << delta
                << " small delta: " << sdelta
                << " (" << sdelta / (float) graph.size() << ")"
                << "\n";

      if (delta <= tolerance || iteration >= maxIterations)
        break;
      max_delta.reset();
      small_delta.reset();
      sum_delta.reset();
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }
  }
};

struct PullAlgo2 {
  struct LNode {
    float value[2];
    unsigned int nout;
    float getPageRank() { return value[1]; }
    float getPageRank(unsigned int it) { return value[it & 1]; }
    void setPageRank(unsigned it, float v) { value[(it+1) & 1] = v; }
  };

  typedef typename galois::graphs::LC_InlineEdge_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type
    InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;

  std::string name() const { return "Pull2"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<unsigned int> small_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.value[0] = 1.0;
      data.value[1] = 1.0;
      int outs = std::distance(g.edge_begin(n, galois::MethodFlag::UNPROTECTED), g.edge_end(n, galois::MethodFlag::UNPROTECTED));
      data.nout = outs;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.value[1] = data.value[0];
    }
  };

  struct Process {
    PullAlgo2* self;
    Graph& graph;
    unsigned int iteration;

    Process(PullAlgo2* s, Graph& g, unsigned int i): self(s), graph(g), iteration(i) { }

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
      operator()(src);
    }

    void operator()(const GNode& src) const {

      LNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, galois::MethodFlag::UNPROTECTED), ej = graph.in_edge_end(src, galois::MethodFlag::UNPROTECTED);
          jj != ej; ++jj) {
        GNode dst = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        sum += ddata.getPageRank(iteration) / ddata.nout;
      }

      float value = (1.0 - alpha) * sum + alpha;
      float diff = std::fabs(value - sdata.getPageRank(iteration));
      sdata.setPageRank(iteration, value);
      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;

    while (true) {
      galois::for_each(graph, Process(this, graph, iteration));
      iteration += 1;

      float delta = max_delta.reduce();
      size_t sdelta = small_delta.reduce();

      std::cout << "iteration: " << iteration
        << " max delta: " << delta
        << " small delta: " << sdelta
        << " (" << sdelta / (float) graph.size() << ")"
        << "\n";

      if (delta <= tolerance || iteration >= maxIterations) {
        break;
      }
      max_delta.reset();
      small_delta.reset();
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }
  }
};










/* ------------------------- Joyce's codes start ------------------------- */
//---------- parallel synchronous algorithm (original copy: PullAlgo2, readGraph is re-written.)

int idcount = 0;

struct Sync {
  struct LNode {
    float value[2];
    int id;
    unsigned int nout;
    float getPageRank() { return value[1]; }
    float getPageRank(unsigned int it) { return value[it & 1]; }
    void setPageRank(unsigned it, float v) { value[(it+1) & 1] = v; }
  };

  typedef galois::graphs::LC_CSR_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  /*
  typedef galois::graphs::LC_InlineEdge_Graph<LNode,void>
    :: with_numa_alloc<true>::type
    InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;
  */

  /*
  typedef typename galois::graphs::LC_InlineEdge_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type
    InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;
  */

  std::string name() const { return "Sync"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<unsigned int> small_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n);
      //data.value[0] = 1.0;
      //data.value[1] = 1.0;
      data.value[0] = (1.0 - alpha2);
      data.value[1] = (1.0 - alpha2);
      int outs = std::distance(g.edge_begin(n), g.edge_end(n));
      data.nout = outs;
      data.id = idcount++;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n);
      data.value[1] = data.value[0];
    }
  };

  struct Process {
    Sync* self;
    Graph& graph;
    unsigned int iteration;

    Process(Sync* s, Graph& g, unsigned int i): self(s), graph(g), iteration(i) { }

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
      operator()(src);
    }

    void operator()(const GNode& src) const {

      LNode& sdata = graph.getData(src);

      //std::cout<<sdata.id<<" picked up...\n";
      double sum = 0;
      for (auto jj = graph.in_edge_begin(src), ej = graph.in_edge_end(src);
          jj != ej; ++jj) {
        GNode dst = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst);
        sum += ddata.getPageRank(iteration) / ddata.nout;
	//std::cout<<"- id: "<<ddata.id<<"\n";
      }

      float value = (1.0 - alpha) * sum + alpha;
      float diff = std::fabs(value - sdata.getPageRank(iteration));
      sdata.setPageRank(iteration, value);
      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;
    auto numNodes = graph.size();
    while (true) {
      galois::do_all(graph, Process(this, graph, iteration));
      iteration += 1;

      float delta = max_delta.reduce();
      size_t sdelta = small_delta.reduce();

      std::cout << "iteration: " << iteration
        << " max delta: " << delta
        << " small delta: " << sdelta
        << " (" << sdelta / numNodes << ")"
        << "\n";

      if (delta <= tolerance || iteration >= maxIterations) {
        break;
      }
      max_delta.reset();
      small_delta.reset();
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      galois::do_all(graph, Copy(graph));
    }
  }
};

//---------- parallel prioritized asynchronous algorithm (max. residual)
template<typename NTy>
int pri(const NTy& n) {
  double d = n.residual / n.deg;
  return (int)(d * amp); //d > 0.1 ? 0 : 1;//-1*(int)sqrt(-1*d*amp);
}

struct PrtRsd {

  struct LNode {
    float pagerank;
    int id;
    float residual;
    bool flag; // tracking if it is in the worklist
    unsigned int nout;
    unsigned int deg;
    float getPageRank() { return pagerank; }
    float getResidual() { return residual; }
  };
  
  /*
  typedef galois::graphs::LC_InlineEdge_Graph<LNode,void>
    :: with_numa_alloc<true>::type
    InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;
  */ 
 
  typedef galois::graphs::LC_CSR_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "PrtRsd"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<unsigned int> small_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    int id=0;
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n);
      data.pagerank = (1.0 - alpha2);
      data.residual = 0.0;
      data.id = idcount++;
      int outs = std::distance(g.edge_begin(n), g.edge_end(n));
      data.nout = outs;
      int ins = std::distance(g.in_edge_begin(n), g.in_edge_end(n));
      data.deg = outs + ins;
    }
  };

  struct Process1 {
    PrtRsd* self;
    Graph& graph;
     
    Process1(PrtRsd* s, Graph& g): self(s), graph(g) { }

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
      operator()(src);
    }

    void operator()(const GNode& src) const {
      LNode& data = graph.getData(src);
      // for each out-going neighbour, add residuals
      for (auto jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj){
        GNode dst = graph.getInEdgeDst(jj);
	LNode& ddata = graph.getData(dst);
	ddata.residual = (float) ddata.residual + (float) 1/data.nout;   
      }
    }
  }; //--- end of Process1

  struct Process2 {
    PrtRsd* self;
    Graph& graph;
     
    Process2(PrtRsd* s, Graph& g): self(s), graph(g) { }

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
      operator()(src);
    }

    void operator()(const GNode& src) const {
      // scale the residual 
      LNode& data = graph.getData(src);
      data.residual = alpha2*(1-alpha2)*data.residual;
    }
  }; //--- end of Process2

  struct Process4 {
    PrtRsd* self;
    Graph& graph;
     
    Process4(PrtRsd* s, Graph& g): self(s), graph(g) { }

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
      operator()(src);
    }

    void operator()(const GNode& src) const {
      // scale the residual 
      LNode& data = graph.getData(src);
      std::cout<<data.residual<<" ";
    }
  }; //--- end of Process4

  // define priority
  typedef std::pair<int, GNode> UpdateRequest;
  struct UpdateRequestIndexer: public std::unary_function<UpdateRequest, int> {
  int operator() (const UpdateRequest& val) const {
    return val.first;
    }
  };

  struct Process3 {
    PrtRsd* self;
    Graph& graph;
     
    galois::Statistic& pre;
    galois::Statistic& post;
     
    Process3(PrtRsd* s, Graph& g, galois::Statistic& _pre, galois::Statistic& _post): self(s), graph(g), pre(_pre), post(_post) { }

    void operator()(const UpdateRequest& srcRq, galois::UserContext<UpdateRequest>& ctx) const {
      GNode src = srcRq.second;
      
      LNode* node = &graph.getData(src); //, galois::MethodFlag::UNPROTECTED);
      if (node->residual < tolerance || pri(*node) != srcRq.first){ // degree biased residual
	post +=1;
        return;
      }
      //if (total<1000)
	//std::cout<<node->id<<" picked up... residual: "<<node->residual<<", priority: "<<srcRq.first<<", rsd/deg: "<<(node->residual/node->deg)<<"\n";
      
      node = &graph.getData(src);

      //galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
      galois::MethodFlag flag = galois::MethodFlag::WRITE;

      // update pagerank (consider each in-coming edge)
      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, flag), ej = graph.in_edge_end(src, flag); jj != ej; ++jj) {
        GNode dst = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, flag);
        sum += ddata.getPageRank() / ddata.nout;
      }

      unsigned nopush = 0;
     
      // update residual (consider each out-going edge)
      for (auto jj = graph.edge_begin(src, flag), ej = graph.edge_end(src, flag); jj != ej; ++jj){
        GNode dst = graph.getEdgeDst(jj); 
        LNode& ddata = graph.getData(dst, flag);
	float oldR = ddata.residual; // degree biased residual
        int oldP = pri(ddata);
        ddata.residual += node->residual*alpha2/node->nout;
	if (ddata.residual >= tolerance && 
            (oldP != pri(ddata) || (oldR <= tolerance))) {
	  ctx.push(std::make_pair(pri(ddata), dst)); // degree biased
	} else {
          ++nopush;
        }
      }
      if (nopush)
        pre += nopush;

      node->pagerank = alpha2*sum + (1-alpha2);
      node->residual = 0.0;

    }

  }; //--- end of Process3

  void operator()(Graph& graph) {
 galois::Statistic pre("PrePrune");
    galois::Statistic post("PostPrune");
    galois::do_all(graph, Process1(this, graph), galois::loopname("P1"));
    galois::do_all(graph, Process2(this, graph), galois::loopname("P2")); 
    std::cout<<"tolerance: "<<tolerance<<", amp: "<<amp<<"\n";
    using namespace galois::worklists;
    typedef dChunkedFIFO<16> dChunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk>::with_block_period<4>::type OBIM;
#ifdef GALOIS_USE_EXP
    typedef WorkListTracker<UpdateRequestIndexer, OBIM> dOBIM;
#else
    typedef OBIM dOBIM;
#endif
    galois::InsertBag<UpdateRequest> initialWL;
    galois::do_all(graph, [&initialWL, &graph] (GNode src) {
	LNode& data = graph.getData(src);
        if(data.residual>=tolerance){
	  initialWL.push_back(std::make_pair(pri(data), src)); // degree biased
        }
      });
    galois::StatTimer T("InnerTime");
    T.start();
    galois::for_each(initialWL, Process3(this, graph, pre, post), galois::wl<OBIM>(), galois::loopname("mainloop"));   
    T.stop();

    for(auto N : graph) {
      auto& data = graph.getData(N);
      if (data.residual > tolerance) {
        std::cout << N 
                  << " id " << data.id
                  << " residual " << data.residual
                  << " pr " << data.pagerank
                  << " nout " << data.nout
                  << " deg " << data.deg
                  << "\n";
      }
    }
        
    //std::cout<<"print residuals\n";
    //galois::for_each(graph, Process4(this, graph));
    //std::cout<<"\n";
  } 

};

//---------- asynchronous without priority (original copy: PagerankDelta, readGraph is re-written.)
struct Async {
  struct LNode {
    float value;
    unsigned int nout;
    bool flag; // tracking if it is in the worklist
    LNode(): value(1.0), nout(0) {}
    float getPageRank() { return value; }
  };

  typedef galois::graphs::LC_CSR_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  /*
  typedef galois::graphs::LC_InlineEdge_Graph<LNode,void>
    :: with_numa_alloc<true>::type
    InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;
  */

  std::string name() const { return "Async"; }

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n);
      data.value = (1.0 - alpha2);
      int outs = std::distance(g.edge_begin(n), g.edge_end(n));
      data.nout = outs;
      data.flag = true;
    }
  };
  
  struct Process {
    Graph& graph;
     
    Process(Graph& g): graph(g) { }

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
      LNode& sdata = graph.getData(src);
      // the node is processed
      sdata.flag = false;

      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, galois::MethodFlag::UNPROTECTED), ej = graph.in_edge_end(src, galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
        GNode dst = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        sum += ddata.value / ddata.nout;
      }
      float value = (1.0 - alpha) * sum + alpha;
      float diff = std::fabs(value - sdata.value);
      if (diff > tolerance) {
        sdata.value = value;
        for (auto jj = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), ej = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
          GNode dst = graph.getEdgeDst(jj);
	  LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
	  // if the node is not in the worklist, then push
	  if(!ddata.flag) {
	    ddata.flag = true;
            ctx.push(dst);
	  } else {
	    //skip++;
	  }
        }
      } 
      
      //if(total%1000==0)
	//std::cout<<"total: "<<total<<", skip: "<<skip<<", diff: "<<diff<<"\n";
    }

  };

  void operator()(Graph& graph) {
    //    skip = 0;
    typedef galois::worklists::dChunkedFIFO<16> WL;
    galois::for_each(graph, Process(graph), galois::wl<WL>());
  }
};

/* ------------------------- Joyce's codes end ------------------------- */










//! Transpose in-edges to out-edges
static void precomputePullData() {
  typedef galois::graphs::LC_CSR_Graph<size_t, void>
    ::with_no_lockable<true>::type InputGraph;
  typedef InputGraph::GraphNode InputNode;
  typedef galois::graphs::FileGraphWriter OutputGraph;
  //typedef OutputGraph::GraphNode OutputNode;

  InputGraph input;
  OutputGraph output;
  galois::graphs::readGraph(input, filename); 

  size_t node_id = 0;
  for (auto ii = input.begin(), ei = input.end(); ii != ei; ++ii) {
    InputNode src = *ii;
    input.getData(src) = node_id++;
  }

  output.setNumNodes(input.size());
  output.setNumEdges(input.sizeEdges());
  output.setSizeofEdgeData(sizeof(float));
  output.phase1();

  for (auto ii = input.begin(), ei = input.end(); ii != ei; ++ii) {
    InputNode src = *ii;
    size_t sid = input.getData(src);
    assert(sid < input.size());

    //size_t num_neighbors = std::distance(input.edge_begin(src), input.edge_end(src));

    for (auto jj = input.edge_begin(src), ej = input.edge_end(src); jj != ej; ++jj) {
      InputNode dst = input.getEdgeDst(jj);
      size_t did = input.getData(dst);
      assert(did < input.size());

      output.incrementDegree(did);
    }
  }

  output.phase2();
  std::vector<float> edgeData;
  edgeData.resize(input.sizeEdges());

  for (auto ii = input.begin(), ei = input.end(); ii != ei; ++ii) {
    InputNode src = *ii;
    size_t sid = input.getData(src);
    assert(sid < input.size());

    size_t num_neighbors = std::distance(input.edge_begin(src), input.edge_end(src));

    float w = 1.0/num_neighbors;
    for (auto jj = input.edge_begin(src), ej = input.edge_end(src); jj != ej; ++jj) {
      InputNode dst = input.getEdgeDst(jj);
      size_t did = input.getData(dst);
      assert(did < input.size());

      size_t idx = output.addNeighbor(did, sid);
      edgeData[idx] = w;
    }
  }

  float* t = output.finish<float>();
  std::uninitialized_copy(std::make_move_iterator(edgeData.begin()), std::make_move_iterator(edgeData.end()), t);
  
  output.toFile(outputPullFilename);
  std::cout << "Wrote " << outputPullFilename << "\n";
}

//! Make values unique
template<typename GNode>
struct TopPair {
  float value;
  GNode id;

  TopPair(float v, GNode i): value(v), id(i) { }

  bool operator<(const TopPair& b) const {
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};

// Joyce modified this function (normalizing PageRank values)
template<typename Graph>
static void printTop(Graph& graph, int topn) {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair,GNode> Top;

  // normalize the PageRank value so that the sum is equal to one
  float sum=0;
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    node_data_reference n = graph.getData(src);
    float value = n.getPageRank();
    sum += value;
  }

  Top top;

  std::cout<<"print PageRank\n";
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    node_data_reference n = graph.getData(src);
    float value = n.getPageRank()/sum; // divide PR by sum
    //float value = n.getPageRank();
    //std::cout<<value<<" "; 
    Pair key(value, src);

    if ((int) top.size() < topn) {
      top.insert(std::make_pair(key, src));
      continue;
    }

    if (top.begin()->first < key) {
      top.erase(top.begin());
      top.insert(std::make_pair(key, src));
    }
  }
  std::cout<<"\nend of print\n";

  int rank = 1;
  std::cout << "Rank PageRank Id\n";
  for (typename Top::reverse_iterator ii = top.rbegin(), ei = top.rend(); ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

template<typename Algo>
void run() {
  typedef typename Algo::Graph Graph;

  Algo algo;
  Graph graph;

  algo.readGraph(graph);

  galois::preAlloc(numThreads + (graph.size() * sizeof(typename Graph::node_data_type)) / galois::runtime::MM::hugePageSize);
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  std::cout << "Target max delta: " << tolerance << "\n";
  T.start();
  galois::do_all(graph, typename Algo::Initialize(graph));
  algo(graph);
  T.stop();
  
  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify)
    printTop(graph, 10);
}

int main(int argc, char **argv) {
  LonestarStart(argc, argv, name, desc, url);
  galois::StatManager statManager;

  if (outputPullFilename.size()) {
    precomputePullData();
    return 0;
  }

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::pull: run<PullAlgo>(); break;
    case Algo::pull2: run<PullAlgo2>(); break;
    case Algo::sync_pr: run<Sync>(); break;
    case Algo::prt_rsd: run<PrtRsd>(); break;
    case Algo::async: run<Async>(); break;
    case Algo::serial: run<SerialAlgo>(); break;
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  return 0;
}
