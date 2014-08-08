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

#include "Galois/config.h"
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/Graph/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"

#ifdef GALOIS_USE_EXP
#include "Galois/WorkList/WorkListDebug.h"
#endif

#include GALOIS_CXX11_STD_HEADER(atomic)
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>

#include "PageRank.h"

namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin";
static const char* url = 0;

enum Algo {
  sync_pr,  
  async,
  async_rsd,
  async_prt
};

cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(10000000));
cll::opt<unsigned int> memoryLimit("memoryLimit",
    cll::desc("Memory limit for out-of-core algorithms (in MB)"), cll::init(~0U));
static cll::opt<int> amp("amp", cll::desc("amp for priority"), cll::init(-100));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.01));
static cll::opt<std::string> algo_str("algo_str", cll::desc("algo_str"), cll::init("NA"));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::sync_pr, "sync_pr", "Synchronous version..."),
      clEnumValN(Algo::async, "async", "Asynchronous without priority version..."),
      clEnumValN(Algo::async_rsd, "async_rsd", "Asynchronous with residual version..."),
      clEnumValN(Algo::async_prt, "async_prt", "Prioritized (degree biased residual) version..."),
      clEnumValEnd), cll::init(Algo::async_prt));



//---------- parallel synchronous algorithm (reference: PullAlgo2)
struct Sync {
  struct LNode {
    float value[2];
    int id;
    unsigned int nout;
    float getPageRank() { return value[1]; }
    float getPageRank(unsigned int it) { return value[it & 1]; }
    void setPageRank(unsigned it, float v) { value[(it+1) & 1] = v; }
  };

  typedef Galois::Graph::LC_CSR_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    InnerGraph;
  typedef Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Sync"; }

  Galois::GReduceMax<float> max_delta;
  Galois::GAccumulator<unsigned int> small_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      Galois::Graph::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    int idcount = 0;
    void operator()(Graph::GraphNode n) {
      LNode& data = g.getData(n);
      data.value[0] = (1.0 - alpha);
      data.value[1] = (1.0 - alpha);
      int outs = std::distance(g.edge_begin(n), g.edge_end(n));
      data.nout = outs;
      data.id = idcount++;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) {
      LNode& data = g.getData(n);
      data.value[1] = data.value[0];
    }
  };

  struct Process {
    Sync* self;
    Graph& graph;
    unsigned int iteration;

    Process(Sync* s, Graph& g, unsigned int i): self(s), graph(g), iteration(i) { }

    void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {

      LNode& sdata = graph.getData(src);

      double sum = 0;
      for (auto jj = graph.in_edge_begin(src), ej = graph.in_edge_end(src);
          jj != ej; ++jj) {
        GNode dst = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst);
        sum += ddata.getPageRank(iteration) / ddata.nout;
      }

      float value = alpha*sum + (1.0 - alpha);
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
      Galois::do_all_local(graph, Process(this, graph, iteration));
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
      Galois::do_all_local(graph, Copy(graph));
    }
  }
};



//---------- asynchronous without priority (reference: PagerankDelta)
struct Async {
  struct LNode {
    float value;
    unsigned int nout;
    bool flag; // tracking if it is in the worklist
    LNode(): value(1.0), nout(0) {}
    float getPageRank() { return value; }
  };

  typedef Galois::Graph::LC_CSR_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    InnerGraph;
  typedef Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Async"; }

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      Galois::Graph::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) {
      LNode& data = g.getData(n);
      data.value = (1.0 - alpha);
      int outs = std::distance(g.edge_begin(n), g.edge_end(n));
      data.nout = outs;
      data.flag = true;
    }
  };
  
  struct Process {
    Graph& graph;
     
    Process(Graph& g): graph(g) { }

    void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) {
      LNode& sdata = graph.getData(src);
      // the node is processed
      sdata.flag = false;

      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, Galois::MethodFlag::NONE), ej = graph.in_edge_end(src, Galois::MethodFlag::NONE); jj != ej; ++jj) {
        GNode dst = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
        sum += ddata.value / ddata.nout;
      }
      float value = alpha*sum + (1.0 - alpha);
      float diff = std::fabs(value - sdata.value);
      if (diff > tolerance) {
        sdata.value = value;
        for (auto jj = graph.edge_begin(src, Galois::MethodFlag::NONE), ej = graph.edge_end(src, Galois::MethodFlag::NONE); jj != ej; ++jj) {
          GNode dst = graph.getEdgeDst(jj);
	  LNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
	  // if the node is not in the worklist, then push
	  if(!ddata.flag) {
	    ddata.flag = true;
            ctx.push(dst);
	  } 
        }
      } 
      
    }

  };

  void operator()(Graph& graph) {
    typedef Galois::WorkList::dChunkedFIFO<16> WL;
    Galois::for_each_local(graph, Process(graph), Galois::wl<WL>());
  }
};



//---------- asynchronous with residual (two level priority)
struct AsyncRsd {
  struct LNode {
    float value;
    unsigned int nout;
    bool flag; // tracking if it is in the worklist
    LNode(): value(1.0), nout(0) {}
    float getPageRank() { return value; }
    float residual; // tracking residual
  };

  typedef Galois::Graph::LC_CSR_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    InnerGraph;
  typedef Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "AsyncRsd"; }

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      Galois::Graph::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) {
      LNode& data = g.getData(n);
      data.value = (1.0 - alpha);
      int outs = std::distance(g.edge_begin(n), g.edge_end(n));
      data.nout = outs;
      data.flag = true;
      data.residual = 0.0;
    }
  };

  struct Process1 {
    AsyncRsd* self;
    Graph& graph;
     
    Process1(AsyncRsd* s, Graph& g): self(s), graph(g) { }

    void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {
      LNode& data = graph.getData(src);
      // for each in-coming neighbour, add residual
      for (auto jj = graph.in_edge_begin(src), ej = graph.in_edge_end(src); jj != ej; ++jj){
        GNode dst = graph.getInEdgeDst(jj);
	LNode& ddata = graph.getData(dst);
	data.residual = (float) data.residual + (float) 1/ddata.nout;  
      }
      data.residual = alpha*(1.0-alpha)*data.residual;
     }
  }; //--- end of Process1
  
  struct Process2 {
    Graph& graph;
     
    Process2(Graph& g): graph(g) { }

    void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) {
      LNode& sdata = graph.getData(src);
      // the node is processed
      sdata.flag = false;
      double sum = 0;
      for (auto jj = graph.in_edge_begin(src, Galois::MethodFlag::NONE), ej = graph.in_edge_end(src, Galois::MethodFlag::NONE); jj != ej; ++jj) {
        GNode dst = graph.getInEdgeDst(jj);
        LNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
        sum += ddata.value / ddata.nout;
      }
      float value = alpha*sum + (1.0 - alpha);
      float diff = std::fabs(value - sdata.value);

      if (diff > tolerance) {
        sdata.value = value;
        // for each out-going neighbors
        for (auto jj = graph.edge_begin(src, Galois::MethodFlag::NONE), ej = graph.edge_end(src, Galois::MethodFlag::NONE); jj != ej; ++jj) {
          GNode dst = graph.getEdgeDst(jj);
	  LNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
	  ddata.residual += sdata.residual*alpha/sdata.nout; // update residual
	  // if the node is not in the worklist and the residual is greater than tolerance
	  if(!ddata.flag && ddata.residual>tolerance) {
	    ddata.flag = true;
            ctx.push(dst);
	  } 
        }
        sdata.residual = 0.0; // update residual
      } // enf of if
      
    } // end of operator
  };

  void operator()(Graph& graph) {
    Galois::do_all_local(graph, Process1(this, graph), Galois::loopname("P1"));
    typedef Galois::WorkList::dChunkedFIFO<16> WL;
    Galois::for_each_local(graph, Process2(graph), Galois::wl<WL>());
  }
};



//---------- asynchronous with multilevel priority (degree normalized residual)
template<typename NTy>
int pri(const NTy& n) {
  double d = n.residual / n.deg;
  //double d = n.residual;
  return (int)(d * amp); //d > 0.1 ? 0 : 1;//-1*(int)sqrt(-1*d*amp);
}

struct AsyncPrt {

  struct LNode {
    float pagerank;
    int id;
    float residual;
    unsigned int nout;
    unsigned int deg;
    float getPageRank() { return pagerank; }
    float getResidual() { return residual; }
  };
 
  typedef Galois::Graph::LC_CSR_Graph<LNode,void>
    ::with_numa_alloc<true>::type
    InnerGraph;
  typedef Galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "AsyncPrt"; }

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      Galois::Graph::readGraph(graph, filename, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    int id=0;
    void operator()(Graph::GraphNode n) {
      LNode& data = g.getData(n);
      data.pagerank = (1.0 - alpha);
      data.residual = 0.0;
      data.id = id++;
      int outs = std::distance(g.edge_begin(n), g.edge_end(n));
      data.nout = outs;
      int ins = std::distance(g.in_edge_begin(n), g.in_edge_end(n));
      data.deg = outs + ins;
    }
  };

  struct Process1 {
    AsyncPrt* self;
    Graph& graph;
     
    Process1(AsyncPrt* s, Graph& g): self(s), graph(g) { }

    void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {
      LNode& data = graph.getData(src);
      // for each in-coming neighbour, add residual
      for (auto jj = graph.in_edge_begin(src), ej = graph.in_edge_end(src); jj != ej; ++jj){
        GNode dst = graph.getInEdgeDst(jj);
	LNode& ddata = graph.getData(dst);
	data.residual = (float) data.residual + (float) 1/ddata.nout;  
      }
      data.residual = alpha*(1.0-alpha)*data.residual;
    }
  }; //--- end of Process1

  // define priority
  typedef std::pair<int, GNode> UpdateRequest;
  struct UpdateRequestIndexer: public std::unary_function<UpdateRequest, int> {
  int operator() (const UpdateRequest& val) const {
    return val.first;
    }
  };

  struct Process2 {
    AsyncPrt* self;
    Graph& graph;
     
    Galois::Statistic& pre;
    Galois::Statistic& post;
     
    Process2(AsyncPrt* s, Graph& g, Galois::Statistic& _pre, Galois::Statistic& _post): self(s), graph(g), pre(_pre), post(_post) { }

    void operator()(const UpdateRequest& srcRq, Galois::UserContext<UpdateRequest>& ctx) {
      GNode src = srcRq.second;

      Galois::MethodFlag flag = Galois::MethodFlag::NONE;
      //Galois::MethodFlag flag = Galois::MethodFlag::ALL;      

      LNode* node = &graph.getData(src, flag);
      if (node->residual < tolerance || pri(*node) != srcRq.first){ 
	post +=1;
        return;
      }
      
      node = &graph.getData(src);

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
	float oldR = ddata.residual; 
        int oldP = pri(ddata);
        ddata.residual += node->residual*alpha/node->nout;
	if (ddata.residual >= tolerance && 
            (oldP != pri(ddata) || (oldR <= tolerance))) {
	  ctx.push(std::make_pair(pri(ddata), dst)); 
	} else {
          ++nopush;
        }
      }
      if (nopush)
        pre += nopush;

      node->pagerank = alpha*sum + (1.0-alpha);
      node->residual = 0.0;

    }

  }; //--- end of Process2

  void operator()(Graph& graph) {
    Galois::Statistic pre("PrePrune");
    Galois::Statistic post("PostPrune");
    std::cout<<"tolerance: "<<tolerance<<", amp: "<<amp<<"\n";

    Galois::do_all_local(graph, Process1(this, graph), Galois::loopname("P1"));
    
    using namespace Galois::WorkList;
    typedef dChunkedFIFO<16> dChunk;
    typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk>::with_block_period<4>::type OBIM;
    #ifdef GALOIS_USE_EXP
    typedef WorkListTracker<UpdateRequestIndexer, OBIM> dOBIM;
    #else
    typedef OBIM dOBIM;
    #endif
    Galois::InsertBag<UpdateRequest> initialWL;
    Galois::do_all_local(graph, [&initialWL, &graph] (GNode src) {
	LNode& data = graph.getData(src);
        if(data.residual>=tolerance){
	  initialWL.push_back(std::make_pair(pri(data), src)); 
        }
      });
    Galois::StatTimer T("InnerTime");
    T.start();
    Galois::for_each_local(initialWL, Process2(this, graph, pre, post), Galois::wl<OBIM>(), Galois::loopname("mainloop"));   
    T.stop();

    /*
    Galois::StatTimer Te("ExtraTime");
    Te.start();
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
    Te.stop();      
    */

  } 

};





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
static void printTop(Graph& graph, int topn, const char *algo_name, int numThreads) {
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
  
  /*
  char filename[256];
  int tamp = amp;
  float ttol = tolerance;
  sprintf(filename,"/scratch/01982/joyce/tmp/%s_t_%d_tol_%f_amp_%d", algo_name,numThreads,ttol,tamp);
  std::ofstream myfile;
  myfile.open (filename);
  */

  //std::cout<<"print PageRank\n";
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    node_data_reference n = graph.getData(src);
    float value = n.getPageRank()/sum; // normalized PR (divide PR by sum)
    //float value = n.getPageRank(); // raw PR 
    //std::cout<<value<<" "; 
    //myfile << value <<" ";
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
  //myfile.close();
  //std::cout<<"\nend of print\n";

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

  Galois::preAlloc(numThreads + (graph.size() * sizeof(typename Graph::node_data_type)) / Galois::Runtime::MM::hugePageSize);
  Galois::reportPageAlloc("MeminfoPre");

  Galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  std::cout << "tolerance: " << tolerance << "\n";
  T.start();
  Galois::do_all_local(graph, typename Algo::Initialize(graph));
  algo(graph);
  T.stop();
  
  Galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify)
    printTop(graph, 10, algo.name().c_str(), numThreads);
}

int main(int argc, char **argv) {
  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;

  Galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::sync_pr: run<Sync>(); break;
    case Algo::async: run<Async>(); break;
    case Algo::async_rsd: run<AsyncRsd>(); break;
    case Algo::async_prt: run<AsyncPrt>(); break;
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  return 0;
}
