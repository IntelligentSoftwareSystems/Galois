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
#include <set>

#include "PageRank.h"
#include "PageRankAsync.h"
#include "PageRankAsyncResidual.h"
#include "PageRankAsyncPri.h"
#include "PageRankAsyncPriSet.h"
#include "PageRankEdge.h"
#include "PPR.h"

namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin";
static const char* url = 0;

enum Algo {
  synch,
  async,
  async_rsd,
  async_prt,
  async_prs,
  async_edge,
  async_ppr_rsd
};

cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(10000000));
cll::opt<unsigned int> memoryLimit("memoryLimit",
    cll::desc("Memory limit for out-of-core algorithms (in MB)"), cll::init(~0U));
static cll::opt<int> amp("amp", cll::desc("amp for priority"), cll::init(100));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.01));
static cll::opt<bool> dbg("dbg", cll::desc("dbg"), cll::init(false));
static cll::opt<std::string> algo_str("algo_str", cll::desc("algo_str"), cll::init("NA"));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::synch, "synch", "Synchronous version..."),
      clEnumValN(Algo::async, "async", "Asynchronous versoin..."),
      clEnumValN(Algo::async_rsd, "async_rsd", "Residual-based asynchronous version..."),
      clEnumValN(Algo::async_prt, "async_prt", "Prioritized (degree biased residual) version..."),
      clEnumValN(Algo::async_prs, "async_prs", "Prioritized Bulk Sync version..."),
      clEnumValN(Algo::async_edge, "async_edge", "Edge based"),
      clEnumValN(Algo::async_ppr_rsd, "async_ppr_rsd", "Asyncronous PPR"),
      clEnumValEnd), cll::init(Algo::async));



struct Sync {
  struct LNode {
    float value[2];
    void init() { value[0] = value[1] = 1.0 - alpha; }
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

  void readGraph(Graph& graph, std::string filename, std::string transposeGraphName) {
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
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, Galois::MethodFlag::NONE);
      data.value[0] = 1.0;
      data.value[1] = 1.0;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      LNode& data = g.getData(n, Galois::MethodFlag::NONE);
      data.value[1] = data.value[0];
    }
  };

  struct Process {
    Sync* self;
    Graph& graph;
    unsigned int iteration;

    Process(Sync* s, Graph& g, unsigned int i): self(s), graph(g), iteration(i) { }

    void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) const {
      (*this)(src);
    }

    void operator()(const GNode& src) const {

      LNode& sdata = graph.getData(src);

      Galois::MethodFlag lockflag = Galois::MethodFlag::NONE;
      double sum = computePageRankInOut(graph, src, iteration, lockflag);

      float value = alpha*sum + (1.0 - alpha);
      float diff = std::fabs(value - sdata.getPageRank(iteration));
      sdata.setPageRank(iteration, value);
      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
    }
  };

  void operator()(Graph& graph, PRTy, PRTy) {
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

  void verify(Graph& graph, PRTy tolerance) {}
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
  
  std::ofstream myfile;
  if(dbg){
    char filename[256];
    int tamp = amp;
    float ttol = tolerance;
    sprintf(filename,"/scratch/01982/joyce/tmp/%s_t_%d_tol_%f_amp_%d", algo_name,numThreads,ttol,tamp);
    myfile.open (filename);
  }

  //std::cout<<"print PageRank\n";
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    node_data_reference n = graph.getData(src);
    float value = n.getPageRank()/sum; // normalized PR (divide PR by sum)
    //float value = n.getPageRank(); // raw PR 
    //std::cout<<value<<" "; 
    if(dbg){
      myfile << value <<" ";
    }
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
  if(dbg){
    myfile.close();
  }
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

  algo.readGraph(graph, filename, transposeGraphName);

  Galois::preAlloc(numThreads + (graph.size() * sizeof(typename Graph::node_data_type)) / Galois::Runtime::MM::hugePageSize);
  Galois::reportPageAlloc("MeminfoPre");

  Galois::StatTimer T;
  auto eamp = -amp;///tolerance;
  std::cout << "Running " << algo.name() << " version\n";
  std::cout << "tolerance: " << tolerance << "\n";
  std::cout << "effective amp: " << eamp << "\n";
  T.start();
  Galois::do_all_local(graph, [&graph] (typename Graph::GraphNode n) { graph.getData(n).init(); });
  algo(graph, tolerance, eamp);
  T.stop();
  
  Galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    algo.verify(graph, tolerance);
    printTop(graph, 10, algo.name().c_str(), numThreads);
  }
}



template<typename Algo>
void runPPR() {
  typedef typename Algo::Graph Graph;

  Algo algo;
  Graph graph;

  algo.readGraph(graph, filename, transposeGraphName);

  std::cout << "Read " << std::distance(graph.begin(), graph.end()) << " Nodes\n";

  Galois::preAlloc(numThreads + (graph.size() * sizeof(typename Graph::node_data_type)) / Galois::Runtime::MM::hugePageSize);
  Galois::reportPageAlloc("MeminfoPre");

  unsigned numSeeds = 2;

  Galois::StatTimer Tt;
  Tt.start();
  auto seeds = findPPRSeeds(graph, numSeeds);
  Tt.stop();
  std::cout << "Find " << numSeeds << " seeds (found " << seeds.size() << ") in " << Tt.get() << "ms\n";

  std::map<typename Graph::GraphNode, std::deque<unsigned> > clusters;

  unsigned counter = 0;
  for (auto seed : seeds) {

    Galois::StatTimer T1, T2, T3;
    std::cout << "Running " << algo.name() << " version\n";
    T1.start();
    Galois::do_all_local(graph, [&graph] (typename Graph::GraphNode n) { graph.getData(n).init(); });
    T1.stop();
    T2.start();
    algo(graph, seed);
    T2.stop();
    T3.start();
    auto c = algo.cluster_from_sweep(graph);
    T3.stop();
    std::cout << T1.get() << " " << T2.get() << " " << T3.get() << "\n";

    std::cout << seed << " : " << c.size() << "\n";
    // for (auto n : c)
    //   std::cout << n << " ";
    // std::cout << "\n";

    for (auto n : c)
      clusters[n].push_back(counter);
    ++counter;

    Galois::reportPageAlloc("MeminfoPost");
  }

  for (auto np : clusters) {
    std::cout << np.first << ": ";
    for (auto n : np.second)
      std::cout << n << " ";
    std::cout << "\n";
  }
}



int main(int argc, char **argv) {
  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;

  Galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::synch: run<Sync>(); break;
    case Algo::async: run<AsyncSet>(); break;
    case Algo::async_rsd: run<AsyncRsd>(); break;
    case Algo::async_prt: run<AsyncPri>(); break;
    case Algo::async_prs: run<AsyncPriSet>(); break;
    case Algo::async_edge: run<AsyncEdge>(); break;
    case Algo::async_ppr_rsd: runPPR<PPRAsyncRsd>(); break;
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  return 0;
}
