/** Betweenness Centrality -*- C++ -*-
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
 * @section Description
 *
 * Betweenness centrality. Implementation from Ligra
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "galois/Galois.h"
#include "galois/Accumulator.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#ifdef GALOIS_USE_EXP
#include <boost/mpl/if.hpp>
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/GraphNodeBag.h"
#include "galois/DomainSpecificExecutors.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <atomic>
#include <string>
#include <deque>
#include <iostream>
#include <iomanip>

static const char* name = "Betweenness Centrality";
static const char* desc = 0;
static const char* url = 0;

//****** Command Line Options ******
enum Algo {
  ligra,
  ligraChi
};

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Input graph is symmetric"));
static cll::opt<unsigned int> startNode("startNode", cll::desc("Node to start search from"), cll::init(0));
static cll::opt<unsigned int> memoryLimit("memoryLimit",
    cll::desc("Memory limit for out-of-core algorithms (in MB)"), cll::init(~0U));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::ligra, "ligra", "Using Ligra programming model"),
      clEnumValN(Algo::ligraChi, "ligraChi", "Using Ligra and GraphChi programming model"),
      clEnumValEnd), cll::init(Algo::ligra));

template<typename Algo>
void initialize(Algo& algo,
    typename Algo::Graph& graph,
    typename Algo::Graph::GraphNode& source) {

  algo.readGraph(graph);
  std::cout << "Read " << graph.size() << " nodes\n";

  if (startNode >= graph.size()) {
    std::cerr << "failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
  
  typename Algo::Graph::iterator it = graph.begin();
  std::advance(it, startNode);
  source = *it;
}

template<typename Graph>
void readInOutGraph(Graph& graph) {
  using namespace galois::graphs;
  if (symmetricGraph) {
    galois::graphs::readGraph(graph, filename);
  } else if (transposeGraphName.size()) {
    galois::graphs::readGraph(graph, filename, transposeGraphName);
  } else {
    GALOIS_DIE("Graph type not supported");
  }
}

#ifdef GALOIS_USE_EXP
template<bool UseGraphChi>
struct LigraAlgo: public galois::LigraGraphChi::ChooseExecutor<UseGraphChi> {

  //ICC v13.1 doesn't yet support std::atomic<float> completely, emmulate its
  //behavor with std::atomic<int>
  struct atomic_float : public std::atomic<int> {
  private:
    operator int() const;
    static_assert(sizeof(int) == sizeof(float), "int and float must be the same size");
  public:
    atomic_float() { }

    float atomicIncrement(float value) {
      while (true) {
        union { float as_float; int as_int; } oldValue = { read() };
        union { float as_float; int as_int; } newValue = { oldValue.as_float + value };
        if (this->compare_exchange_strong(oldValue.as_int, newValue.as_int))
          return oldValue.as_float;
      }
    }

    float read() {
      union { int as_int; float as_float; } caster = { this->load(std::memory_order_relaxed) };
      return caster.as_float;
    }

    void write(float v) {
      union { float as_float; int as_int; } caster = { v };
      this->store(caster.as_int, std::memory_order_relaxed);
    }
  };

  struct SNode {
    atomic_float numPaths;
    atomic_float dependencies;
    bool visited;
  };

  typedef typename galois::graphs::LC_CSR_Graph<SNode,void>
    ::template with_no_lockable<true>::type 
    ::template with_numa_alloc<true>::type InnerGraph;

  typedef typename boost::mpl::if_c<UseGraphChi,
          galois::graphs::OCImmutableEdgeGraph<SNode,void>,
          galois::graphs::LC_InOut_Graph<InnerGraph> >::type
          Graph;
  typedef typename Graph::GraphNode GNode;
  typedef galois::graphsNodeBag<1024*4> Bag;

  std::string name() const { return UseGraphChi ? "LigraChi" : "Ligra"; }

  void readGraph(Graph& graph) { 
    readInOutGraph(graph); 
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(typename Graph::GraphNode n) const {
      SNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.numPaths.write(0.0);
      data.dependencies.write(0.0);
      data.visited = false;
    }
  };

  struct ForwardPass {
    template<typename GTy>
    bool cond(GTy& graph, typename GTy::GraphNode n) { 
      return !graph.getData(n, galois::MethodFlag::UNPROTECTED).visited; 
    }

    template<typename GTy>
    bool operator()(GTy& graph, typename GTy::GraphNode src, typename GTy::GraphNode dst, typename GTy::edge_data_reference) {
      SNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      float oldValue = ddata.numPaths.atomicIncrement(sdata.numPaths.read());
      return oldValue == 0.0;
    }
  };

  struct BackwardPass {
    template<typename GTy>
    bool cond(GTy& graph, typename GTy::GraphNode n) { 
      return !graph.getData(n, galois::MethodFlag::UNPROTECTED).visited; 
    }

    template<typename GTy>
    bool operator()(GTy& graph, typename GTy::GraphNode src, typename GTy::GraphNode dst, typename GTy::edge_data_reference) {
      SNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      float oldValue = ddata.dependencies.atomicIncrement(sdata.dependencies.read());
      return oldValue == 0.0;
    }
  };

  void operator()(Graph& graph, GNode source) {
    typedef galois::worklists::dChunkedFIFO<256> WL;
    std::deque<Bag*> levels;

    graph.getData(source).visited = true;
    graph.getData(source).numPaths.write(1);
    Bag* frontier = new Bag(graph.size());
    frontier->push(source, 1);
    levels.push_back(frontier);
    int round = 0;

    while (!frontier->empty()) {
      ++round;
      Bag* output = new Bag(graph.size());
      this->outEdgeMap(memoryLimit, graph, ForwardPass(), *frontier, *output, false);
      //galois::do_all_local(*output, [&](GNode n) {
      //galois::do_all(output->begin(), output->end(), [&](GNode n) {
      galois::for_each_local(*output, [&](size_t id, galois::UserContext<size_t>&) {
        SNode& d = graph.getData(graph.nodeFromId(id), galois::MethodFlag::UNPROTECTED);
        d.visited = true;
        },galois::wl<WL>()); 
      levels.push_back(output);
      frontier = output;
    }

    delete levels[round];

    galois::do_all_local(graph, [&](GNode n) {
        SNode& d = graph.getData(n, galois::MethodFlag::UNPROTECTED);
        d.numPaths.write(1.0/d.numPaths.read());
        d.visited = false;
    });

    frontier = levels[round-1];

    //galois::do_all_local(*frontier, [&](GNode n) {
    galois::for_each_local(*frontier, [&](size_t id, galois::UserContext<size_t>&) {
      SNode& d = graph.getData(graph.nodeFromId(id), galois::MethodFlag::UNPROTECTED);
      d.visited = true;
      d.dependencies.write(d.dependencies.read() + d.numPaths.read());
      }, galois::wl<WL>());

    for (int r = round - 2; r >= 0; --r) {
      Bag output(graph.size());
      this->inEdgeMap(memoryLimit, graph, BackwardPass(), *frontier, output, false);
      delete frontier;
      frontier = levels[r];
      //galois::do_all_local(*frontier, [&](GNode n) {
      galois::for_each_local(*frontier, [&](size_t id, galois::UserContext<size_t>&) {
        SNode& d = graph.getData(graph.nodeFromId(id), galois::MethodFlag::UNPROTECTED);
        d.visited = true;
        d.dependencies.write(d.dependencies.read() + d.numPaths.read());
        }, galois::wl<WL>());
    }

    delete frontier;

    galois::do_all_local(graph, [&](GNode n) {
      SNode& d = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      d.dependencies.write((d.dependencies.read() - d.numPaths.read())
          / d.numPaths.read());
    });
  }
};
#endif

template<typename Algo>
void run() {
  typedef typename Algo::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  Algo algo;
  Graph graph;
  GNode source;

  initialize(algo, graph, source);

  galois::preAlloc(numThreads + (3*graph.size() * sizeof(typename Graph::node_data_type)) / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  T.start();
  galois::do_all_local(graph, typename Algo::Initialize(graph));
  algo(graph, source);
  T.stop();
  
  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    int count = 0;
    for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei && count < 20; ++ii, ++count) {
      std::cout << count << ": "
        << std::setiosflags(std::ios::fixed) << std::setprecision(6) << graph.getData(*ii).dependencies.read()
                << " " << (int)round(1.0 / graph.getData(*ii).numPaths.read())
                << "\n";
    }
    count = 0;
    // for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii, ++count)
    //   std::cout << ((count % 128 == 0) ? "\n" : " ") << (int)round(1.0 / graph.getData(*ii).numPaths.read());
    std::cout << "\n";
  }
}

int main(int argc, char **argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
#ifdef GALOIS_USE_EXP
    case Algo::ligra: run<LigraAlgo<false> >(); break;
    case Algo::ligraChi: run<LigraAlgo<true> >(); break;
#endif
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  return 0;
}
