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
 */
#include "Galois/config.h"
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/Graph/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"

#include GALOIS_CXX11_STD_HEADER(atomic)
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>

#include "PageRank.h"
#ifdef GALOIS_USE_EXP
#include "GraphLabAlgo.h"
#include "LigraAlgo.h"
#endif

namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin";
static const char* url = 0;

enum Algo {
  graphlab,
  graphlabAsync,
  ligra,
  ligraChi,
  pull,
  serial
};

cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Input graph is symmetric"));
static cll::opt<std::string> outputPullFilename("outputPull", cll::desc("Precompute data for Pull algorithm to file"));
cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(100));
cll::opt<unsigned int> memoryLimit("memoryLimit",
    cll::desc("Memory limit for out-of-core algorithms (in MB)"), cll::init(~0U));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::pull, "pull", "Use precomputed data perform pull-based algorithm"),
      clEnumValN(Algo::serial, "serial", "Compute PageRank in serial"),
#ifdef GALOIS_USE_EXP
      clEnumValN(Algo::graphlab, "graphlab", "Use GraphLab programming model"),
      clEnumValN(Algo::graphlabAsync, "graphlabAsync", "Use GraphLab-Asynchronous programming model"),
      clEnumValN(Algo::ligra, "ligra", "Use Ligra programming model"),
      clEnumValN(Algo::ligraChi, "ligraChi", "Use Ligra and GraphChi programming model"),
#endif
      clEnumValEnd), cll::init(Algo::pull));

struct SerialAlgo {
  typedef Galois::Graph::LC_CSR_Graph<PNode,void>
    ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Serial"; }
  
  void readGraph(Graph& graph) { Galois::Graph::readGraph(graph, filename); }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) {
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
        PNode& sdata = graph.getData(src, Galois::MethodFlag::NONE);
        float value = (1.0 - alpha) * sdata.accum.read() + alpha;
        float diff = std::fabs(value - sdata.value);
        if (diff <= tolerance)
          ++small_delta;
        if (diff > max_delta)
          max_delta = diff;
        sdata.value = value;
        sdata.accum.write(0);
      }

      iteration += 1;

      std::cout << "iteration: " << iteration
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
  typedef Galois::Graph::LC_InlineEdge_Graph<LNode,float>
    ::with_compressed_node_ptr<true>::type
    ::with_no_lockable<true>::type
    ::with_numa_alloc<true>::type
    Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Pull"; }

  Galois::GReduceMax<double> max_delta;
  Galois::GAccumulator<unsigned int> small_delta;

  void readGraph(Graph& graph) {
    if (transposeGraphName.size()) {
      Galois::Graph::readGraph(graph, transposeGraphName); 
    } else {
      std::cerr << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) {
      LNode& data = g.getData(n, Galois::MethodFlag::NONE);
      data.value[0] = 1.0;
      data.value[1] = 1.0;
    }
  };

  struct Copy {
    Graph& g;
    Copy(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) {
      LNode& data = g.getData(n, Galois::MethodFlag::NONE);
      data.value[1] = data.value[0];
    }
  };

  struct Process {
    PullAlgo* self;
    Graph& graph;
    unsigned int iteration;

    Process(PullAlgo* s, Graph& g, unsigned int i): self(s), graph(g), iteration(i) { }

    void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) {
      (*this)(src);
    }

    void operator()(const GNode& src) {
      LNode& sdata = graph.getData(src, Galois::MethodFlag::NONE);
      double sum = 0;

      for (auto jj = graph.edge_begin(src, Galois::MethodFlag::NONE), ej = graph.edge_end(src, Galois::MethodFlag::NONE); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        float w = graph.getEdgeData(jj);

        LNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
        sum += ddata.getPageRank(iteration) * w;
      }

      float value = sum * (1.0 - alpha) + alpha;
      float diff = std::fabs(value - sdata.getPageRank(iteration));
       
      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
      sdata.setPageRank(iteration, value);
    }
  };

  void operator()(Graph& graph) {
    unsigned int iteration = 0;
    
    while (true) {
      Galois::for_each_local(graph, Process(this, graph, iteration));
      iteration += 1;

      float delta = max_delta.reduce();
      size_t sdelta = small_delta.reduce();

      std::cout << "iteration: " << iteration
                << " max delta: " << delta
                << " small delta: " << sdelta
                << " (" << sdelta / (float) graph.size() << ")"
                << "\n";

      if (delta <= tolerance || iteration >= maxIterations)
        break;
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

//! Transpose in-edges to out-edges
static void precomputePullData() {
  typedef Galois::Graph::LC_CSR_Graph<size_t, void>
    ::with_no_lockable<true>::type InputGraph;
  typedef InputGraph::GraphNode InputNode;
  typedef Galois::Graph::FileGraphWriter OutputGraph;
  //typedef OutputGraph::GraphNode OutputNode;

  InputGraph input;
  OutputGraph output;
  Galois::Graph::readGraph(input, filename); 

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
  memcpy(t, &edgeData[0], sizeof(edgeData[0]) * edgeData.size());
  
  output.structureToFile(outputPullFilename);
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

template<typename Graph>
static void printTop(Graph& graph, int topn) {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair,GNode> Top;

  Top top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    node_data_reference n = graph.getData(src);
    float value = n.getPageRank();
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

  Galois::preAlloc(numThreads + (graph.size() * sizeof(typename Graph::node_data_type)) / Galois::Runtime::MM::pageSize);
  Galois::reportPageAlloc("MeminfoPre");

  Galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  std::cout << "Target max delta: " << tolerance << "\n";
  T.start();
  Galois::do_all_local(graph, typename Algo::Initialize(graph));
  algo(graph);
  T.stop();
  
  Galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify)
    printTop(graph, 10);
}

int main(int argc, char **argv) {
  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;

  if (outputPullFilename.size()) {
    precomputePullData();
    return 0;
  }

  Galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::pull: run<PullAlgo>(); break;
#ifdef GALOIS_USE_EXP
    case Algo::ligra: run<LigraAlgo<false> >(); break;
    case Algo::ligraChi: run<LigraAlgo<true> >(); break;
    case Algo::graphlab: run<GraphLabAlgo<false,false> >(); break;
    case Algo::graphlabAsync: run<GraphLabAlgo<true,true> >(); break;
#endif
    case Algo::serial: run<SerialAlgo>(); break;
    default: std::cerr << "Unknown algorithm\n"; abort();
  }
  T.stop();

  return 0;
}
