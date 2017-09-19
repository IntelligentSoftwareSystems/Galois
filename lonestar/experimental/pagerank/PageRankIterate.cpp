/** Page rank application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/UnionFind.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Graphs/LCGraph.h"

#include "Lonestar/BoilerPlate.h"

#ifdef USE_POSKI
#include <poski.h>
#endif

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <sys/mman.h>

namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin\n";
static const char* url = NULL;

enum Algo {
  transpose,
  synchronous,
  serializable,
  nondeterministic,
  serial,
  asynchronous,
  dummy,
  spmv
};

static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> otherFilename(cll::Positional, cll::desc("[output or transpose graph]"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(100));
static cll::opt<Algo> algo(cll::desc("Algorithm:"),
    cll::values(
      clEnumValN(Algo::transpose, "transpose", "Transpose graph"),
      clEnumValN(Algo::synchronous, "synchronous", "Compute PageRank using synchronous, parallel algorithm (default)"),
      clEnumValN(Algo::serializable, "serializable", "Compute PageRank using non-deterministic but serializable parallel algorithm"),
      clEnumValN(Algo::nondeterministic, "nondeterministic", "Compute PageRank using non-deterministic parallel algorithm"),
      clEnumValN(Algo::asynchronous, "asynchronous", "Compute PageRank using asynchronous, parallel algorithm"),
      clEnumValN(Algo::dummy, "dummy", ""),
#ifdef USE_POSKI
      clEnumValN(Algo::spmv, "spmv", "iterative sparse matrix vector multiply"),
#endif
      clEnumValN(Algo::serial, "serial", "Compute PageRank in serial"),
      clEnumValEnd), cll::init(Algo::synchronous));
static cll::opt<bool> useOnlySmallDegree("useOnlySmallDegree", cll::desc("Use only small degree nodes"), cll::init(false));

//! d is the damping factor. Alpha is the prob that user will do a random jump, i.e., 1 - d
static const double alpha = 1.0 - 0.85;

//! maximum relative change until we deem convergence
static const double tolerance = 0.1;

struct TData: public Galois::UnionFindNode<TData> {
  double values[8];
  unsigned int id;

  TData(): Galois::UnionFindNode<TData>(const_cast<TData*>(this)) { }

  double getPageRank(unsigned int it) {
    return values[it & (8-1)];
  }

  void setPageRank(unsigned int it, double v) {
    values[(it+1) & (8-1)] = v;
  }
};

struct GData: public Galois::UnionFindNode<GData> { 
  GData(): Galois::UnionFindNode<GData>(const_cast<GData*>(this)) { }
};

// A graph and its transpose. The main operation in pagerank is computing
// values based on incoming edges, so we will mainly use the transpose of G,
// G^T, (tgraph). We keep G around to compute dependencies in G^T.
typedef Galois::Graph::LC_CSR_Graph<GData, void> Graph;
typedef Galois::Graph::LC_CSR_Graph<TData, double> TGraph;
typedef Graph::GraphNode GNode;
typedef TGraph::GraphNode TNode;

Graph graph;
TGraph tgraph;
Galois::LargeArray<GNode> tgraphOrder;
size_t numSmallNeighbors;

struct SerialAlgo {
  unsigned int operator()() {
    unsigned int iteration = 0;
    unsigned int numNodes = tgraph.size();
    float tol = tolerance;

    std::cout << "target max delta: " << tol << "\n";

    while (true) {
      float max_delta = std::numeric_limits<float>::min();
      unsigned int small_delta = 0;

      for (TNode src : tgraph) {
        TData& sdata = tgraph.getData(src, Galois::MethodFlag::UNPROTECTED);
        float sum = 0;

        for (TGraph::edge_iterator edge : tgraph.out_edges(src, Galois::MethodFlag::UNPROTECTED)) {
          TNode dst = tgraph.getEdgeDst(edge);
          float w = tgraph.getEdgeData(edge);

          TData& ddata = tgraph.getData(dst, Galois::MethodFlag::UNPROTECTED);
          sum += ddata.getPageRank(iteration) * w;
        }
         
        float value = sum * (1.0 - alpha) + alpha;
        float diff = (value - sdata.getPageRank(iteration));

        if (diff < 0)
          diff = -diff;
        if (diff > max_delta)
          max_delta = diff;
        if (diff <= tol)
          ++small_delta;
        sdata.setPageRank(iteration, value);
      }

      std::cout << "iteration: " << iteration
                << " max delta: " << max_delta
                << " small delta: " << small_delta
                << " (" << small_delta / (float) numNodes << ")"
                << "\n";

      if (++iteration < maxIterations && max_delta > tol) {
        continue;
      } else
        break;
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }
    return iteration;
  }
};

static bool checkEnviron() {
  const char* envVal = getenv("OSKI_BYPASS_CHECK");
  if (envVal && strcmp(envVal, "yes") == 0) 
    return getenv("GALOIS_DO_NOT_BIND_MAIN_THREAD") != 0;
  else
    return false;
}

#ifdef USE_POSKI
struct SpMVAlgo {
  unsigned int operator()() {
    if (!checkEnviron()) {
      std::cerr << "please set environment variables:\n" 
        << "  GALOIS_DO_NOT_BIND_MAIN_THREAD=1\n"
        << "  OSKI_BYPASS_CHECK=yes\n";
      abort();
    }

    poski_Init();
    poski_threadarg_t* poski_thread = poski_InitThreads();
    poski_ThreadHints(poski_thread, NULL, POSKI_THREADPOOL, numThreads);

    Galois::StatTimer CT("ConvertTime");
    CT.start();
    // OSKI CSR only supports int indices.
    if (tgraph.sizeEdges() > std::numeric_limits<int>::max()
        || tgraph.size() > std::numeric_limits<int>::max()) {
      std::cerr << "graph too big for OSKI: "
        << "|V| = " << tgraph.size()
        << " |E| = " << tgraph.sizeEdges() << "\n";
      abort();
    }
    int curEdge = 0;
    int curNode = 0;
    int nrows = tgraph.size();
    int ncols = nrows;
    int nnz = tgraph.sizeEdges();
    double initial = 1.0;

    Galois::LargeArray<int> Aptr;
    Galois::LargeArray<int> Aind;
    Galois::LargeArray<double> Aval;
    Galois::LargeArray<double> xval;
    Galois::LargeArray<double> yval;

    Aptr.create(nrows + 1);
    Aind.create(nnz);
    Aval.create(nnz);
    xval.create(ncols);
    yval.create(ncols);

    Aptr[curNode++] = 0;
    for (TNode src : tgraph) {
      for (TGraph::edge_iterator edge : tgraph.out_edges(src)) {
        TNode dst = tgraph.getEdgeDst(edge);
        Aind[curEdge] = dst;
        Aval[curEdge] = tgraph.getEdgeData(edge);
        ++curEdge;
      }
      xval[curNode-1] = initial;
      Aptr[curNode] = curEdge;
      ++curNode;
    }
    CT.stop();

    poski_mat_t Atunable = poski_CreateMatCSR(&Aptr[0], &Aind[0], &Aval[0], nrows, ncols, nnz,
        SHARE_INPUTMAT,
        poski_thread,
        NULL,
        2, INDEX_ZERO_BASED, MAT_GENERAL);

    poski_vec_t xview = poski_CreateVec(&xval[0], ncols, STRIDE_UNIT, NULL);
    poski_vec_t yview = poski_CreateVec(&yval[0], ncols, STRIDE_UNIT, NULL);

    if (false) {
      Galois::StatTimer TT("TuningTime");
      TT.start();
      poski_TuneHint_Structure(Atunable, HINT_NO_BLOCKS, ARGS_MethodFlag::UNPROTECTED);
      poski_TuneHint_MatMult(Atunable, OP_NORMAL, 1, SYMBOLIC_VECTOR, 1, SYMBOLIC_VECTOR, ALWAYS_TUNE_AGGRESSIVELY);
      poski_TuneMat(Atunable);
      TT.stop();
    }

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
      poski_MatMult(Atunable, OP_NORMAL, 1, xview, 1, yview);
      std::swap(xview, yview);
      std::cout << "iteration: " << iteration << "\n";
    }

    poski_DestroyMat(Atunable);
    poski_DestroyVec(xview);
    poski_DestroyVec(yview);
    poski_DestroyThreads(poski_thread);
    poski_Close();

    return 0;
  }
};
#endif

struct DummyAlgo {
  static const int timeBlockSize = 10;
  static const int graphBlockSize = 1;

  struct Accum {
    Galois::GReduceMax<double> max_delta;
    Galois::GAccumulator<unsigned int> small_delta;
    void reset() {
      max_delta.reset();
      small_delta.reset();
    }
  };

  Accum accum;

  struct Process {
    Accum& accum;
    double tol;
    double addend;
    unsigned int baseIteration;
    unsigned int iteration;

    Process(Accum& a, double t, unsigned int i):
      accum(a), tol(t), addend(alpha/tgraph.size()), baseIteration(i) { }

    void operator()(unsigned tid, unsigned numThreads) {
      //size_t N = tgraph.size();
      size_t N = numSmallNeighbors; // XXX
      size_t blockSize = (N + numThreads - 1) / numThreads;
      size_t begin = std::min(tid * blockSize, N);
      size_t end = std::min(begin + blockSize, N);

      size_t b = begin;
      for (; b < end; b += graphBlockSize) {
        for (size_t x = 0; x < graphBlockSize; ++x) {
          for (unsigned int it = 0; it < timeBlockSize; ++it) {
            process(tgraphOrder[b+x], baseIteration + it);
          }
        }
      }
      // Epilogue
      b -= graphBlockSize - 1;
      for (; b < end; ++b) {
        for (unsigned int it = 0; it < timeBlockSize; ++it) {
          process(tgraphOrder[b], baseIteration + it);
        }
      }
    }

    void process(const TNode& src, unsigned int iteration) {
      TData& sdata = tgraph.getData(src, Galois::MethodFlag::UNPROTECTED);

      double sum = 0;
      for (TGraph::edge_iterator edge : tgraph.out_edges(src, Galois::MethodFlag::UNPROTECTED)) {
        TNode dst = tgraph.getEdgeDst(edge);
        double w = tgraph.getEdgeData(edge);

        TData& ddata = tgraph.getData(dst, Galois::MethodFlag::UNPROTECTED);
        sum += ddata.getPageRank(iteration) * w;
      }
       
      // assuming uniform prior probability, i.e., 1 / numNodes
      double value = sum * (1.0 - alpha) + alpha;
      double diff = value - sdata.getPageRank(iteration);
      
      if (diff < 0)
        diff = -diff;
      accum.max_delta.update(diff);
      if (diff <= tol)
        accum.small_delta += 1;
      sdata.setPageRank(iteration, value);
    }
  };

  unsigned int operator()() {
    //unsigned int numNodes = tgraph.size();
    unsigned int numNodes = numSmallNeighbors;
    double tol = tolerance;

    std::cout << "target max delta: " << tol << "\n";
    unsigned int iteration;
    for (iteration = 0; iteration < maxIterations; iteration += timeBlockSize) {
      Galois::on_each(Process(accum, tol, iteration));
      unsigned int small_delta = accum.small_delta.reduce();
      double max_delta = accum.max_delta.reduce();

      accum.reset();

      std::cout << "iteration: " << iteration << " - " << iteration + timeBlockSize
                << " max delta: " << max_delta
                << " small delta: " << small_delta
                << " (" << small_delta / (timeBlockSize * (float) numNodes) << ")"
                << "\n";
    }

    return iteration;
  }
};

template<bool useND, bool useS>
struct SynchronousAlgo {
  struct Accum {
    Galois::GReduceMax<double> max_delta;
    Galois::GAccumulator<unsigned int> small_delta;
    void reset() {
      max_delta.reset();
      small_delta.reset();
    }
  };

  Accum accum;

  struct Process {
    Accum& accum;
    double tol;
    unsigned int iteration;

    Process(Accum& a, double t, unsigned int i):
      accum(a), tol(t), iteration(i) { }

    void operator()(const TNode& src, Galois::UserContext<TNode>& ctx) const {
      operator()(src);
    }

    void operator()(const TNode& src) const {
      TData& sdata = tgraph.getData(src, Galois::MethodFlag::UNPROTECTED);
      double sum = 0;

      for (TGraph::edge_iterator edge : tgraph.out_edges(src, useND && useS ? Galois::MethodFlag::WRITE : Galois::MethodFlag::UNPROTECTED)) {
        TNode dst = tgraph.getEdgeDst(edge);
        double w = tgraph.getEdgeData(edge);

        TData& ddata = tgraph.getData(dst, useND && useS ? Galois::MethodFlag::WRITE : Galois::MethodFlag::UNPROTECTED);
        sum += ddata.getPageRank(useND ? 0 : iteration) * w;
      }

      double value = sum * (1.0 - alpha) + alpha;
      double diff = value - sdata.getPageRank(useND ? 0 : iteration);
       
      if (diff < 0)
        diff = -diff;
      accum.max_delta.update(diff);
      if (diff <= tol)
        accum.small_delta += 1;
      sdata.setPageRank(useND ? 1 : iteration, value);
    }
  };

  unsigned int operator()() {
    unsigned int iteration = 0;
    //unsigned int numNodes = tgraph.size(); // XXX
    unsigned int numNodes = numSmallNeighbors;
    double tol = tolerance;

    std::cout << "target max delta: " << tol << "\n";
    
    while (true) {
      if (useND && useS) {
        Galois::for_each(tgraphOrder.begin(), tgraphOrder.begin() + numNodes,
            Process(accum, tol, iteration));
      } else {
        Galois::do_all(tgraphOrder.begin(), tgraphOrder.begin() + numNodes,
            Process(accum, tol, iteration));
        //Galois::do_all_local(tgraph,
        //    Process(accum, tol, iteration));
      }

      unsigned int small_delta = accum.small_delta.reduce();
      double max_delta = accum.max_delta.reduce();

      accum.reset();

      std::cout << "iteration: " << iteration
                << " max delta: " << max_delta
                << " small delta: " << small_delta
                << " (" << small_delta / (float) numNodes << ")"
                << "\n";

      if (++iteration < maxIterations && max_delta > tol) {
        continue;
      } else {
        break;
      }
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    return iteration;
  }
};

template<bool useND, bool useS>
struct AsynchronousAlgo {
  struct Accum {
    Galois::GReduceMax<double> max_delta;
    Galois::GAccumulator<unsigned int> small_delta;
    void reset() {
      max_delta.reset();
      small_delta.reset();
    }
  };

  Accum accum;

  struct Process {
    Accum& accum;
    double tol;
    unsigned int iteration;

    Process(Accum& a, double t, unsigned int i):
      accum(a), tol(t), iteration(i) { }

    void operator()(const TNode& src, Galois::UserContext<TNode>& ctx) {
      TData& sdata = tgraph.getData(src, Galois::MethodFlag::UNPROTECTED);
      double sum = 0;

      for (TGraph::edge_iterator edge : tgraph.out_edges(src, useND && useS ? Galois::MethodFlag::WRITE : Galois::MethodFlag::UNPROTECTED)) {
        TNode dst = tgraph.getEdgeDst(edge);
        double w = tgraph.getEdgeData(edge);

        TData& ddata = tgraph.getData(dst, useND && useS ? Galois::MethodFlag::WRITE : Galois::MethodFlag::UNPROTECTED);
        sum += ddata.getPageRank(useND ? 0 : iteration) * w;
      }

      double value = sum * (1.0 - alpha) + alpha;
      double diff = value - sdata.getPageRank(useND ? 0 : iteration);
       
      if (diff < 0)
        diff = -diff;
      if (diff > tol)
        ctx.push(src);

      sdata.setPageRank(useND ? -1 : iteration, value);
    }
  };

  unsigned int operator()() {
    unsigned int iteration = 0;
    //unsigned int numNodes = tgraph.size(); // XXX
    unsigned int numNodes = numSmallNeighbors;
    double tol = tolerance;

    std::cout << "target max delta: " << tol << "\n";
    Galois::for_each(tgraphOrder.begin(), tgraphOrder.begin() + numNodes,
        Process(accum, tol, iteration));
    
    return 0;
  }
};


//! Transpose in-edges to out-edges
static void transposeGraph() {
  typedef Galois::Graph::LC_CSR_Graph<size_t, void> InputGraph;
  typedef InputGraph::GraphNode InputNode;
  typedef Galois::Graph::FileGraphWriter OutputGraph;
  typedef OutputGraph::GraphNode OutputNode;

  InputGraph input;
  OutputGraph output;
  Galois::Graph::readGraph(input, inputFilename);

  size_t node_id = 0;
  for (InputNode src : input) {
    input.getData(src) = node_id++;
  }

  output.setNumNodes(input.size());
  output.setNumEdges(input.sizeEdges());
  output.setSizeofEdgeData(sizeof(double));
  output.phase1();

  for (InputNode src : input) {
    size_t sid = input.getData(src);
    assert(sid < input.size());

    //size_t num_neighbors = std::distance(input.edge_begin(src), input.edge_end(src));

    for (InputGraph::edge_iterator edge : input.out_edges(src)) {
      InputNode dst = input.getEdgeDst(edge);
      size_t did = input.getData(dst);
      assert(did < input.size());

      output.incrementDegree(did);
    }
  }

  output.phase2();
  std::vector<double> edgeData;
  edgeData.resize(input.sizeEdges());

  for (InputNode src : input) {
    size_t sid = input.getData(src);
    assert(sid < input.size());

    size_t num_neighbors = std::distance(input.edge_begin(src), input.edge_end(src));

    double w = 1.0/num_neighbors;
    for (InputGraph::edge_iterator edge : input.out_edges(src)) {
      InputNode dst = input.getEdgeDst(edge);
      size_t did = input.getData(dst);
      assert(did < input.size());

      size_t idx = output.addNeighbor(did, sid);
      edgeData[idx] = w;
    }
  }

  double* t = output.finish<double>();
  std::uninitialized_copy(std::make_move_iterator(edgeData.begin()), std::make_move_iterator(edgeData.end()), t);
  
  output.toFile(otherFilename);
  std::cout << "Wrote " << otherFilename << "\n";
}

static void readGraph() {
  Galois::Graph::readGraph(graph, inputFilename);
  Galois::Graph::readGraph(tgraph, otherFilename);

  if (graph.size() != tgraph.size() || graph.sizeEdges() != tgraph.sizeEdges()) {
    std::cerr << "Graph and its transpose have different number of nodes or edges\n";
    abort();
  }

  size_t node_id = 0;
  double initial = 1.0;

  // Zip iterate graph and tgraph together
  Graph::iterator gii = graph.begin(), gei = graph.end();
  TGraph::iterator tii = tgraph.begin(), tei = tgraph.end();
  
  for (; gii != gei; ++node_id, ++gii, ++tii) {
    TNode src = *tii;
    TData& n = tgraph.getData(src);
    memset(n.values, 0, sizeof(n.values));
    n.setPageRank(-1, initial);
    n.id = node_id;
  }
}


//! Make values unique
struct TopPair {
  double value;
  unsigned int id;

  TopPair(double v, unsigned int i): value(v), id(i) { }

  bool operator<(const TopPair& b) const {
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};

static void printTop(int topn, unsigned int iteration) {
  typedef std::map<TopPair,TNode> Top;
  Top top;

  for (TNode src : tgraph) {
    TData& n = tgraph.getData(src);
    double value = n.getPageRank(iteration);
    TopPair key(value, n.id);

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
  for (Top::reverse_iterator ii = top.rbegin(), ei = top.rend(); ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

template<typename A>
unsigned int run() {
  A a;
  return a();
}

unsigned int runAlgo() {
  switch (algo) {
#ifdef USE_POSKI
    case Algo::spmv: return run<SpMVAlgo>();
#endif
    case Algo::dummy: return run<DummyAlgo>();
    case Algo::synchronous: return run<SynchronousAlgo<false, false> >();
    case Algo::nondeterministic: return run<SynchronousAlgo<true, false> >();
    case Algo::serializable: return run<SynchronousAlgo<true, true> >();
    case Algo::asynchronous: return run<AsynchronousAlgo<true,false> >();
    case Algo::serial: return run<SerialAlgo>();
    default:
      std::cerr << "Unknown option\n"; abort();
  }
}

int main(int argc, char **argv) {
  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;

  Galois::StatTimer RT("ReadTime");
  RT.start();
  if (algo == Algo::transpose) {
    transposeGraph();
    RT.stop();
    return 0;
  }

  readGraph();
  tgraphOrder.create(tgraph.size());
  std::copy(tgraph.begin(), tgraph.end(), tgraphOrder.begin());
  if (useOnlySmallDegree) {
    numSmallNeighbors = 
      std::distance(tgraphOrder.begin(),
        std::partition(tgraphOrder.begin(), tgraphOrder.end(), [](TNode x) { return std::distance(tgraph.edge_begin(x), tgraph.edge_end(x)) < 100; }));
    std::cout << "Num Small Neighbors: " << numSmallNeighbors << " (" << numSmallNeighbors / (float) tgraph.size() << ")\n";
  } else {
    numSmallNeighbors = tgraph.size();
  }
  //std::random_shuffle(tgraphOrder.begin(), tgraphOrder.end()); // XXX isolate locality issues
  RT.stop();

  Galois::StatTimer T;
  T.start();
  unsigned int lastIteration = runAlgo();
  T.stop();

  if (!skipVerify) printTop(10, lastIteration);

  return 0;
}
