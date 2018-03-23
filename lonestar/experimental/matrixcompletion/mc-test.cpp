/*
 * License:
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 * Stochastic gradient descent for matrix factorization, implemented with Galois.
 *
 * Author: Prad Nelluru <pradn@cs.utexas.edu>
 * Author: Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "MC.h"

#include "galois/runtime/TiledExecutor.h"
#include "galois/ParallelSTL.h"
#include "galois/graphs/Graph.h"
#include "Lonestar/BoilerPlate.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>

#ifdef HAS_EIGEN
#include <Eigen/Sparse>
#include <Eigen/Dense>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static const char* const name = "Matrix Completion";
static const char* const desc = "Computes Matrix Decomposition using Stochastic "
                                "Gradient Descent or Alternating Least Squares";
static const char* const url = 0;

enum Algo {
  syncALS,
  simpleALS,
  blockedEdge,
  blockJump,
  dotProductFixedTiling,
  dotProductRecursiveTiling
};

enum Step {
  bold,
  bottou,
  intel,
  inverse,
  purdue
};

enum OutputType {
  binary,
  ascii
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputFilename(cll::Positional, 
                                           cll::desc("<input file>"),
                                           cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional, 
                                            cll::desc("[output file]"),
                                            cll::init(""));
static cll::opt<std::string> transposeGraphName("graphTranspose", 
  cll::desc("Transpose of input graph"));
static cll::opt<OutputType> outputType("output", 
                                       cll::desc("Output type:"),
                                       cll::values(
                                         clEnumValN(OutputType::binary, 
                                                    "binary", "Binary"),
                                         clEnumValN(OutputType::ascii, 
                                                    "ascii", "ASCII"),
                                       clEnumValEnd), 
                                       cll::init(OutputType::binary));

// (Purdue, Netflix): 0.05, (Purdue, Yahoo Music): 1.0, (Purdue, HugeWiki): 0.01
// Intel: 0.001 
static cll::opt<float> lambda("lambda", 
                              cll::desc("regularization parameter [lambda]"), 
                              cll::init(0.05));
// (Purdue, Neflix): 0.012, (Purdue, Yahoo Music): 0.00075, (Purdue, HugeWiki): 0.001
// Intel: 0.001
// Bottou: 0.1
static cll::opt<float> learningRate("learningRate",
                                    cll::desc("learning rate parameter [alpha] "
                                              "for Bold, Bottou, Intel and "
                                              "Purdue step size function"),
                                    cll::init(0.012));
// (Purdue, Netflix): 0.015, (Purdue, Yahoo Music): 0.01, 
// (Purdue, HugeWiki): 0.0, Intel: 0.9
static cll::opt<float> decayRate("decayRate", 
                                 cll::desc("decay rate parameter [beta] for "
                                           "Intel and Purdue step size "
                                           "function"), 
                                 cll::init(0.015));
static cll::opt<float> tolerance("tolerance", 
                                 cll::desc("convergence tolerance"), 
                                 cll::init(0.01));
static cll::opt<int> updatesPerEdge("updatesPerEdge", 
                                    cll::desc("number of updates per edge"), 
                                    cll::init(1));
static cll::opt<int> usersPerBlock("usersPerBlock", 
                                   cll::desc("users per block"), 
                                   cll::init(2048));
static cll::opt<int> itemsPerBlock("itemsPerBlock", 
                                   cll::desc("items per block"), 
                                   cll::init(350));
static cll::opt<int> fixedRounds("fixedRounds", 
                                 cll::desc("run for a fixed number of rounds"), 
                                 cll::init(-1));
static cll::opt<bool> useExactError("useExactError", 
                                    cll::desc("use exact error for testing "
                                              "convergence"), 
                                    cll::init(false));

static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
  cll::values(
    clEnumValN(Algo::syncALS, "syncALS", "Alternating least squares"),
    clEnumValN(Algo::simpleALS, "simpleALS", "Simple alternating least squares"),
    clEnumValN(Algo::blockedEdge, "blockedEdge", "Edge blocking (default)"),
    clEnumValN(Algo::blockJump, "blockJump", "Block jumping "),
    clEnumValN(Algo::dotProductFixedTiling, "dotProductFixedTiling", 
               "Dot product fixed tiling test"),
    clEnumValN(Algo::dotProductRecursiveTiling, "dotProductRecursiveTiling", 
               "Dot product recursive tiling test"),
  clEnumValEnd), 
  cll::init(Algo::blockedEdge));

static cll::opt<Step> learningRateFunction("learningRateFunction", 
  cll::desc("Choose learning rate function:"),
  cll::values(
    clEnumValN(Step::intel, "intel", "Intel"),
    clEnumValN(Step::purdue, "purdue", "Purdue"),
    clEnumValN(Step::bottou, "bottou", "Bottou"),
    clEnumValN(Step::bold, "bold", "Bold (default)"),
    clEnumValN(Step::inverse, "inverse", "Inverse"),
  clEnumValEnd), 
  cll::init(Step::bold));

static cll::opt<bool> useSameLatentVector("useSameLatentVector",
                                          cll::desc("initialize all nodes to "
                                                    "use same latent vector"),
                                          cll::init(false));

static cll::opt<int> cutoff("cutoff");

static const unsigned ALS_CHUNK_SIZE = 4;

size_t NUM_ITEM_NODES = 0;

struct PurdueStepFunction : public StepFunction {
  virtual std::string name() const { return "Purdue"; }
  virtual LatentValue stepSize(int round) const {
    return learningRate * 1.5 / (1.0 + decayRate * pow(round + 1, 1.5));
  }
};

struct IntelStepFunction : public StepFunction {
  virtual std::string name() const { return "Intel"; }
  virtual LatentValue stepSize(int round) const {
    return learningRate * pow(decayRate, round);
  }
};

struct BottouStepFunction : public StepFunction {
  virtual std::string name() const { return "Bottou"; }
  virtual LatentValue stepSize(int round) const {
    return learningRate / (1.0 + learningRate*lambda*round);
  }
};

struct InverseStepFunction : public StepFunction {
  virtual std::string name() const { return "Inverse"; }
  virtual LatentValue stepSize(int round) const {
    return 1.0 / (round + 1);
  }
};

struct BoldStepFunction : public StepFunction {
  virtual std::string name() const { return "Bold"; }
  virtual bool isBold() const { return true; }
  virtual LatentValue stepSize(int round) const { return 0.0; }
};

template<typename Graph>
double sumSquaredError(Graph& g) {
  typedef typename Graph::GraphNode GNode;
  // computing Root Mean Square Error
  // Assuming only item nodes have edges
  galois::GAccumulator<double> error;

  // Save for performance testing
#if 0
  galois::do_all(g.begin(), g.begin() + NUM_ITEM_NODES, [&](GNode n) {
    for (auto ii = g.edge_begin(n), ei = g.edge_end(n); ii != ei; ++ii) {
      GNode dst = g.getEdgeDst(ii);
      LatentValue e = predictionError(g.getData(n).latentVector, 
                                      g.getData(dst).latentVector, 
                                      g.getEdgeData(ii));
      error += (e * e);
    }
  });
#else
  galois::runtime::Fixed2DGraphTiledExecutor<Graph> executor(g);
  executor.execute(
    g.begin(), g.begin() + NUM_ITEM_NODES,
    g.begin() + NUM_ITEM_NODES, g.end(),
    itemsPerBlock, usersPerBlock,
    [&](GNode src, GNode dst, typename Graph::edge_iterator edge) {
      LatentValue e = predictionError(g.getData(src).latentVector, 
                                      g.getData(dst).latentVector, 
                                      g.getEdgeData(edge));
      error += (e * e);
    }, 
    false
  );
#endif
  return error.reduce();
}

template<typename Graph>
size_t countEdges(Graph& g) {
  typedef typename Graph::GraphNode GNode;
  galois::GAccumulator<size_t> edges;
  galois::runtime::Fixed2DGraphTiledExecutor<Graph> executor(g);
  executor.execute(
    g.begin(), g.begin() + NUM_ITEM_NODES,
    g.begin() + NUM_ITEM_NODES, g.end(),
    itemsPerBlock, usersPerBlock,
    [&](GNode src, GNode dst, typename Graph::edge_iterator edge) {
      edges += 1;
    }, 
    false
  ); // false = no locks
  return edges.reduce();
}

template<typename Graph>
void verify(Graph& g, const std::string& prefix) {
  if (countEdges(g) != g.sizeEdges()) {
    GALOIS_DIE("Error: edge list of input graph probably not sorted");
  }

  double error = sumSquaredError(g);
  double rmse = std::sqrt(error/g.sizeEdges());

  std::cout << prefix << "RMSE: " << rmse << "\n";
}

template<typename T, unsigned Size>
struct ExplicitFiniteChecker { };

template<typename T>
struct ExplicitFiniteChecker<T,4U> {
  static_assert(std::numeric_limits<T>::is_iec559, "Need IEEE floating point");
  bool isFinite(T v) {
    union { T value; uint32_t bits; } a = { v };
    if (a.bits == 0x7F800000) {
      return false; // +inf
    } else if (a.bits == 0xFF800000) {
      return false; // -inf
    } else if (a.bits >= 0x7F800001 && a.bits <= 0x7FBFFFFF) {
      return false; // signaling NaN
    } else if (a.bits >= 0xFF800001 && a.bits <= 0xFFBFFFFF) {
      return false; // signaling NaN
    } else if (a.bits >= 0x7FC00000 && a.bits <= 0x7FFFFFFF) {
      return false; // quiet NaN
    } else if (a.bits >= 0xFFC00000 && a.bits <= 0xFFFFFFFF) {
      return false; // quiet NaN
    }
    return true;
  }
};

template<typename T>
struct ExplicitFiniteChecker<T,8U> {
  static_assert(std::numeric_limits<T>::is_iec559, "Need IEEE floating point");
  bool isFinite(T v) {
    union { T value; uint64_t bits; } a = { v };
    if (a.bits == 0x7FF0000000000000) {
      return false; // +inf
    } else if (a.bits == 0xFFF0000000000000) {
      return false; // -inf
    } else if (a.bits >= 0x7FF0000000000001 && a.bits <= 0x7FF7FFFFFFFFFFFF) {
      return false; // signaling NaN
    } else if (a.bits >= 0xFFF0000000000001 && a.bits <= 0xFFF7FFFFFFFFFFFF) {
      return false; // signaling NaN
    } else if (a.bits >= 0x7FF8000000000000 && a.bits <= 0x7FFFFFFFFFFFFFFF) {
      return false; // quiet NaN
    } else if (a.bits >= 0xFFF8000000000000 && a.bits <= 0xFFFFFFFFFFFFFFFF) {
      return false; // quiet NaN
    }
    return true;
  }
};


template<typename T>
bool isFinite(T v) {
#ifdef __FAST_MATH__
  return ExplicitFiniteChecker<T,sizeof(T)>().isFinite(v);
#else
  return std::isfinite(v);
#endif
}

double countFlops(size_t nnz, int rounds, int k) {
  double flop = 0;
  if (useExactError) {
    // dotProduct = 2K, square = 1, sum = 1
    flop += nnz * (2.0 * k + 1 + 1);
  } else {
    // Computed during gradient update: square = 1, sum = 1
    flop += nnz * (1 + 1);
  }
  // dotProduct = 2K, gradient = 10K, 
  flop += rounds * (nnz * (12.0 * k));
  return flop;
}

/**
 * TODO
 *
 */
template<typename Graph, typename Fn>
void executeUntilConverged(const StepFunction& sf, Graph& g, Fn fn) {
  galois::GAccumulator<double> errorAccum;
  std::vector<LatentValue> steps(updatesPerEdge);
  LatentValue last = -1.0;
  int deltaRound = updatesPerEdge;
  LatentValue rate = learningRate;

  galois::TimeAccumulator elapsed;
  elapsed.start();

  unsigned long lastTime = 0;

  for (int round = 0; ; round += deltaRound) {
    if (fixedRounds > 0 && round >= fixedRounds)
      break;
    if (fixedRounds > 0) 
      deltaRound = std::min(deltaRound, fixedRounds - round);
    
    for (int i = 0; i < updatesPerEdge; ++i) {
      // Assume that loss decreases
      if (sf.isBold())
        steps[i] = i == 0 ? rate : steps[i-1] * 1.05;
      else
        steps[i] = sf.stepSize(round + i);
    }

    fn(&steps[0], round + deltaRound, useExactError ? NULL : &errorAccum);
    double error = useExactError ? sumSquaredError(g) : errorAccum.reduce();

    elapsed.stop();

    unsigned long curElapsed = elapsed.get();
    elapsed.start();
    unsigned long millis = curElapsed - lastTime;
    lastTime = curElapsed;

    double gflops = countFlops(g.sizeEdges(), deltaRound, LATENT_VECTOR_SIZE) /
                    millis / 1e6;

    int curRound = round + deltaRound;
    std::cout
      << "R: " << curRound
      << " elapsed (ms): " << curElapsed
      << " GFLOP/s: " << gflops;
    if (useExactError) {
      std::cout << " RMSE (R " << curRound << "): " << std::sqrt(error/g.sizeEdges());
    } else {
      std::cout << " Approx. RMSE (R " << (curRound - 1) << ".5): " << std::sqrt(std::abs(error/g.sizeEdges()));
    }
    std::cout << "\n";

    if (!isFinite(error))
      break;
    //TODO: Should this be std::abs as last can be negavtive if not using squaredError
    if (fixedRounds <= 0 && std::abs(last) >= 0.0 && std::abs((last - error) / last) < tolerance)
      break;
    if (sf.isBold()) {
      // Assume that loss decreases first round
      if (last >= 0.0 && last < error)
        rate = steps[deltaRound - 1] * 0.5;
      else
        rate = steps[deltaRound - 1] * 1.05;
    }
    last = error;
  }
}

//#include "Test2.h"

//! Simple edge-wise operator
class BlockedEdgeAlgo {
  static const bool makeSerializable = false;

  struct BasicNode {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  using Node = BasicNode;

 public:
  bool isSgd() const { return true; }

  typedef typename galois::graphs::LC_CSR_Graph<Node, double>
    //::template with_numa_alloc<true>::type
    ::template with_out_of_line_lockable<true>::type
    ::template with_no_lockable<!makeSerializable>::type Graph;

  void readGraph(Graph& g) { galois::graphs::readGraph(g, inputFilename); }

  std::string name() const { return "blockedEdge"; }

  size_t numItems() const { return NUM_ITEM_NODES; }

 private:
  using GNode = typename Graph::GraphNode;
  using edge_iterator = typename Graph::edge_iterator;

  struct Execute {
    Graph& g;
    galois::GAccumulator<unsigned>& edgesVisited;

    void operator()(LatentValue* steps, int maxUpdates,
                    galois::GAccumulator<double>* errorAccum) {
      galois::runtime::Fixed2DGraphTiledExecutor<Graph> executor(g);
      executor.execute(
        g.begin(), g.begin() + NUM_ITEM_NODES,
        g.begin() + NUM_ITEM_NODES, g.end(),
        itemsPerBlock, usersPerBlock,
        [&](GNode src, GNode dst, edge_iterator edge) {
          // TODO choose one
          //const LatentValue stepSize =
          //  steps[updatesPerEdge - maxUpdates + task.updates];
          //const LatentValue stepSize = steps[1 - maxUpdates + 0];

          //TODO: Previous value
          //const LatentValue stepSize = 0.5;
          const LatentValue stepSize = steps[0];
          LatentValue error =
            doGradientUpdate(g.getData(src).latentVector,
                             g.getData(dst).latentVector, lambda,
                             g.getEdgeData(edge), stepSize);
          edgesVisited += 1;
          if(!useExactError)
            *errorAccum += error;
        },
        true // use locks
      );

    }
  };

 public:
  void operator()(Graph& g, const StepFunction& sf) {
    galois::GAccumulator<unsigned> edgesVisited;

    galois::StatTimer execute("ExecuteTime");
    execute.start();

    Execute fn2 { g, edgesVisited };
    executeUntilConverged(sf, g, fn2);

    execute.stop();

    galois::runtime::reportStat_Single("BlockedEdgeAlgo", "EdgesVisited", 
                                       edgesVisited.reduce());
  }
};


/** 
 * Initializes latent vector with random values and returns basic graph 
 * parameters.
 *
 * @tparam Graph type of g
 * @param g Graph to initialize
 * @returns number of item nodes, i.e. nodes with outgoing edges. They should
 * be the first nodes of the graph in memory
 */
template<typename Graph>
size_t initializeGraphData(Graph& g) {
  double top = 1.0 / std::sqrt(LATENT_VECTOR_SIZE);
  galois::substrate::PerThreadStorage<std::mt19937> gen;

  #if __cplusplus >= 201103L || defined(HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
  std::uniform_real_distribution<LatentValue> dist(0, top);
  #else
  std::uniform_real<LatentValue> dist(0, top);
  #endif

  galois::do_all(galois::iterate(g),
    [&](typename Graph::GraphNode n) {
      auto& data = g.getData(n);

      // all threads initialize their assignment with same generator or
      // a thread local one
      if (useSameLatentVector) {
        std::mt19937 sameGen;
        for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
          data.latentVector[i] = dist(sameGen);
        }
      } else {
        for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
          data.latentVector[i] = dist(*gen.getLocal());
        }
      }
    }
  );

  // Count number of item nodes, i.e. nodes with edges
  size_t numItemNodes = galois::ParallelSTL::count_if(g.begin(), g.end(), 
                        [&](typename Graph::GraphNode n) -> bool {
                          return std::distance(g.edge_begin(n), 
                                               g.edge_end(n)) != 0;
                        });

  return numItemNodes;
}

StepFunction* newStepFunction() {
  switch (learningRateFunction) {
    case Step::intel: 
      return new IntelStepFunction;
    case Step::purdue: 
      return new PurdueStepFunction;
    case Step::bottou: 
      return new BottouStepFunction;
    case Step::inverse: 
      return new InverseStepFunction;
    case Step::bold: 
      return new BoldStepFunction;
    default:
      GALOIS_DIE("unknown step function");
  }
}

template<typename Graph>
void writeBinaryLatentVectors(Graph& g, const std::string& filename) {
  std::ofstream file(filename);
  for (auto ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
    auto& v = g.getData(*ii).latentVector;
    for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
      file.write(reinterpret_cast<char*>(&v[i]), sizeof(v[i]));
    }
  }
  file.close();
}

template<typename Graph>
void writeAsciiLatentVectors(Graph& g, const std::string& filename) {
  std::ofstream file(filename);
  for (auto ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
    auto& v = g.getData(*ii).latentVector;
    for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
      file << v[i] << " ";
    }
    file << "\n";
  }
  file.close();
}


/**
 * Run the provided algorithm (provided through the template argument).
 *
 * @param Algo algorithm to run
 */
template<typename Algo>
void run() {
  typename Algo::Graph g;
  Algo algo;

  galois::runtime::reportNumaAlloc("NumaAlloc0");

  // Bipartite graph in general graph data structure should be following:
  // * items are the first m nodes
  // * users are the next n nodes
  // * only items have outedges
  algo.readGraph(g);

  galois::runtime::reportNumaAlloc("NumaAlloc1");

  // initialize latent vectors and get number of item nodes
  NUM_ITEM_NODES = initializeGraphData(g);

  galois::runtime::reportNumaAlloc("NumaAlloc2");

  std::cout 
    << "num users: " << g.size() - NUM_ITEM_NODES 
    << " num items: " << NUM_ITEM_NODES 
    << " num ratings: " << g.sizeEdges()
    << "\n";

  std::unique_ptr<StepFunction> sf { newStepFunction() };

  std::cout
    << "latent vector size: " << LATENT_VECTOR_SIZE
    << " algo: " << algo.name()
    << " lambda: " << lambda;

  if (algo.isSgd()) {
    std::cout
      << " learning rate: " << learningRate
      << " decay rate: " << decayRate
      << " step function: " << sf->name();
  }

  std::cout << "\n";
      
  if (!skipVerify) {
    verify(g, "Initial");
  }

  // algorithm call
  galois::StatTimer timer;
  timer.start();
  algo(g, *sf);
  timer.stop();

  if (!skipVerify) {
    verify(g, "Final");
  }

  if (outputFilename != "") {
    std::cout << "Writing latent vectors to " << outputFilename << "\n";
    switch (outputType) {
      case OutputType::binary: 
        writeBinaryLatentVectors(g, outputFilename);
        break;
      case OutputType::ascii: 
        writeAsciiLatentVectors(g, outputFilename);
        break;
      default:
        GALOIS_DIE("Invalid output type for latent vector output");
    }
  }

  galois::runtime::reportNumaAlloc("NumaAlloc");
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  switch (algo) {
    //#ifdef HAS_EIGEN
    //case Algo::syncALS:
    //  run<AsyncALSalgo<syncALS>>();
    //  break;
    //case Algo::simpleALS:
    //  run<SimpleALSalgo>();
    //  break;
    //#endif
    case Algo::blockedEdge:
      run<BlockedEdgeAlgo>();
      break;
    //case Algo::dotProductFixedTiling:
    //  run<DotProductFixedTilingAlgo>();
    //  break;
    //case Algo::dotProductRecursiveTiling:
    //  run<DotProductRecursiveTilingAlgo>();
    //  break;
    default: 
      GALOIS_DIE("unknown algorithm");
      break;
  }

  return 0;
}
