/* 
 * License:
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
#include "TcpServer.h"

#include "galois/config.h"
#include "galois/Accumulator.h"
#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/Graph/Graph.h"
#include "galois/Graph/LCGraph.h"
#include "galois/ParallelSTL/ParallelSTL.h"
#include "galois/Runtime/ll/PaddedLock.h"
#include "galois/Runtime/ll/EnvCheck.h"
#include "galois/Runtime/TiledExecutor.h"
#include "galois/Runtime/DetSchedules.h"
#include "Lonestar/BoilerPlate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <type_traits>

#ifdef HAS_EIGEN
#include <Eigen/Sparse>
#include <Eigen/Dense>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static const char* const name = "Matrix Completion";
static const char* const desc = "Computes Matrix Decomposition using Stochastic Gradient Descent";
static const char* const url = 0;

enum Algo {
  syncALS,
  asyncALSkdg_i,
  asyncALSkdg_ar,
  asyncALSreuse,
  simpleALS,
  blockedEdge,
  blockedEdgeServer,
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
static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional, cll::desc("[output file]"), cll::init(""));
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<OutputType> outputType("output", cll::desc("Output type:"),
    cll::values(
      clEnumValN(OutputType::binary, "binary", "Binary"),
      clEnumValN(OutputType::ascii, "ascii", "ASCII"),
      clEnumValEnd), cll::init(OutputType::binary));
// (Purdue, Netflix): 0.05, (Purdue, Yahoo Music): 1.0, (Purdue, HugeWiki): 0.01
// Intel: 0.001 
static cll::opt<float> lambda("lambda", cll::desc("regularization parameter [lambda]"), cll::init(0.05));
// (Purdue, Neflix): 0.012, (Purdue, Yahoo Music): 0.00075, (Purdue, HugeWiki): 0.001
// Intel: 0.001
// Bottou: 0.1
static cll::opt<float> learningRate("learningRate",
    cll::desc("learning rate parameter [alpha] for Bold, Bottou, Intel and Purdue step size function"),
    cll::init(0.012));
// (Purdue, Netflix): 0.015, (Purdue, Yahoo Music): 0.01, (Purdue, HugeWiki): 0.0
// Intel: 0.9
static cll::opt<float> decayRate("decayRate", 
    cll::desc("decay rate parameter [beta] for Intel and Purdue step size function"), 
    cll::init(0.015));
static cll::opt<float> tolerance("tolerance", cll::desc("convergence tolerance"), cll::init(0.01));
static cll::opt<int> updatesPerEdge("updatesPerEdge", cll::desc("number of updates per edge"), cll::init(1));
static cll::opt<int> usersPerBlock("usersPerBlock", cll::desc("users per block"), cll::init(2048));
static cll::opt<int> itemsPerBlock("itemsPerBlock", cll::desc("items per block"), cll::init(350));
static cll::opt<int> fixedRounds("fixedRounds", cll::desc("run for a fixed number of rounds"), cll::init(-1));
static cll::opt<bool> useExactError("useExactError", cll::desc("use exact error for testing convergence"), cll::init(false));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
  cll::values(
    clEnumValN(Algo::syncALS, "syncALS", "Alternating least squares"),
    clEnumValN(Algo::simpleALS, "simpleALS", "Simple alternating least squares"),
    clEnumValN(Algo::asyncALSkdg_i, "asyncALSkdg_i", "Asynchronous alternating least squares"),
    clEnumValN(Algo::asyncALSkdg_ar, "asyncALSkdg_ar", "Asynchronous alternating least squares"),
    clEnumValN(Algo::asyncALSreuse, "asyncALSreuse", "Asynchronous alternating least squares"),
    clEnumValN(Algo::blockedEdge, "blockedEdge", "Edge blocking (default)"),
    clEnumValN(Algo::blockedEdgeServer, "blockedEdgeServer", "Edge blocking with server support"),
    clEnumValN(Algo::blockJump, "blockJump", "Block jumping "),
    clEnumValN(Algo::dotProductFixedTiling, "dotProductFixedTiling", "Dot product fixed tiling test"),
    clEnumValN(Algo::dotProductRecursiveTiling, "dotProductRecursiveTiling", "Dot product recursive tiling test"),
  clEnumValEnd), 
  cll::init(Algo::blockedEdge));

static cll::opt<Step> learningRateFunction("learningRateFunction", cll::desc("Choose learning rate function:"),
  cll::values(
    clEnumValN(Step::intel, "intel", "Intel"),
    clEnumValN(Step::purdue, "purdue", "Purdue"),
    clEnumValN(Step::bottou, "bottou", "Bottou"),
    clEnumValN(Step::bold, "bold", "Bold (default)"),
    clEnumValN(Step::inverse, "inverse", "Inverse"),
  clEnumValEnd), 
  cll::init(Step::bold));

static cll::opt<int> serverPort("serverPort", cll::desc("enter server mode on specified port"), cll::init(-1));

static cll::opt<bool> useSameLatentVector("useSameLatentVector",
    cll::desc("initialize all nodes to use same latent vector"),
    cll::init(false));

static cll::opt<int> cutoff("cutoff");

static const unsigned ALS_CHUNK_SIZE = 4;

size_t NUM_ITEM_NODES = 0;

struct PurdueStepFunction: public StepFunction {
  virtual std::string name() const { return "Purdue"; }
  virtual LatentValue stepSize(int round) const {
    return learningRate * 1.5 / (1.0 + decayRate * pow(round + 1, 1.5));
  }
};

struct IntelStepFunction: public StepFunction {
  virtual std::string name() const { return "Intel"; }
  virtual LatentValue stepSize(int round) const {
    return learningRate * pow(decayRate, round);
  }
};

struct BottouStepFunction: public StepFunction {
  virtual std::string name() const { return "Bottou"; }
  virtual LatentValue stepSize(int round) const {
    return learningRate / (1.0 + learningRate*lambda*round);
  }
};

struct InverseStepFunction: public StepFunction {
  virtual std::string name() const { return "Inverse"; }
  virtual LatentValue stepSize(int round) const {
    return 1.0 / (round + 1);
  }
};

struct BoldStepFunction: public StepFunction {
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
      LatentValue e = predictionError(g.getData(n).latentVector, g.getData(dst).latentVector, g.getEdgeData(ii));

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
    LatentValue e = predictionError(g.getData(src).latentVector, g.getData(dst).latentVector, g.getEdgeData(edge));
    error += (e * e);
  }, false);
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
  }, false);
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

    double gflops = countFlops(g.sizeEdges(), deltaRound, LATENT_VECTOR_SIZE) / millis / 1e6;

    int curRound = round + deltaRound;
    std::cout
      << "R: " << curRound
      << " elapsed (ms): " << curElapsed
      << " GFLOP/s: " << gflops;
    if (useExactError) {
      std::cout << " RMSE (R " << curRound << "): " << std::sqrt(error/g.sizeEdges());
    } else {
      std::cout << " Approx. RMSE (R " << (curRound - 1) << ".5): " << std::sqrt(error/g.sizeEdges());
    }
    std::cout << "\n";
    if (!isFinite(error))
      break;
    if (fixedRounds <= 0 && last >= 0.0 && std::abs((last - error) / last) < tolerance)
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

#include "Test.h"

// TODO refactor to use TiledExecutor
// TODO To store previous error on edge, need LC graphs to support different edge data than serialized form 
//! Simple edge-wise operator
template<bool WithServer>
class BlockedEdgeAlgo {
  static const bool makeSerializable = false;
  typedef galois::runtime::LL::PaddedLock<true> SpinLock;

  struct BasicNode {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  struct ServerNode: public BasicNode {
    double sum;
    size_t count;
    bool deleted;
    ServerNode(): sum(0), count(0), deleted(false) { }
  };

  typedef typename boost::mpl::if_c<WithServer, ServerNode, BasicNode>::type Node;

  template<typename NodeData, bool Enabled=WithServer>
  static bool deleted(NodeData& n, typename std::enable_if<Enabled>::type* = 0) {
    return n.deleted;
  }

  template<typename NodeData, bool Enabled=WithServer>
  static bool deleted(NodeData& n, typename std::enable_if<!Enabled>::type* = 0) {
    return false;
  }

public:
  bool isSgd() const { return true; }
  typedef typename galois::graphs::LC_CSR_Graph<Node, unsigned int>
    //::template with_numa_alloc<true>::type
    ::template with_out_of_line_lockable<true>::type
    ::template with_no_lockable<!makeSerializable>::type Graph;

  void readGraph(Graph& g) { galois::graphs::readGraph(g, inputFilename); }

  std::string name() const { return WithServer ? "blockedEdgeServer" : "blockedEdge"; }

  size_t numItems() const { return NUM_ITEM_NODES; }

private:
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::iterator iterator;
  typedef typename Graph::edge_iterator edge_iterator;

  /**
   * Tasks are 2D ranges [start1, end1) x [start2, end2]
   */
  struct Task {
    iterator start1;
    GNode start2;
    iterator end1;
    GNode end2;
    size_t id;
    size_t x;
    size_t y;
    double error;
    int updates;
  };

  struct GetDst: public std::unary_function<edge_iterator, GNode> {
    Graph* g;
    GetDst() { }
    GetDst(Graph* _g): g(_g) { }
    GNode operator()(typename Graph::edge_iterator ii) const {
      return g->getEdgeDst(ii);
    }
  };

  typedef galois::NoDerefIterator<edge_iterator> no_deref_iterator;
  typedef boost::transform_iterator<GetDst, no_deref_iterator> edge_dst_iterator;

  struct Process {
    Graph& g;
    galois::Statistic& edgesVisited;
    galois::Statistic& failures;
    std::vector<SpinLock>& xLocks;
    std::vector<SpinLock>& yLocks;
    std::vector<Task>& tasks;
    LatentValue* steps;
    int maxUpdates;
    galois::GAccumulator<double>* errorAccum;

#if 0
    void updateBlock(Task& task) {
      const int innerCount = std::numeric_limits<int>::max(); // XXX
      const LatentValue stepSize = steps[updatesPerEdge - maxUpdates + task.updates];
      GetDst fn { &g };
      no_deref_iterator xxx;
      std::array<no_deref_iterator,1024> starts;
      double error = 0.0;

      // TODO add round blocking -- Added by not very useful
      // TODO modify edge data to support agnostic edge blocking
      for (int phase = makeSerializable ? 0 : 1; phase < 2; ++phase) {
        galois::MethodFlag flag = phase == 0 ? galois::MethodFlag::ALL : galois::MethodFlag::UNPROTECTED;
        int numWorking;
        int round = 0;
        int limit = 0;
        do {
          numWorking = 0;
          int index = 0;
          for (auto ii = task.start1; ii != task.end1; ++ii, ++index) {
            Node& nn = g.getData(*ii, round == 0 ? flag : galois::MethodFlag::UNPROTECTED);
            Graph::edge_iterator begin = g.edge_begin(*ii, galois::MethodFlag::UNPROTECTED);
            no_deref_iterator nbegin(round == 0 ? no_deref_iterator(begin) : starts[index]);
            no_deref_iterator nend(no_deref_iterator(g.edge_end(*ii, galois::MethodFlag::UNPROTECTED)));
            edge_dst_iterator dbegin(nbegin, fn);
            edge_dst_iterator dend(nend, fn);
            edge_dst_iterator jj = round == 0 ? std::lower_bound(dbegin, dend, task.start2) : dbegin;
            int i = 0;
            bool done = false;
            //for (i = 0; jj != dend && i < innerCount; ++jj, ++i) { // XXX
            for (i = 0; jj != dend; ++jj, ++i) {
              Graph::edge_iterator edge = *jj.base();

              if (g.getEdgeDst(edge) > task.end2) {
                done = true;
                break;
              }
              if (g.getEdgeDst(edge) > task.start2 + limit)
                break;

              Node& mm = g.getData(g.getEdgeDst(edge), flag);
              if (phase == 1) {
                LatentValue e = doGradientUpdate(nn.latentVector, mm.latentVector, static_cast<LatentValue>(g.getEdgeData(edge)), stepSize);
                error += e * e;
                edgesVisited += 1;
              }
            }
            if (done)
              starts[index] = nend;
            else
              starts[index] = jj.base();

            //if (!done && jj != dend && i == innerCount)
            if (!done && jj != dend)
              numWorking += 1;
          }
          round += 1;
          limit += innerCount;
        } while (numWorking > 0);
      }
      task.updates += 1;
      errorAccum += (error - task.error);
      task.error = error;
    }
#endif

    void updateBlock(Task& task) {
//      const int innerCount = std::numeric_limits<int>::max(); // XXX
      const LatentValue stepSize = steps[updatesPerEdge - maxUpdates + task.updates];
      GetDst fn { &g };
      double error = 0.0;

      // TODO modify edge data to support agnostic edge blocking
      for (int phase = makeSerializable ? 0 : 1; phase < 2; ++phase) {
        galois::MethodFlag flag = phase == 0 ? galois::MethodFlag::WRITE : galois::MethodFlag::UNPROTECTED;
        for (auto ii = task.start1; ii != task.end1; ++ii) {
          Node& nn = g.getData(*ii, phase == 0 ? flag : galois::MethodFlag::UNPROTECTED);
          if (deleted(nn))
            continue;
          edge_iterator begin = g.edge_begin(*ii, galois::MethodFlag::UNPROTECTED);
          no_deref_iterator nbegin(begin);
          no_deref_iterator nend(g.edge_end(*ii, galois::MethodFlag::UNPROTECTED));
          edge_dst_iterator dbegin(nbegin, fn);
          edge_dst_iterator dend(nend, fn);
          for (auto jj = std::lower_bound(dbegin, dend, task.start2); jj != dend; ++jj) {
            edge_iterator edge = *jj.base();
            if (g.getEdgeDst(edge) > task.end2)
              break;
            Node& mm = g.getData(g.getEdgeDst(edge), flag);
            if (deleted(mm))
              continue;
            if (phase == 1) {
              LatentValue e = doGradientUpdate(nn.latentVector, mm.latentVector, lambda, g.getEdgeData(edge), stepSize);
              if (errorAccum)
                error += e * e;
              edgesVisited += 1;
            }
          }
        }
      }
      task.updates += 1;
      if (errorAccum) {
        *errorAccum += (error - task.error);
        task.error = error;
      }
    }

    void operator()(Task& task) {
      updateBlock(task);
    }

    void operator()(Task& task, galois::UserContext<Task>& ctx) {
#if 1
      if (std::try_lock(xLocks[task.x], yLocks[task.y]) >= 0) {
        ctx.push(task);
        return;
      }
#endif
      updateBlock(task);
      if (task.updates < maxUpdates)
        ctx.push(task);

#if 1
      xLocks[task.x].unlock();
      yLocks[task.y].unlock();
#endif
    }

    size_t probeBlock(size_t start, size_t by, size_t n, size_t numBlocks) {
      for (size_t i = 0; i < n; ++i, start += by) {
        while (start >= numBlocks)
          start -= numBlocks;
        Task& b = tasks[start];

        // TODO racy
        if (b.updates < maxUpdates) {
          if (std::try_lock(xLocks[b.x], yLocks[b.y]) < 0) {
            // Return while holding locks
            return start;
          }
        }
      }
      failures += 1;
      return numBlocks;
    }

    // Nested Y then X
    size_t nextBlock(size_t start, size_t numBlocks, bool inclusive) {
      const size_t yDelta = xLocks.size();
      const size_t xDelta = 1;
      size_t b;

      // TODO: Add x-then-y and all-x schedules back to show performance impact
      for (int times = 0; times < 2; ++times) {
        size_t yLimit = yLocks.size();
        size_t xLimit = xLocks.size();
        // First iteration is exclusive of start
        if ((b = probeBlock(start + (inclusive ? 0 : yDelta), yDelta, yLimit - (inclusive ? 0 : 1), numBlocks)) != numBlocks)
          return b;
        if ((b = probeBlock(start + (inclusive ? 0 : xDelta), xDelta, xLimit - (inclusive ? 0 : 1), numBlocks)) != numBlocks)
          return b;
        start += yDelta + xDelta;
        while (yLimit > 0 || xLimit > 0) {
          while (start >= numBlocks)
            start -= numBlocks;
          // Subsequent iterations are inclusive of start
          if (yLimit > 0 && (b = probeBlock(start, yDelta, yLimit - 1, numBlocks)) != numBlocks)
            return b;
          if (xLimit > 0 && (b = probeBlock(start, xDelta, xLimit - 1, numBlocks)) != numBlocks)
            return b;
          if (yLimit > 0) {
            yLimit--;
            start += yDelta;
          }
          if (xLimit > 0) {
            xLimit--;
            start += xDelta;
          }
        }
      }

      return numBlocks;
    }

    void operator()(unsigned tid, unsigned total) {
      // TODO see if just randomly picking different starting points is enough
      const size_t numYBlocks = yLocks.size();
      const size_t numXBlocks = xLocks.size();
      const size_t numBlocks = numXBlocks * numYBlocks;
      const size_t xBlock = (numXBlocks + total - 1) / total;
      size_t xStart = std::min(xBlock * tid, numXBlocks - 1);
      const size_t yBlock = (numYBlocks + total - 1) / total;
      size_t yStart = std::min(yBlock * tid, numYBlocks - 1);

      //std::uniform_int_distribution<size_t> distY(0, numYBlocks - 1);
      //std::uniform_int_distribution<size_t> distX(0, numXBlocks - 1);
      //xStart = distX(gen);
      //yStart = distY(gen);

      size_t start = xStart + yStart * numXBlocks;

      for (int i = 0; ; ++i) {
        start = nextBlock(start, numBlocks, i == 0);
        Task* t = &tasks[start];
        if (t == &tasks[numBlocks])
          break;
        //galois::runtime::LL::gInfo("XXX ", tid, " ", t->x, " ", t->y);
        updateBlock(*t);

        xLocks[t->x].unlock();
        yLocks[t->y].unlock();
      }
    }
  };

  struct Inspect {
    Graph& g;
    galois::InsertBag<Task>& initial;
    std::vector<SpinLock>& xLocks;
    std::vector<SpinLock>& yLocks;
    std::vector<Task>& tasks;

    /**
     * Divide edges into bundles (tasks) of approximately the same size M.
     *
     * Works best if graph has been reordered with RCM or similar such that
     * XXX.
     *
     * 2D square tiling. If the graph were dense, we could simply divide the
     * graph in M^1/2 by M^1/2 tiles
     *
     * Since it is sparse *and* we want to minimize number of graph
     * traversals, we read a small square of M^1/4 items by M^1/4 users/out
     * edges under the assumption that the graph is dense. Then, we figure out
     * the actual number of edges in the region and extrapolate that out to
     * figure out the dimensions of whole tile.
     */
    void operator()(unsigned tid, unsigned total) {
      //adaptiveTiling(tid, total);
      fixedTiling(tid, total);
    }

    void fixedTiling(unsigned tid, unsigned total) {
      if (tid != 0)
        return;
      const size_t numUsers = g.size() - NUM_ITEM_NODES;
      const size_t numBlocks0 = (NUM_ITEM_NODES + itemsPerBlock - 1) / itemsPerBlock;
      const size_t numBlocks1 = (numUsers + usersPerBlock - 1) / usersPerBlock;
      const size_t numBlocks = numBlocks0 * numBlocks1;

      std::cout
        << "itemsPerBlock: " << itemsPerBlock
        << " usersPerBlock: " << usersPerBlock
        << " numBlocks: " << numBlocks
        << " numBlocks0: " << numBlocks0
        << " numBlocks1: " << numBlocks1 << "\n";

      xLocks.resize(numBlocks0);
      yLocks.resize(numBlocks1);
      tasks.resize(numBlocks);

      GetDst fn { &g };
      //int maxNnz = std::numeric_limits<int>::min();
      //int minNnz = std::numeric_limits<int>::max();
      //int maxFloats = std::numeric_limits<int>::min();
      //int minFloats = std::numeric_limits<int>::max();

      for (size_t i = 0; i < numBlocks; ++i) {
        Task& task = tasks[i];
        task.x = i % numBlocks0;
        task.y = i / numBlocks0;
        task.id = i;
        task.updates = 0;
        task.error = 0.0;
        task.start1 = g.begin();
        std::tie(task.start1, task.end1) = galois::block_range(g.begin(), g.begin() + NUM_ITEM_NODES, task.x, numBlocks0);
        task.start2 = task.y * usersPerBlock + NUM_ITEM_NODES;
        task.end2 = (task.y + 1) * usersPerBlock + NUM_ITEM_NODES - 1;

        initial.push(task);
      }

      if (false) {
        for (size_t i = 0; i < numBlocks1; ++i) {
          std::mt19937 gen;
          std::cout << (i * numBlocks0) << " " << (i+1) * numBlocks0 << " " << numBlocks << "\n";
          std::shuffle(&tasks[i * numBlocks0], &tasks[(i+1) * numBlocks0], gen);
        }
      }
    }

    void adaptiveTiling(unsigned tid, unsigned total) {
      galois::Statistic numTasks("Tasks");
      galois::Statistic underTasks("UnderTasks");
      galois::Statistic overTasks("OverTasks");

      size_t totalSize = usersPerBlock * (size_t) itemsPerBlock;
      size_t targetSize = static_cast<size_t>(std::max(std::sqrt(totalSize), 1.0));
      size_t sampleSize = static_cast<size_t>(std::max(std::sqrt(targetSize), 1.0));
      iterator cur, end;
      std::tie(cur, end) = galois::block_range(g.begin(), g.begin() + NUM_ITEM_NODES, tid, total);
      //std::tie(cur, end) = galois::block_range(g.begin(), g.end(), tid, total);
      std::vector<edge_iterator> prevStarts;

      while (cur != end) {
        Task task;
        task.start1 = cur;
        task.updates = 0;

        // Sample tile
        size_t sampleNumEdges = 0;
        // NB: both limits are inclusive
        GNode sampleUpperLimit;
        GNode sampleLowerLimit;
        while (sampleNumEdges == 0 && cur != end) {
          for (unsigned i = 0; cur != end && i < sampleSize; ++cur, ++i) {
            if (sampleNumEdges == 0) {
              unsigned j = 0;
              for (auto jj = g.edge_begin(*cur), ej = g.edge_end(*cur); jj != ej && j < sampleSize; ++jj, ++j) {
                GNode dst = g.getEdgeDst(jj);
                if (sampleNumEdges == 0)
                  sampleLowerLimit = dst;
                sampleUpperLimit = dst;
                sampleNumEdges += 1;
              }
            } else {
              for (auto jj = g.edge_begin(*cur), ej = g.edge_end(*cur); jj != ej; ++jj) {
                GNode dst = g.getEdgeDst(jj);
                if (dst > sampleUpperLimit)
                  break;
                if (dst < sampleLowerLimit)
                  sampleLowerLimit = dst;
                sampleNumEdges += 1;
              }
            }
          }
        }

        // TODO: use difference between sampleNumEdges to correct for skew
        // TODO: add "FIFO" abort policy and retest Galois conflict versions

        // Extrapolate tile
        if (sampleNumEdges) {
          // sampleSize : sqrt(sampleNumEdges) :: multiple : targetSize
          double multiple = sampleSize / std::sqrt(sampleNumEdges) * targetSize;
          size_t nodeBlockSize = std::max(std::distance(task.start1, cur) * multiple, 1.0);
          // FIXME(ddn): Only works for graphs where GNodes are ids
          size_t edgeBlockSize = std::max((sampleUpperLimit - sampleLowerLimit + 1) * multiple, 1.0);
          // Testing code
          //nodeBlockSize = itemsPerBlock;
          //edgeBlockSize = usersPerBlock;

          task.end1 = galois::safe_advance(task.start1, end, nodeBlockSize);
          // Adjust lower limit because new range (start1, end1) may include
          // more nodes than sample range
          if (std::distance(task.start1, cur) < std::distance(task.start1, task.end1)) {
            for (auto ii = task.start1; ii != task.end1; ++ii) {
              // Just sample first edge
              for (auto jj = g.edge_begin(*ii), ej = g.edge_end(*ii); jj != ej; ++jj) {
                GNode dst = g.getEdgeDst(jj);
                if (dst < sampleLowerLimit)
                  sampleLowerLimit = dst;
                break;
              }
            }
          }
          task.start2 = sampleLowerLimit;
          task.end2 = task.start2 + edgeBlockSize;

#if 0
          std::cout << "Sampled "
            << (std::distance(task.start1, cur)) 
            << " x "
            << (sampleUpperLimit - sampleLowerLimit + 1)
            << ": edges: " 
            << sampleNumEdges 
            << " multiple: " << multiple
            << "\n";
#endif
          cur = task.end1;
          prevStarts.resize(nodeBlockSize);

          Task newTask = task;
          for (int phase = 0; ; ++phase) {
            size_t actualNumEdges = 0;
            bool someMore = false;
            unsigned i = 0;
            for (auto ii = newTask.start1; ii != newTask.end1; ++ii, ++i) {
              auto jj = phase == 0 ? g.edge_begin(*ii) : prevStarts[i];

              for (auto ej = g.edge_end(*ii); jj != ej; ++jj) {
                GNode dst = g.getEdgeDst(jj);
                if (dst > newTask.end2) {
                  someMore = true;
                  break;
                }
                actualNumEdges += 1;
              }
              prevStarts[i] = jj;
            }
            if (actualNumEdges) {
              numTasks += 1;
              initial.push_back(newTask);
#if 0
              std::cout
                << " tid: " << tid 
                << " phase: " << phase 
                << " actual edges: " << actualNumEdges 
                << " ratio: " << (actualNumEdges / (double) totalSize)
                << " start1: " << *newTask.start1 
                << " end1: " << *newTask.end1
                << " start2: " << newTask.start2
                << " end2: " << newTask.end2
                << "\n"; // XXX
#endif
              if (actualNumEdges < 0.25*totalSize) 
                underTasks += 1;
              else if (actualNumEdges > 4.0 * totalSize)
                overTasks += 1;
            } else if (!someMore) {
              break;
            }
            newTask.start2 = newTask.end2 + 1;
            newTask.end2 = newTask.start2 + edgeBlockSize;
          }
        }
      }
    }
  };

  struct Execute {
    Graph& g;
    galois::Statistic& edgesVisited;

    void operator()(LatentValue* steps, int maxUpdates, galois::GAccumulator<double>* errorAccum) {
      if (errorAccum)
        GALOIS_DIE("not yet implemented");

      galois::runtime::Fixed2DGraphTiledExecutor<Graph> executor(g);
      executor.execute(
          g.begin(), g.begin() + NUM_ITEM_NODES,
          g.begin() + NUM_ITEM_NODES, g.end(),
          itemsPerBlock, usersPerBlock,
          [&](GNode src, GNode dst, typename Graph::edge_iterator edge) {
        if (deleted(g.getData(src)))
          return;
        //const LatentValue stepSize = steps[updatesPerEdge - maxUpdates + task.updates]; XXX
        //const LatentValue stepSize = steps[1 - maxUpdates + 0];
        // const LatentValue stepSize = steps[0];

        // LatentValue e = doGradientUpdate(g.getData(src).latentVector, g.getData(dst).latentVector, lambda, g.getEdgeData(edge), stepSize);
        // XXX non exact error
        //error += (e * e);
        edgesVisited += 1;
      }, true);
    }
  };

public:
  void operator()(Graph& g, const StepFunction& sf) {
    galois::StatTimer inspect("InspectTime");
    inspect.start();
    galois::InsertBag<Task> initial;
    std::vector<SpinLock> xLocks;
    std::vector<SpinLock> yLocks;
    std::vector<Task> tasks;
    Inspect fn1 { g, initial, xLocks, yLocks, tasks };
    galois::on_each(fn1);
    inspect.stop();

    galois::Statistic edgesVisited("EdgesVisited");
    galois::Statistic failures("PopFailures");
    galois::StatTimer execute("ExecuteTime");
    execute.start();

#if 1
    Execute fn2 { g, edgesVisited };
    executeUntilConverged(sf, g, fn2);
#endif
#if 0
    executeUntilConverged(sf, g, [&](LatentValue* steps, int maxUpdates, galois::GAccumulator<double>* errorAccum) {
      Process fn2 { g, edgesVisited, failures, xLocks, yLocks, tasks, steps, maxUpdates, errorAccum };
      // Testing sufficient optimizations by moving towards BlockJump
      //galois::for_each(initial.begin(), initial.end(), fn2, galois::wl<galois::worklists::dChunkedFIFO<1>>());
      //galois::for_each_local(initial, fn2, galois::wl<galois::worklists::dChunkedFIFO<1>>());
      //galois::do_all_local(initial, fn2, galois::wl<galois::worklists::dChunkedLIFO<1>>());
      galois::on_each(fn2);
      //TODO: delete when racy fix is in
      if (!std::all_of(tasks.begin(), tasks.end(), [maxUpdates](Task& t) { return t.updates == maxUpdates; }))
        std::cerr << "WARNING: Missing tasks\n";
    });
#endif
    execute.stop();
  }
};

//! Initializes latent vector with random values and returns basic graph parameters
template<typename Graph>
size_t initializeGraphData(Graph& g) {
  double top = 1.0/std::sqrt(LATENT_VECTOR_SIZE);

  galois::runtime::PerThreadStorage<std::mt19937> gen;

#if __cplusplus >= 201103L || defined(HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
  std::uniform_real_distribution<LatentValue> dist(0, top);
#else
  std::uniform_real<LatentValue> dist(0, top);
#endif

  galois::do_all_local(g, [&](typename Graph::GraphNode n) {
    auto& data = g.getData(n);

    if (useSameLatentVector) {
      std::mt19937 sameGen;
      for (int i = 0; i < LATENT_VECTOR_SIZE; i++)
        data.latentVector[i] = dist(sameGen);
    } else {
      for (int i = 0; i < LATENT_VECTOR_SIZE; i++)
        data.latentVector[i] = dist(*gen.getLocal());
    }
  });

  size_t numItemNodes = galois::ParallelSTL::count_if(g.begin(), g.end(), [&](typename Graph::GraphNode n) -> bool {
    return std::distance(g.edge_begin(n), g.edge_end(n)) != 0;
  });

  return numItemNodes;
}


StepFunction* newStepFunction() {
  switch (learningRateFunction) {
    case Step::intel: return new IntelStepFunction;
    case Step::purdue: return new PurdueStepFunction;
    case Step::bottou: return new BottouStepFunction;
    case Step::inverse: return new InverseStepFunction;
    case Step::bold: return new BoldStepFunction;
  }
  GALOIS_DIE("unknown step function");
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


template<typename Algo>
void run() {
  typename Algo::Graph g;
  Algo algo;

  galois::runtime::reportNumaAlloc("NumaAlloc0");

  // Represent bipartite graph in general graph data structure:
  //  * items are the first m nodes
  //  * users are the next n nodes
  //  * only items have outedges
  algo.readGraph(g);

  galois::runtime::reportNumaAlloc("NumaAlloc1");

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
      case OutputType::binary: writeBinaryLatentVectors(g, outputFilename); break;
      case OutputType::ascii: writeAsciiLatentVectors(g, outputFilename); break;
      default: abort();
    }
  }

  if (serverPort >= 0) {
    startServer(algo, g, serverPort, std::cerr);
  }

  galois::runtime::reportNumaAlloc("NumaAlloc");
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  galois::StatManager statManager;

  switch (algo) {
#ifdef HAS_EIGEN
    case Algo::syncALS: run<AsyncALSalgo<syncALS>>(); break;
    case Algo::simpleALS: run<SimpleALSalgo>(); break;
    case Algo::asyncALSkdg_i: run<AsyncALSalgo<asyncALSkdg_i>>(); break;
    case Algo::asyncALSkdg_ar: run<AsyncALSalgo<asyncALSkdg_ar>>(); break;
    case Algo::asyncALSreuse: run<AsyncALSalgo<asyncALSreuse>>(); break;
#endif
    case Algo::blockedEdge: run<BlockedEdgeAlgo<false> >(); break;
    case Algo::blockedEdgeServer: run<BlockedEdgeAlgo<true> >(); break;
    case Algo::blockJump: run<BlockJumpAlgo>(); break;
    case Algo::dotProductFixedTiling: run<DotProductFixedTilingAlgo>(); break;
    case Algo::dotProductRecursiveTiling: run<DotProductRecursiveTilingAlgo>(); break;
    default: GALOIS_DIE("unknown algorithm"); break;
  }

  return 0;
}
