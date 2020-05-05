/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
static const char* const desc =
    "Computes Matrix Decomposition using Stochastic "
    "Gradient Descent or Alternating Least Squares";
static const char* const url = 0;

enum Algo {
  syncALS,
  simpleALS,
  blockedEdge,
  blockJump,
};

enum Step { bold, bottou, intel, inverse, purdue };

enum OutputType { binary, ascii };

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string>
    outputFilename(cll::Positional, cll::desc("[output file]"), cll::init(""));
static cll::opt<std::string>
    transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<OutputType> outputType(
    "output", cll::desc("Output type:"),
    cll::values(clEnumValN(OutputType::binary, "binary", "Binary"),
                clEnumValN(OutputType::ascii, "ascii", "ASCII"), clEnumValEnd),
    cll::init(OutputType::binary));

// (Purdue, Netflix): 0.05, (Purdue, Yahoo Music): 1.0, (Purdue, HugeWiki): 0.01
// Intel: 0.001
static cll::opt<float> lambda("lambda",
                              cll::desc("regularization parameter [lambda]"),
                              cll::init(0.05));
// (Purdue, Neflix): 0.012, (Purdue, Yahoo Music): 0.00075, (Purdue, HugeWiki):
// 0.001 Intel: 0.001 Bottou: 0.1
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
static cll::opt<float>
    tolerance("tolerance", cll::desc("convergence tolerance"), cll::init(0.01));
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

static cll::opt<Algo>
    algo("algo", cll::desc("Choose an algorithm:"),
         cll::values(clEnumValN(Algo::syncALS, "syncALS",
                                "Alternating least squares"),
                     clEnumValN(Algo::simpleALS, "simpleALS",
                                "Simple alternating least squares"),
                     clEnumValN(Algo::blockedEdge, "blockedEdge",
                                "Edge blocking (default)"),
                     clEnumValN(Algo::blockJump, "blockJump", "Block jumping "),
                     clEnumValEnd),
         cll::init(Algo::blockedEdge));

static cll::opt<Step> learningRateFunction(
    "learningRateFunction", cll::desc("Choose learning rate function:"),
    cll::values(clEnumValN(Step::intel, "intel", "Intel"),
                clEnumValN(Step::purdue, "purdue", "Purdue"),
                clEnumValN(Step::bottou, "bottou", "Bottou"),
                clEnumValN(Step::bold, "bold", "Bold (default)"),
                clEnumValN(Step::inverse, "inverse", "Inverse"), clEnumValEnd),
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
    return learningRate / (1.0 + learningRate * lambda * round);
  }
};

struct InverseStepFunction : public StepFunction {
  virtual std::string name() const { return "Inverse"; }
  virtual LatentValue stepSize(int round) const { return 1.0 / (round + 1); }
};

struct BoldStepFunction : public StepFunction {
  virtual std::string name() const { return "Bold"; }
  virtual bool isBold() const { return true; }
  virtual LatentValue stepSize(int round) const { return 0.0; }
};

template <typename Graph>
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
      g.begin(), g.begin() + NUM_ITEM_NODES, g.begin() + NUM_ITEM_NODES,
      g.end(), itemsPerBlock, usersPerBlock,
      [&](GNode src, GNode dst, typename Graph::edge_iterator edge) {
        LatentValue e =
            predictionError(g.getData(src).latentVector,
                            g.getData(dst).latentVector, g.getEdgeData(edge));
        error += (e * e);
      },
      false);
#endif
  return error.reduce();
}

template <typename Graph>
size_t countEdges(Graph& g) {
  typedef typename Graph::GraphNode GNode;
  galois::GAccumulator<size_t> edges;
  galois::runtime::Fixed2DGraphTiledExecutor<Graph> executor(g);
  executor.execute(
      g.begin(), g.begin() + NUM_ITEM_NODES, g.begin() + NUM_ITEM_NODES,
      g.end(), itemsPerBlock, usersPerBlock,
      [&](GNode src, GNode dst, typename Graph::edge_iterator edge) {
        edges += 1;
      },
      false); // false = no locks
  return edges.reduce();
}

template <typename Graph>
void verify(Graph& g, const std::string& prefix) {
  if (countEdges(g) != g.sizeEdges()) {
    GALOIS_DIE("Error: edge list of input graph probably not sorted");
  }

  double error = sumSquaredError(g);
  double rmse  = std::sqrt(error / g.sizeEdges());

  std::cout << prefix << "RMSE: " << rmse << "\n";
}

template <typename T, unsigned Size>
struct ExplicitFiniteChecker {};

template <typename T>
struct ExplicitFiniteChecker<T, 4U> {
  static_assert(std::numeric_limits<T>::is_iec559, "Need IEEE floating point");
  bool isFinite(T v) {
    union {
      T value;
      uint32_t bits;
    } a = {v};
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

template <typename T>
struct ExplicitFiniteChecker<T, 8U> {
  static_assert(std::numeric_limits<T>::is_iec559, "Need IEEE floating point");
  bool isFinite(T v) {
    union {
      T value;
      uint64_t bits;
    } a = {v};
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

template <typename T>
bool isFinite(T v) {
#ifdef __FAST_MATH__
  return ExplicitFiniteChecker<T, sizeof(T)>().isFinite(v);
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
template <typename Graph, typename Fn>
void executeUntilConverged(const StepFunction& sf, Graph& g, Fn fn) {
  galois::GAccumulator<double> errorAccum;
  std::vector<LatentValue> steps(updatesPerEdge);
  LatentValue last = -1.0;
  int deltaRound   = updatesPerEdge;
  LatentValue rate = learningRate;

  galois::TimeAccumulator elapsed;
  elapsed.start();

  unsigned long lastTime = 0;

  for (int round = 0;; round += deltaRound) {
    if (fixedRounds > 0 && round >= fixedRounds)
      break;
    if (fixedRounds > 0)
      deltaRound = std::min(deltaRound, fixedRounds - round);

    for (int i = 0; i < updatesPerEdge; ++i) {
      // Assume that loss decreases
      if (sf.isBold())
        steps[i] = i == 0 ? rate : steps[i - 1] * 1.05;
      else
        steps[i] = sf.stepSize(round + i);
    }

    // fn(&steps[0], round + deltaRound, useExactError ? NULL : &errorAccum);
    fn(&steps[0], round + deltaRound, useExactError ? &errorAccum : NULL);
    double error = useExactError ? errorAccum.reduce() : sumSquaredError(g);

    elapsed.stop();

    unsigned long curElapsed = elapsed.get();
    elapsed.start();
    unsigned long millis = curElapsed - lastTime;
    lastTime             = curElapsed;

    double gflops = countFlops(g.sizeEdges(), deltaRound, LATENT_VECTOR_SIZE) /
                    millis / 1e6;

    int curRound = round + deltaRound;
    std::cout << "R: " << curRound << " elapsed (ms): " << curElapsed
              << " GFLOP/s: " << gflops;
    if (useExactError) {
      std::cout << " RMSE (R " << curRound
                << "): " << std::sqrt(error / g.sizeEdges());
    } else {
      std::cout << " Approx. RMSE (R " << (curRound - 1)
                << ".5): " << std::sqrt(std::abs(error / g.sizeEdges()));
    }
    std::cout << "\n";

    if (!isFinite(error))
      break;
    // TODO: Should this be std::abs as last can be negavtive if not using
    // squaredError
    if (fixedRounds <= 0 && std::abs(last) >= 0.0 &&
        std::abs((last - error) / last) < tolerance)
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
// uses locks
struct BlockJumpAlgo {
  bool isSgd() const { return true; }
  typedef galois::substrate::PaddedLock<true> SpinLock;
  static const bool precomputeOffsets = false;

  std::string name() const { return "BlockAlgo"; }

  struct Node {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  typedef galois::graphs::LC_CSR_Graph<Node, double>
      //    ::with_numa_alloc<true>::type
      ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  void readGraph(Graph& g) { galois::graphs::readGraph(g, inputFilename); }

  size_t userIdToUserNode(size_t userId) { return userId + NUM_ITEM_NODES; }

  struct BlockInfo {
    size_t id;
    size_t x;
    size_t y;
    size_t userStart;
    size_t userEnd;
    size_t itemStart;
    size_t itemEnd;
    size_t numitems;
    size_t updates;
    double error;
    int* userOffsets;

    std::ostream& print(std::ostream& os) {
      os << "id: " << id << " x: " << x << " y: " << y
         << " userStart: " << userStart << " userEnd: " << userEnd
         << " itemStart: " << itemStart << " itemEnd: " << itemEnd
         << " updates: " << updates << "\n";
      return os;
    }
  };

  struct Process {
    Graph& g;
    SpinLock *xLocks, *yLocks;
    BlockInfo* blocks;
    size_t numXBlocks, numYBlocks;
    LatentValue* steps;
    size_t maxUpdates;
    galois::GAccumulator<double>* errorAccum;

    struct GetDst : public std::unary_function<Graph::edge_iterator, GNode> {
      Graph* g;
      GetDst() {}
      GetDst(Graph* _g) : g(_g) {}
      GNode operator()(Graph::edge_iterator ii) const {
        return g->getEdgeDst(ii);
      }
    };

    /**
     * Preconditions: row and column of slice are locked.
     *
     * Postconditions: increments update count, does sgd update on each item
     * and user in the slice
     */
    template <bool Enable = precomputeOffsets>
    size_t runBlock(BlockInfo& si,
                    typename std::enable_if<!Enable>::type* = 0) {
      typedef galois::NoDerefIterator<Graph::edge_iterator> no_deref_iterator;
      typedef boost::transform_iterator<GetDst, no_deref_iterator>
          edge_dst_iterator;

      LatentValue stepSize = steps[si.updates - maxUpdates + updatesPerEdge];
      size_t seen          = 0;
      double error         = 0.0;

      // Set up item iterators
      size_t itemId      = 0;
      Graph::iterator mm = g.begin(), em = g.begin();
      std::advance(mm, si.itemStart);
      std::advance(em, si.itemEnd);

      GetDst fn{&g};

      // For each item in the range
      for (; mm != em; ++mm, ++itemId) {
        GNode item      = *mm;
        Node& itemData  = g.getData(item);
        size_t lastUser = si.userEnd + NUM_ITEM_NODES;

        edge_dst_iterator start(no_deref_iterator(g.edge_begin(
                                    item, galois::MethodFlag::UNPROTECTED)),
                                fn);
        edge_dst_iterator end(no_deref_iterator(g.edge_end(
                                  item, galois::MethodFlag::UNPROTECTED)),
                              fn);

        // For each edge in the range
        for (auto ii =
                 std::lower_bound(start, end, si.userStart + NUM_ITEM_NODES);
             ii != end; ++ii) {
          GNode user = g.getEdgeDst(*ii.base());

          if (user >= lastUser)
            break;

          LatentValue e = doGradientUpdate(itemData.latentVector,
                                           g.getData(user).latentVector, lambda,
                                           g.getEdgeData(*ii.base()), stepSize);
          if (errorAccum)
            error += e * e;
          ++seen;
        }
      }

      si.updates += 1;
      if (errorAccum) {
        *errorAccum += (error - si.error);
        si.error = error;
      }

      return seen;
    }

    template <bool Enable = precomputeOffsets>
    size_t runBlock(BlockInfo& si, typename std::enable_if<Enable>::type* = 0) {
      LatentValue stepSize = steps[si.updates - maxUpdates + updatesPerEdge];
      size_t seen          = 0;
      double error         = 0.0;

      // Set up item iterators
      size_t itemId      = 0;
      Graph::iterator mm = g.begin(), em = g.begin();
      std::advance(mm, si.itemStart);
      std::advance(em, si.itemEnd);

      // For each item in the range
      for (; mm != em; ++mm, ++itemId) {
        if (si.userOffsets[itemId] < 0)
          continue;

        GNode item      = *mm;
        Node& itemData  = g.getData(item);
        size_t lastUser = si.userEnd + NUM_ITEM_NODES;

        // For each edge in the range
        for (auto ii = g.edge_begin(item) + si.userOffsets[itemId],
                  ei = g.edge_end(item);
             ii != ei; ++ii) {
          GNode user = g.getEdgeDst(ii);

          if (user >= lastUser)
            break;

          LatentValue e = doGradientUpdate(itemData.latentVector,
                                           g.getData(user).latentVector, lambda,
                                           g.getEdgeData(ii), stepSize);
          if (errorAccum)
            error += e * e;
          ++seen;
        }
      }

      si.updates += 1;
      if (errorAccum) {
        *errorAccum += (error - si.error);
        si.error = error;
      }

      return seen;
    }

    /**
     * Searches next slice to work on.
     *
     * @returns slice id to work on, x and y locks are held on the slice
     */
    size_t getNextBlock(BlockInfo* sp) {
      size_t numBlocks   = numXBlocks * numYBlocks;
      size_t nextBlockId = sp->id + 1;
      for (size_t i = 0; i < 2 * numBlocks; ++i, ++nextBlockId) {
        // Wrap around
        if (nextBlockId == numBlocks)
          nextBlockId = 0;

        BlockInfo& nextBlock = blocks[nextBlockId];

        if (nextBlock.updates < maxUpdates && xLocks[nextBlock.x].try_lock()) {
          if (yLocks[nextBlock.y].try_lock()) {
            // Return while holding locks
            return nextBlockId;
          } else {
            xLocks[nextBlock.x].unlock();
          }
        }
      }

      return numBlocks;
    }

    void operator()(unsigned tid, unsigned total) {
      galois::StatTimer timer("PerThreadTime");
      // TODO: Report Accumulators at the end
      galois::GAccumulator<size_t> edgesVisited;
      galois::GAccumulator<size_t> blocksVisited;
      size_t numBlocks = numXBlocks * numYBlocks;
      size_t xBlock    = (numXBlocks + total - 1) / total;
      size_t xStart    = std::min(xBlock * tid, numXBlocks - 1);
      size_t yBlock    = (numYBlocks + total - 1) / total;
      size_t yStart    = std::min(yBlock * tid, numYBlocks - 1);
      BlockInfo* sp    = &blocks[xStart + yStart + numXBlocks];

      timer.start();

      while (true) {
        sp = &blocks[getNextBlock(sp)];
        if (sp == &blocks[numBlocks])
          break;
        blocksVisited += 1;
        edgesVisited += runBlock(*sp);

        xLocks[sp->x].unlock();
        yLocks[sp->y].unlock();
      }

      timer.stop();
    }
  };

  void operator()(Graph& g, const StepFunction& sf) {
    const size_t numUsers = g.size() - NUM_ITEM_NODES;
    const size_t numYBlocks =
        (NUM_ITEM_NODES + itemsPerBlock - 1) / itemsPerBlock;
    const size_t numXBlocks = (numUsers + usersPerBlock - 1) / usersPerBlock;
    const size_t numBlocks  = numXBlocks * numYBlocks;

    SpinLock* xLocks = new SpinLock[numXBlocks];
    SpinLock* yLocks = new SpinLock[numYBlocks];

    std::cout << "itemsPerBlock: " << itemsPerBlock
              << " usersPerBlock: " << usersPerBlock
              << " numBlocks: " << numBlocks << " numXBlocks: " << numXBlocks
              << " numYBlocks: " << numYBlocks << "\n";

    // Initialize
    BlockInfo* blocks = new BlockInfo[numBlocks];
    for (size_t i = 0; i < numBlocks; i++) {
      BlockInfo& si = blocks[i];
      si.id         = i;
      si.x          = i % numXBlocks;
      si.y          = i / numXBlocks;
      si.updates    = 0;
      si.error      = 0.0;
      si.userStart  = si.x * usersPerBlock;
      si.userEnd    = std::min((si.x + 1) * usersPerBlock, numUsers);
      si.itemStart  = si.y * itemsPerBlock;
      si.itemEnd    = std::min((si.y + 1) * itemsPerBlock, NUM_ITEM_NODES);
      si.numitems   = si.itemEnd - si.itemStart;
      if (precomputeOffsets) {
        si.userOffsets = new int[si.numitems];
      } else {
        si.userOffsets = nullptr;
      }
    }

    // Partition item edges in blocks to users according to range [userStart,
    // userEnd)
    if (precomputeOffsets) {
      galois::do_all(galois::iterate(g.begin(), g.begin() + NUM_ITEM_NODES),
                     [&](GNode item) {
                       size_t sliceY = item / itemsPerBlock;
                       BlockInfo* s  = &blocks[sliceY * numXBlocks];

                       size_t pos = item - s->itemStart;
                       auto ii = g.edge_begin(item), ei = g.edge_end(item);
                       size_t offset = 0;
                       for (size_t i = 0; i < numXBlocks; ++i, ++s) {
                         size_t start = userIdToUserNode(s->userStart);
                         size_t end   = userIdToUserNode(s->userEnd);

                         if (ii != ei && g.getEdgeDst(ii) >= start &&
                             g.getEdgeDst(ii) < end) {
                           s->userOffsets[pos] = offset;
                         } else {
                           s->userOffsets[pos] = -1;
                         }
                         for (; ii != ei && g.getEdgeDst(ii) < end;
                              ++ii, ++offset)
                           ;
                       }
                     });
    }

    executeUntilConverged(sf, g,
                          [&](LatentValue* steps, size_t maxUpdates,
                              galois::GAccumulator<double>* errorAccum) {
                            Process fn{g,      xLocks,     yLocks,
                                       blocks, numXBlocks, numYBlocks,
                                       steps,  maxUpdates, errorAccum};
                            galois::on_each(fn);
                          });
  }
};

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
      ::template with_out_of_line_lockable<true>::type ::
          template with_no_lockable<!makeSerializable>::type Graph;

  void readGraph(Graph& g) { galois::graphs::readGraph(g, inputFilename); }

  std::string name() const { return "blockedEdge"; }

  size_t numItems() const { return NUM_ITEM_NODES; }

private:
  using GNode         = typename Graph::GraphNode;
  using edge_iterator = typename Graph::edge_iterator;

  struct Execute {
    Graph& g;
    galois::GAccumulator<unsigned>& edgesVisited;

    void operator()(LatentValue* steps, int maxUpdates,
                    galois::GAccumulator<double>* errorAccum) {
      galois::runtime::Fixed2DGraphTiledExecutor<Graph> executor(g);
      executor.execute(
          g.begin(), g.begin() + NUM_ITEM_NODES, g.begin() + NUM_ITEM_NODES,
          g.end(), itemsPerBlock, usersPerBlock,
          [&](GNode src, GNode dst, edge_iterator edge) {
            // TODO choose one
            // const LatentValue stepSize =
            //  steps[updatesPerEdge - maxUpdates + task.updates];
            // const LatentValue stepSize = steps[1 - maxUpdates + 0];

            const LatentValue stepSize = steps[0];
            LatentValue error          = doGradientUpdate(
                g.getData(src).latentVector, g.getData(dst).latentVector,
                lambda, g.getEdgeData(edge), stepSize);
            edgesVisited += 1;
            if (useExactError)
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

    Execute fn2{g, edgesVisited};
    executeUntilConverged(sf, g, fn2);

    execute.stop();

    galois::runtime::reportStat_Single("BlockedEdgeAlgo", "EdgesVisited",
                                       edgesVisited.reduce());
  }
};

/**
 * ALS algorithms
 */

#ifdef HAS_EIGEN

struct SimpleALSalgo {
  bool isSgd() const { return false; }
  std::string name() const { return "AlternatingLeastSquares"; }
  struct Node {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  typedef typename galois::graphs::LC_CSR_Graph<Node, double>::with_no_lockable<
      true>::type Graph;
  typedef Graph::GraphNode GNode;
  // Column-major access
  typedef Eigen::SparseMatrix<LatentValue> Sp;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, Eigen::Dynamic> MT;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, 1> V;
  typedef Eigen::Map<V> MapV;

  Sp A;
  Sp AT;

  void readGraph(Graph& g) { galois::graphs::readGraph(g, inputFilename); }

  void copyToGraph(Graph& g, MT& WT, MT& HT) {
    // Copy out
    for (GNode n : g) {
      LatentValue* ptr = &g.getData(n).latentVector[0];
      MapV mapV{ptr};
      if (n < NUM_ITEM_NODES) {
        mapV = WT.col(n);
      } else {
        mapV = HT.col(n - NUM_ITEM_NODES);
      }
    }
  }

  void copyFromGraph(Graph& g, MT& WT, MT& HT) {
    for (GNode n : g) {
      LatentValue* ptr = &g.getData(n).latentVector[0];
      MapV mapV{ptr};
      if (n < NUM_ITEM_NODES) {
        WT.col(n) = mapV;
      } else {
        HT.col(n - NUM_ITEM_NODES) = mapV;
      }
    }
  }

  void initializeA(Graph& g) {
    typedef Eigen::Triplet<int> Triplet;
    std::vector<Triplet> triplets{g.sizeEdges()};
    auto it = triplets.begin();
    for (auto n : g) {
      for (auto edge : g.out_edges(n)) {
        *it++ = Triplet(n, g.getEdgeDst(edge) - NUM_ITEM_NODES,
                        g.getEdgeData(edge));
      }
    }
    A.resize(NUM_ITEM_NODES, g.size() - NUM_ITEM_NODES);
    A.setFromTriplets(triplets.begin(), triplets.end());
    AT = A.transpose();
  }

  void operator()(Graph& g, const StepFunction&) {
    galois::TimeAccumulator elapsed;
    elapsed.start();

    // Find W, H that minimize ||W H^T - A||_2^2 by solving alternating least
    // squares problems:
    //   (W^T W + lambda I) H^T = W^T A (solving for H^T)
    //   (H^T H + lambda I) W^T = H^T A^T (solving for W^T)
    MT WT{LATENT_VECTOR_SIZE, NUM_ITEM_NODES};
    MT HT{LATENT_VECTOR_SIZE, g.size() - NUM_ITEM_NODES};
    typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, LATENT_VECTOR_SIZE>
        XTX;
    typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, Eigen::Dynamic> XTSp;
    typedef galois::substrate::PerThreadStorage<XTX> PerThrdXTX;

    galois::gPrint("ALS::Start initializeA\n");
    initializeA(g);
    galois::gPrint("ALS::End initializeA\n");
    galois::gPrint("ALS::Start copyFromGraph\n");
    copyFromGraph(g, WT, HT);
    galois::gPrint("ALS::End copyFromGraph\n");

    double last = -1.0;
    galois::StatTimer mmTime("MMTime");
    galois::StatTimer update1Time("UpdateTime1");
    galois::StatTimer update2Time("UpdateTime2");
    galois::StatTimer copyTime("CopyTime");
    PerThrdXTX xtxs;

    for (int round = 1;; ++round) {
      mmTime.start();
      // TODO parallelize this using tiled executor
      XTSp WTA = WT * A;
      mmTime.stop();

      update1Time.start();
      // TODO: Change to Do_all, pass ints to iterator
      galois::for_each(
          galois::iterate(boost::counting_iterator<int>(0),
                          boost::counting_iterator<int>(A.outerSize())),
          [&](int col, galois::UserContext<int>&) {
            // Compute WTW = W^T * W for sparse A
            XTX& WTW = *xtxs.getLocal();
            WTW.setConstant(0);
            for (Sp::InnerIterator it(A, col); it; ++it)
              WTW.triangularView<Eigen::Upper>() +=
                  WT.col(it.row()) * WT.col(it.row()).transpose();
            for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
              WTW(i, i) += lambda;
            HT.col(col) =
                WTW.selfadjointView<Eigen::Upper>().llt().solve(WTA.col(col));
          });
      update1Time.stop();

      mmTime.start();
      XTSp HTAT = HT * AT;
      mmTime.stop();

      update2Time.start();
      galois::for_each(
          galois::iterate(boost::counting_iterator<int>(0),
                          boost::counting_iterator<int>(AT.outerSize())),
          [&](int col, galois::UserContext<int>&) {
            // Compute HTH = H^T * H for sparse A
            XTX& HTH = *xtxs.getLocal();
            HTH.setConstant(0);
            for (Sp::InnerIterator it(AT, col); it; ++it)
              HTH.triangularView<Eigen::Upper>() +=
                  HT.col(it.row()) * HT.col(it.row()).transpose();
            for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
              HTH(i, i) += lambda;
            WT.col(col) =
                HTH.selfadjointView<Eigen::Upper>().llt().solve(HTAT.col(col));
          });
      update2Time.stop();

      copyTime.start();
      copyToGraph(g, WT, HT);
      copyTime.stop();

      double error = sumSquaredError(g);
      elapsed.stop();
      std::cout << "R: " << round << " elapsed (ms): " << elapsed.get()
                << " RMSE (R " << round
                << "): " << std::sqrt(error / g.sizeEdges()) << "\n";
      elapsed.start();

      if (fixedRounds <= 0 && round > 1 &&
          std::abs((last - error) / last) < tolerance)
        break;
      if (fixedRounds > 0 && round >= fixedRounds)
        break;

      last = error;
    }
  }
};

struct SyncALSalgo {

  bool isSgd() const { return false; }

  std::string name() const { return "AsynchronousAlternatingLeastSquares"; }

  struct Node {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  static const bool NEEDS_LOCKS = false;
  typedef typename galois::graphs::LC_CSR_Graph<Node, double> BaseGraph;
  typedef typename std::conditional<
      NEEDS_LOCKS,
      typename BaseGraph::template with_out_of_line_lockable<true>::type,
      typename BaseGraph::template with_no_lockable<true>::type>::type Graph;
  typedef typename Graph::GraphNode GNode;
  // Column-major access
  typedef Eigen::SparseMatrix<LatentValue> Sp;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, Eigen::Dynamic> MT;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, 1> V;
  typedef Eigen::Map<V> MapV;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, LATENT_VECTOR_SIZE>
      XTX;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, Eigen::Dynamic> XTSp;

  typedef galois::substrate::PerThreadStorage<XTX> PerThrdXTX;
  typedef galois::substrate::PerThreadStorage<V> PerThrdV;

  Sp A;
  Sp AT;

  void readGraph(Graph& g) { galois::graphs::readGraph(g, inputFilename); }

  void copyToGraph(Graph& g, MT& WT, MT& HT) {
    // Copy out
    for (GNode n : g) {
      LatentValue* ptr = &g.getData(n).latentVector[0];
      MapV mapV{ptr};
      if (n < NUM_ITEM_NODES) {
        mapV = WT.col(n);
      } else {
        mapV = HT.col(n - NUM_ITEM_NODES);
      }
    }
  }

  void copyFromGraph(Graph& g, MT& WT, MT& HT) {
    for (GNode n : g) {
      LatentValue* ptr = &g.getData(n).latentVector[0];
      MapV mapV{ptr};
      if (n < NUM_ITEM_NODES) {
        WT.col(n) = mapV;
      } else {
        HT.col(n - NUM_ITEM_NODES) = mapV;
      }
    }
  }

  void initializeA(Graph& g) {
    typedef Eigen::Triplet<int> Triplet;
    std::vector<Triplet> triplets{g.sizeEdges()};
    auto it = triplets.begin();
    for (auto n : g) {
      for (auto edge : g.out_edges(n)) {
        *it++ = Triplet(n, g.getEdgeDst(edge) - NUM_ITEM_NODES,
                        g.getEdgeData(edge));
      }
    }
    A.resize(NUM_ITEM_NODES, g.size() - NUM_ITEM_NODES);
    A.setFromTriplets(triplets.begin(), triplets.end());
    AT = A.transpose();
  }

  void update(Graph& g, size_t col, MT& WT, MT& HT, PerThrdXTX& xtxs,
              PerThrdV& rhs) {
    // Compute WTW = W^T * W for sparse A
    V& r = *rhs.getLocal();
    if (col < NUM_ITEM_NODES) {
      r.setConstant(0);
      // HTAT = HT * AT; r = HTAT.col(col)
      for (Sp::InnerIterator it(AT, col); it; ++it)
        r += it.value() * HT.col(it.row());
      XTX& HTH = *xtxs.getLocal();
      HTH.setConstant(0);
      for (Sp::InnerIterator it(AT, col); it; ++it)
        HTH.triangularView<Eigen::Upper>() +=
            HT.col(it.row()) * HT.col(it.row()).transpose();
      for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
        HTH(i, i) += lambda;
      WT.col(col) = HTH.selfadjointView<Eigen::Upper>().llt().solve(r);
    } else {
      col = col - NUM_ITEM_NODES;
      r.setConstant(0);
      // WTA = WT * A; x = WTA.col(col)
      for (Sp::InnerIterator it(A, col); it; ++it)
        r += it.value() * WT.col(it.row());
      XTX& WTW = *xtxs.getLocal();
      WTW.setConstant(0);
      for (Sp::InnerIterator it(A, col); it; ++it)
        WTW.triangularView<Eigen::Upper>() +=
            WT.col(it.row()) * WT.col(it.row()).transpose();
      for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
        WTW(i, i) += lambda;
      HT.col(col) = WTW.selfadjointView<Eigen::Upper>().llt().solve(r);
    }
  }

  struct NonDetTraits {
    typedef std::tuple<> base_function_traits;
  };

  struct Process {
    struct LocalState {
      LocalState(Process&, galois::PerIterAllocTy&) {}
    };

    struct DeterministicId {
      uintptr_t operator()(size_t x) const { return x; }
    };

    typedef std::tuple<galois::per_iter_alloc, galois::intent_to_read,
                       galois::local_state<LocalState>,
                       galois::det_id<DeterministicId>>
        ikdg_function_traits;
    typedef std::tuple<galois::per_iter_alloc, galois::fixed_neighborhood,
                       galois::local_state<LocalState>,
                       galois::det_id<DeterministicId>>
        add_remove_function_traits;
    typedef std::tuple<> nondet_function_traits;

    SyncALSalgo& self;
    Graph& g;
    MT& WT;
    MT& HT;
    PerThrdXTX& xtxs;
    PerThrdV& rhs;

    Process(SyncALSalgo& self, Graph& g, MT& WT, MT& HT, PerThrdXTX& xtxs,
            PerThrdV& rhs)
        : self(self), g(g), WT(WT), HT(HT), xtxs(xtxs), rhs(rhs) {}

    void operator()(size_t col, galois::UserContext<size_t>& ctx) {
      self.update(g, col, WT, HT, xtxs, rhs);
    }
  };

  void operator()(Graph& g, const StepFunction&) {
    if (!useSameLatentVector) {
      galois::gWarn("Results are not deterministic with different numbers of "
                    "threads unless -useSameLatentVector is true");
    }
    galois::TimeAccumulator elapsed;
    elapsed.start();

    // Find W, H that minimize ||W H^T - A||_2^2 by solving alternating least
    // squares problems:
    //   (W^T W + lambda I) H^T = W^T A (solving for H^T)
    //   (H^T H + lambda I) W^T = H^T A^T (solving for W^T)
    MT WT{LATENT_VECTOR_SIZE, NUM_ITEM_NODES};
    MT HT{LATENT_VECTOR_SIZE, g.size() - NUM_ITEM_NODES};

    initializeA(g);
    copyFromGraph(g, WT, HT);

    double last = -1.0;
    galois::StatTimer updateTime("UpdateTime");
    galois::StatTimer copyTime("CopyTime");
    PerThrdXTX xtxs;
    PerThrdV rhs;

    for (int round = 1;; ++round) {

      updateTime.start();

      typedef galois::worklists::PerThreadChunkLIFO<ALS_CHUNK_SIZE> WL_ty;
      galois::for_each(
          galois::iterate(boost::counting_iterator<size_t>(0),
                          boost::counting_iterator<size_t>(NUM_ITEM_NODES)),
          Process(*this, g, WT, HT, xtxs, rhs), galois::wl<WL_ty>(),
          galois::loopname("syncALS-users"));
      galois::for_each(
          galois::iterate(boost::counting_iterator<size_t>(NUM_ITEM_NODES),
                          boost::counting_iterator<size_t>(g.size())),
          Process(*this, g, WT, HT, xtxs, rhs), galois::wl<WL_ty>(),
          galois::loopname("syncALS-movies"));

      updateTime.stop();

      copyTime.start();
      copyToGraph(g, WT, HT);
      copyTime.stop();

      double error = sumSquaredError(g);
      elapsed.stop();
      std::cout << "R: " << round << " elapsed (ms): " << elapsed.get()
                << " RMSE (R " << round
                << "): " << std::sqrt(error / g.sizeEdges()) << "\n";
      elapsed.start();

      if (fixedRounds <= 0 && round > 1 &&
          std::abs((last - error) / last) < tolerance)
        break;
      if (fixedRounds > 0 && round >= fixedRounds)
        break;

      last = error;
    } // end for
  }
};

#endif // HAS_EIGEN

/**
 * Initializes latent vector with random values and returns basic graph
 * parameters.
 *
 * @tparam Graph type of g
 * @param g Graph to initialize
 * @returns number of item nodes, i.e. nodes with outgoing edges. They should
 * be the first nodes of the graph in memory
 */
template <typename Graph>
size_t initializeGraphData(Graph& g) {
  double top = 1.0 / std::sqrt(LATENT_VECTOR_SIZE);
  galois::substrate::PerThreadStorage<std::mt19937> gen;

#if __cplusplus >= 201103L || defined(HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
  std::uniform_real_distribution<LatentValue> dist(0, top);
#else
  std::uniform_real<LatentValue> dist(0, top);
#endif

  galois::do_all(galois::iterate(g), [&](typename Graph::GraphNode n) {
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
  });

  // Count number of item nodes, i.e. nodes with edges
  size_t numItemNodes = galois::ParallelSTL::count_if(
      g.begin(), g.end(), [&](typename Graph::GraphNode n) -> bool {
        return std::distance(g.edge_begin(n), g.edge_end(n)) != 0;
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

template <typename Graph>
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

template <typename Graph>
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
template <typename Algo>
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

  std::cout << "num users: " << g.size() - NUM_ITEM_NODES
            << " num items: " << NUM_ITEM_NODES
            << " num ratings: " << g.sizeEdges() << "\n";

  std::unique_ptr<StepFunction> sf{newStepFunction()};

  std::cout << "latent vector size: " << LATENT_VECTOR_SIZE
            << " algo: " << algo.name() << " lambda: " << lambda;

  if (algo.isSgd()) {
    std::cout << " learning rate: " << learningRate
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
#ifdef HAS_EIGEN
  case Algo::syncALS:
    run<SyncALSalgo>();
    break;
  case Algo::simpleALS:
    run<SimpleALSalgo>();
    break;
#endif
  case Algo::blockedEdge:
    run<BlockedEdgeAlgo>();
    break;
  case Algo::blockJump:
    run<BlockJumpAlgo>();
    break;
  default:
    GALOIS_DIE("unknown algorithm");
    break;
  }

  return 0;
}
