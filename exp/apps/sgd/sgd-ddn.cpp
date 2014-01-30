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
#include "Galois/config.h"
#include "Galois/Accumulator.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/Graph.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Lonestar/BoilerPlate.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <type_traits>

#include <ostream>
#include <fstream>

static const char* const name = "Stochastic Gradient Descent";
static const char* const desc = "Computes Matrix Decomposition using Stochastic Gradient Descent";
static const char* const url = "sgd";

typedef double LatentValue;
static const int LATENT_VECTOR_SIZE = 100; // Purdue, CSGD: 100; Intel: 20

enum Algo {
  edgeMovie,
  blockJump
};

enum Step {
  Intel,
  Purdue,
  Bottou,
  Bold,
  Inv
};

enum OutputType {
  binary,
  ascii
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional, cll::desc("[output file]"), cll::init(""));
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
static cll::opt<int> updatesPerEdge("updatesPerEdge", cll::desc("number of updates per edge"), cll::init(1));
static cll::opt<int> usersPerBlock("usersPerBlock", cll::desc("users per block"), cll::init(2048));
static cll::opt<int> moviesPerBlock("moviesPerBlock", cll::desc("movies per block"), cll::init(350));
static cll::opt<int> fixedRounds("fixedRounds", cll::desc("run for a fixed number of rounds"), cll::init(-1));
static cll::opt<bool> useExactError("useExactError", cll::desc("use exact error for testing convergence"), cll::init(false));
static cll::opt<Algo> algo(cll::desc("Choose an algorithm:"),
  cll::values(
    clEnumVal(edgeMovie, "Edgewise"),
    clEnumVal(blockJump, "Block jumping (default)"),
  clEnumValEnd), 
  cll::init(blockJump));
static cll::opt<Step> learningRateFunction("learningRateFunction", cll::desc("Choose learning rate function:"),
  cll::values(
    clEnumVal(Intel, "Intel"),
    clEnumVal(Purdue, "Purdue (default)"),
    clEnumVal(Bottou, "Bottou"),
    clEnumVal(Bold, "Bold"),
    clEnumVal(Inv, "Inverse"),
  clEnumValEnd), 
  cll::init(Purdue));

typedef Galois::Runtime::LL::PaddedLock<true> SpinLock;

size_t NUM_MOVIE_NODES = 0;
size_t NUM_USER_NODES = 0;
size_t NUM_RATINGS = 0;

// like std::inner_product but rewritten here to check vectorization
template<typename T>
T innerProduct(
    T* __restrict__ first1,
    T* __restrict__ last1,
    T* __restrict__ first2,
    T init) {
  assert(first1 + LATENT_VECTOR_SIZE == last1);
  for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
    init += first1[i] * first2[i];
  return init;
}

template<typename T>
T predictionError(
    T* __restrict__ movieLatent,
    T* __restrict__ userLatent,
    T init)
{
  return innerProduct(movieLatent, movieLatent + LATENT_VECTOR_SIZE, userLatent, init);
}

// Objective: squared loss with weighted-square-norm regularization
template<typename T>
T doGradientUpdate(
    T* __restrict__ movieLatent,
    T* __restrict__ userLatent,
    T edgeRating,
    T stepSize) 
{
  T error = innerProduct(movieLatent, movieLatent + LATENT_VECTOR_SIZE, userLatent, -edgeRating);

  // Take gradient step
  for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
      T prevMovie = movieLatent[i];
      T prevUser = userLatent[i];
      movieLatent[i] -= stepSize * (error * prevUser  + lambda * prevMovie);
      userLatent[i]  -= stepSize * (error * prevMovie + lambda * prevUser);
  }

  return error;
}

size_t userIdToUserNode(size_t userId) {
  return userId + NUM_MOVIE_NODES;
}

struct StepFunction {
  virtual LatentValue stepSize(int round) const = 0;
  virtual std::string name() const = 0;
  virtual bool isBold() const { return false; }
};

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

struct InvStepFunction: public StepFunction {
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

template<typename Graph, bool UseLocks>
class TestFixed2DGraphTiledExecutor {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::iterator iterator;
  typedef typename Graph::edge_iterator edge_iterator;

  template<typename T>
  struct SimpleAtomic {
    std::atomic<T> value;
    SimpleAtomic(): value(0) { }
    SimpleAtomic(const SimpleAtomic& o): value(o.value.load()) { }
    T relaxedLoad() { return value.load(std::memory_order_relaxed); }
    void relaxedAdd(T delta) { value.store(relaxedLoad() + delta, std::memory_order_relaxed); }
  };

  /**
   * Tasks are 2D ranges [start1, end1) x [start2, end2]
   */
  struct Task {
    iterator start1;
    GNode start2;
    iterator end1;
    GNode end2;
    size_t id;
    size_t d1;
    size_t d2;
    SimpleAtomic<size_t> updates;
  };

  Graph& g;
  std::vector<SpinLock> locks1;
  std::vector<SpinLock> locks2;
  std::vector<Task> tasks;
  size_t maxUpdates;
  Galois::Statistic failures;

  struct GetDst: public std::unary_function<edge_iterator, GNode> {
    Graph* g;
    GetDst() { }
    GetDst(Graph* _g): g(_g) { }
    GNode operator()(edge_iterator ii) const {
      return g->getEdgeDst(ii);
    }
  };

  typedef Galois::NoDerefIterator<edge_iterator> no_deref_iterator;
  typedef boost::transform_iterator<GetDst, no_deref_iterator> edge_dst_iterator;

  template<typename Function>
  void executeBlock(Function& fn, Task& task) {
    GetDst getDst { &g };

    for (auto ii = task.start1; ii != task.end1; ++ii) {
      auto& src = g.getData(*ii);
      edge_iterator begin = g.edge_begin(*ii);
      no_deref_iterator nbegin(begin);
      no_deref_iterator nend(g.edge_end(*ii));
      edge_dst_iterator dbegin(nbegin, getDst);
      edge_dst_iterator dend(nend, getDst);
      if (std::distance(g.edge_begin(*ii), g.edge_end(*ii)) > 1024)
        continue;

      for (auto jj = std::lower_bound(dbegin, dend, task.start2); jj != dend; ++jj) {
        bool done = false;
        for (int times = 0; times < 5; ++times) {
          for (int i = 0; i < 5; ++i) {
            edge_iterator edge = *(jj+i).base();
            if (g.getEdgeDst(edge) > task.end2) {
              done = true;
              break;
            }

            auto& dst = g.getData(g.getEdgeDst(edge));
              
            fn(src, dst, g.getEdgeData(edge));
          }
        }
        if (done)
          break;
        for (int i = 0; jj != dend && i < 5; ++jj, ++i)
          ;
        if (jj == dend)
          break;
      }
    }
  }

  template<typename Function>
  void executeLoop(Function fn, unsigned tid, unsigned total) {
    const size_t numBlocks1 = locks1.size();
    const size_t numBlocks2 = locks2.size();
    const size_t numBlocks = numBlocks1 * numBlocks2;
    const size_t block1 = (numBlocks1 + total - 1) / total;
    const size_t start1 = std::min(block1 * tid, numBlocks1 - 1);
    const size_t block2 = (numBlocks2 + total - 1) / total;
    const size_t start2 = std::min(block2 * tid, numBlocks2 - 1);

    //size_t start = start1 + start2 * numBlocks1; // XXX
    //size_t start = block1 * 10 * (tid / 10) + start2 * numBlocks1;
    size_t start = start1 + block2 * 10 * (tid / 10) * numBlocks1;

    for (int i = 0; ; ++i) {
      start = nextBlock(start, numBlocks, i == 0);
      Task* t = &tasks[start];
      if (t == &tasks[numBlocks])
        break;
      executeBlock(fn, *t);

      locks1[t->d1].unlock();
      locks2[t->d2].unlock();
    }
  }

  size_t probeBlock(size_t start, size_t by, size_t n, size_t numBlocks) {
    for (size_t i = 0; i < n; ++i, start += by) {
      while (start >= numBlocks)
        start -= numBlocks;
      Task& b = tasks[start];
      if (b.updates.relaxedLoad() < maxUpdates) {
        if (UseLocks) {
          if (std::try_lock(locks1[b.d1], locks2[b.d2]) < 0) {
            // Return while holding locks
            b.updates.relaxedAdd(1);
            return start;
          }
        } else {
          if (b.updates.value.fetch_add(1) < maxUpdates)
            return start;
        }
      }
    }
    return numBlocks;
    failures += 1;
  }

  // Nested dim1 then dim2
  size_t nextBlock(size_t origStart, size_t numBlocks, bool origInclusive) {
    const size_t delta2 = locks1.size();
    const size_t delta1 = 1;
    size_t b;

    for (int times = 0; times < 2; ++times) {
      size_t limit2 = locks2.size();
      size_t limit1 = locks1.size();
      size_t start = origStart;
      bool inclusive = origInclusive && times == 0;
      // First iteration is exclusive of start
      if ((b = probeBlock(start + (inclusive ? 0 : delta1), delta1, limit1 - (inclusive ? 0 : 1), numBlocks)) != numBlocks)
        return b;
      if ((b = probeBlock(start + (inclusive ? 0 : delta2), delta2, limit2 - (inclusive ? 0 : 1), numBlocks)) != numBlocks)
        return b;
      start += delta1 + delta2;
      while (limit1 > 0 || limit2 > 0) {
        while (start >= numBlocks)
          start -= numBlocks;
        // Subsequent iterations are inclusive of start
        if (limit1 > 0 && (b = probeBlock(start, delta1, limit1 - 1, numBlocks)) != numBlocks)
          return b;
        if (limit2 > 0 && (b = probeBlock(start, delta2, limit2 - 1, numBlocks)) != numBlocks)
          return b;
        if (limit1 > 0) {
          limit1--;
          start += delta1;
        }
        if (limit2 > 0) {
          limit2--;
          start += delta2;
        }
      }
    }

    return numBlocks;
  }

  void initializeTasks(iterator first1, iterator last1, iterator first2, iterator last2, size_t size1, size_t size2) {
    const size_t numBlocks1 = (std::distance(first1, last1) + size1 - 1) / size1;
    const size_t numBlocks2 = (std::distance(first2, last2) + size2 - 1) / size2;
    const size_t numBlocks = numBlocks1 * numBlocks2;

    locks1.resize(numBlocks1);
    locks2.resize(numBlocks2);
    tasks.resize(numBlocks);

    GetDst fn { &g };

    for (size_t i = 0; i < numBlocks; ++i) {
      Task& task = tasks[i];
      task.d1 = i % numBlocks1;
      task.d2 = i / numBlocks1;
      task.id = i;
      std::tie(task.start1, task.end1) = Galois::block_range(first1, last1, task.d1, numBlocks1);
      // XXX: Works for CSR graphs
      task.start2 = task.d2 * size2 + *first2;
      task.end2 = (task.d2 + 1) * size2 + *first2 - 1;
    }
  }

  template<typename Function>
  struct Process {
    TestFixed2DGraphTiledExecutor* self;
    Function fn;

    void operator()(unsigned tid, unsigned total) {
      self->executeLoop(fn, tid, total);
    }
  };

public:
  TestFixed2DGraphTiledExecutor(Graph& _g): g(_g), failures("PopFailures") { }

  template<typename Function>
  void execute(iterator first1, iterator last1, iterator first2, iterator last2, size_t size1, size_t size2, Function fn, size_t numIterations = 1) {
    initializeTasks(first1, last1, first2, last2, size1, size2);
    maxUpdates = numIterations;
    Process<Function> p = { this, fn };
    Galois::on_each(p);
  }
};

template<typename Graph, bool UseLocks>
class Fixed2DGraphTiledExecutor {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::iterator iterator;
  typedef typename Graph::edge_iterator edge_iterator;

  template<typename T>
  struct SimpleAtomic {
    std::atomic<T> value;
    SimpleAtomic(): value(0) { }
    SimpleAtomic(const SimpleAtomic& o): value(o.value.load()) { }
    T relaxedLoad() { return value.load(std::memory_order_relaxed); }
    void relaxedAdd(T delta) { value.store(relaxedLoad() + delta, std::memory_order_relaxed); }
  };

  /**
   * Tasks are 2D ranges [start1, end1) x [start2, end2]
   */
  struct Task {
    iterator start1;
    GNode start2;
    iterator end1;
    GNode end2;
    size_t id;
    size_t d1;
    size_t d2;
    SimpleAtomic<size_t> updates;
  };

  Graph& g;
  std::vector<SpinLock> locks1;
  std::vector<SpinLock> locks2;
  std::vector<Task> tasks;
  size_t maxUpdates;
  Galois::Statistic failures;

  struct GetDst: public std::unary_function<edge_iterator, GNode> {
    Graph* g;
    GetDst() { }
    GetDst(Graph* _g): g(_g) { }
    GNode operator()(edge_iterator ii) const {
      return g->getEdgeDst(ii);
    }
  };

  typedef Galois::NoDerefIterator<edge_iterator> no_deref_iterator;
  typedef boost::transform_iterator<GetDst, no_deref_iterator> edge_dst_iterator;

  template<typename Function>
  void executeBlock(Function& fn, Task& task) {
    GetDst getDst { &g };

    for (auto ii = task.start1; ii != task.end1; ++ii) {
      auto& src = g.getData(*ii);
      edge_iterator begin = g.edge_begin(*ii);
      no_deref_iterator nbegin(begin);
      no_deref_iterator nend(g.edge_end(*ii));
      edge_dst_iterator dbegin(nbegin, getDst);
      edge_dst_iterator dend(nend, getDst);

      for (auto jj = std::lower_bound(dbegin, dend, task.start2); jj != dend; ++jj) {
        edge_iterator edge = *jj.base();
        if (g.getEdgeDst(edge) > task.end2)
          break;

        auto& dst = g.getData(g.getEdgeDst(edge));
          
        fn(src, dst, g.getEdgeData(edge));
      }
    }
  }

  template<typename Function>
  void executeLoop(Function fn, unsigned tid, unsigned total) {
    const size_t numBlocks1 = locks1.size();
    const size_t numBlocks2 = locks2.size();
    const size_t numBlocks = numBlocks1 * numBlocks2;
    const size_t block1 = (numBlocks1 + total - 1) / total;
    const size_t start1 = std::min(block1 * tid, numBlocks1 - 1);
    const size_t block2 = (numBlocks2 + total - 1) / total;
    const size_t start2 = std::min(block2 * tid, numBlocks2 - 1);

    // TODO get actual cores per package
    size_t start;
    if (UseLocks)
      //start = block1 * 10 * (tid / 10) + start2 * numBlocks1;
      start = start1 + block2 * 10 * (tid / 10) * numBlocks1;
    else
      start = start1 + start2 * numBlocks1;

    for (int i = 0; ; ++i) {
      start = nextBlock(start, numBlocks, i == 0);
      Task* t = &tasks[start];
      if (t == &tasks[numBlocks])
        break;
      executeBlock(fn, *t);

      locks1[t->d1].unlock();
      locks2[t->d2].unlock();
    }
  }

  size_t probeBlock(size_t start, size_t by, size_t n, size_t numBlocks) {
    for (size_t i = 0; i < n; ++i, start += by) {
      while (start >= numBlocks)
        start -= numBlocks;
      Task& b = tasks[start];
      if (b.updates.relaxedLoad() < maxUpdates) {
        if (UseLocks) {
          if (std::try_lock(locks1[b.d1], locks2[b.d2]) < 0) {
            // Return while holding locks
            b.updates.relaxedAdd(1);
            return start;
          }
        } else {
          if (b.updates.value.fetch_add(1) < maxUpdates)
            return start;
        }
      }
    }
    return numBlocks;
    failures += 1;
  }

  // Nested dim1 then dim2
  size_t nextBlock(size_t origStart, size_t numBlocks, bool origInclusive) {
    const size_t delta2 = locks1.size();
    const size_t delta1 = 1;
    size_t b;

    for (int times = 0; times < 2; ++times) {
      size_t limit2 = locks2.size();
      size_t limit1 = locks1.size();
      size_t start = origStart;
      bool inclusive = origInclusive && times == 0;
      // First iteration is exclusive of start
      if ((b = probeBlock(start + (inclusive ? 0 : delta1), delta1, limit1 - (inclusive ? 0 : 1), numBlocks)) != numBlocks)
        return b;
      if ((b = probeBlock(start + (inclusive ? 0 : delta2), delta2, limit2 - (inclusive ? 0 : 1), numBlocks)) != numBlocks)
        return b;
      start += delta1 + delta2;
      while (limit1 > 0 || limit2 > 0) {
        while (start >= numBlocks)
          start -= numBlocks;
        // Subsequent iterations are inclusive of start
        if (limit1 > 0 && (b = probeBlock(start, delta1, limit1 - 1, numBlocks)) != numBlocks)
          return b;
        if (limit2 > 0 && (b = probeBlock(start, delta2, limit2 - 1, numBlocks)) != numBlocks)
          return b;
        if (limit1 > 0) {
          limit1--;
          start += delta1;
        }
        if (limit2 > 0) {
          limit2--;
          start += delta2;
        }
      }
    }

    return numBlocks;
  }

  void initializeTasks(iterator first1, iterator last1, iterator first2, iterator last2, size_t size1, size_t size2) {
    const size_t numBlocks1 = (std::distance(first1, last1) + size1 - 1) / size1;
    const size_t numBlocks2 = (std::distance(first2, last2) + size2 - 1) / size2;
    const size_t numBlocks = numBlocks1 * numBlocks2;

    locks1.resize(numBlocks1);
    locks2.resize(numBlocks2);
    tasks.resize(numBlocks);

    GetDst fn { &g };

    for (size_t i = 0; i < numBlocks; ++i) {
      Task& task = tasks[i];
      task.d1 = i % numBlocks1;
      task.d2 = i / numBlocks1;
      task.id = i;
      std::tie(task.start1, task.end1) = Galois::block_range(first1, last1, task.d1, numBlocks1);
      // XXX: Works for CSR graphs
      task.start2 = task.d2 * size2 + *first2;
      task.end2 = (task.d2 + 1) * size2 + *first2 - 1;
    }
  }

  template<typename Function>
  struct Process {
    Fixed2DGraphTiledExecutor* self;
    Function fn;

    void operator()(unsigned tid, unsigned total) {
      self->executeLoop(fn, tid, total);
    }
  };

public:
  Fixed2DGraphTiledExecutor(Graph& _g): g(_g), failures("PopFailures") { }

  template<typename Function>
  void execute(iterator first1, iterator last1, iterator first2, iterator last2, size_t size1, size_t size2, Function fn, size_t numIterations = 1) {
    initializeTasks(first1, last1, first2, last2, size1, size2);
    maxUpdates = numIterations;
    Process<Function> p = { this, fn };
    Galois::on_each(p);
  }
};

template<typename Graph>
double sumSquaredError(Graph& g) {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_type NodeData;
  // computing Root Mean Square Error
  // Assuming only movie nodes have edges
  Galois::GAccumulator<double> error;

  // Save for performance testing
#if 0
  Galois::do_all(g.begin(), g.begin() + NUM_MOVIE_NODES, [&](GNode n) {
    for (auto ii = g.edge_begin(n), ei = g.edge_end(n); ii != ei; ++ii) {
      GNode dst = g.getEdgeDst(ii);
      LatentValue e = predictionError(g.getData(n).latentVector, g.getData(dst).latentVector, -static_cast<LatentValue>(g.getEdgeData(ii)));

      error += (e * e);
    }
  });
#else
  Fixed2DGraphTiledExecutor<Graph,false> executor(g);
  executor.execute(g.begin(), g.begin() + NUM_MOVIE_NODES, g.begin() + NUM_MOVIE_NODES, g.end(),
      moviesPerBlock, usersPerBlock, [&](NodeData& nn, NodeData& mm, unsigned int edgeData) {
    LatentValue e = predictionError(nn.latentVector, mm.latentVector, -static_cast<LatentValue>(edgeData));

    error += (e * e);
  });
#endif
  return error.reduce();
}

template<typename Graph>
size_t countEdges(Graph& g) {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_type NodeData;
  Galois::GAccumulator<size_t> edges;
  Fixed2DGraphTiledExecutor<Graph,false> executor(g);
  executor.execute(g.begin(), g.begin() + NUM_MOVIE_NODES, g.begin() + NUM_MOVIE_NODES, g.end(),
      moviesPerBlock, usersPerBlock, [&](NodeData& nn, NodeData& mm, unsigned int edgeData) {
    edges += 1;
  });
  return edges.reduce();
}

template<typename Graph>
void verify(Graph& g) {
  if (countEdges(g) != NUM_RATINGS) {
    GALOIS_DIE("Error: edge list of input graph probably not sorted");
  }

  double error = sumSquaredError(g);
  double rmse = std::sqrt(error/NUM_RATINGS);

  std::cout << "Final RMSE: " << rmse << "\n";
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
  Galois::GAccumulator<double> errorAccum;
  std::vector<LatentValue> steps(updatesPerEdge);
  LatentValue last = -1.0;
  int deltaRound = updatesPerEdge;
  LatentValue rate = learningRate;
  Galois::TimeAccumulator elapsed;

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

    double gflops = countFlops(NUM_RATINGS, deltaRound, LATENT_VECTOR_SIZE) / millis / 1e6;

    int curRound = round + deltaRound;
    std::cout
      << "R: " << curRound
      << " elapsed (ms): " << curElapsed
      << " GFLOP/s: " << gflops;
    if (useExactError) {
      std::cout << " RMSE (R " << curRound << "): " << std::sqrt(error/NUM_RATINGS);
    } else {
      std::cout << " Approx. RMSE (R " << (curRound - 1) << ".5): " << std::sqrt(error/NUM_RATINGS);
    }
    std::cout << "\n";
    if (!isFinite(error))
      break;
    if (fixedRounds <= 0 && last >= 0.0 && std::abs((last - error) / last) < 0.01)
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

//! Simple edge-wise operator
struct EdgeAlgo {
  static const bool makeSerializable = false;

  std::string name() const { return "EdgeAlgo"; }

  struct Node {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  typedef typename Galois::Graph::LC_CSR_Graph<Node, unsigned int>
    ::with_numa_alloc<true>::type
//    ::with_compressed_node_ptr<true>::type
    ::with_out_of_line_lockable<true>::type
    ::with_no_lockable<!makeSerializable>::type Graph;
  typedef Graph::GraphNode GNode;

  Graph& g;

  /**
   * Tasks are 2D ranges [start1, end1) x [start2, end2]
   */
  struct Task {
    Graph::iterator start1;
    GNode start2;
    Graph::iterator end1;
    GNode end2;
    size_t id;
    size_t x;
    size_t y;
    double error;
    int updates;
  };

  struct GetDst: public std::unary_function<Graph::edge_iterator, GNode> {
    Graph* g;
    GetDst() { }
    GetDst(Graph* _g): g(_g) { }
    GNode operator()(Graph::edge_iterator ii) const {
      return g->getEdgeDst(ii);
    }
  };

  typedef Galois::NoDerefIterator<Graph::edge_iterator> no_deref_iterator;
  typedef boost::transform_iterator<GetDst, no_deref_iterator> edge_dst_iterator;

  struct Process {
    Graph& g;
    Galois::Statistic& edgesVisited;
    Galois::Statistic& failures;
    std::vector<SpinLock>& xLocks;
    std::vector<SpinLock>& yLocks;
    std::vector<Task>& tasks;
    LatentValue* steps;
    int maxUpdates;
    Galois::GAccumulator<double>* errorAccum;

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
        Galois::MethodFlag flag = phase == 0 ? Galois::ALL : Galois::NONE;
        int numWorking;
        int round = 0;
        int limit = 0;
        do {
          numWorking = 0;
          int index = 0;
          for (auto ii = task.start1; ii != task.end1; ++ii, ++index) {
            Node& nn = g.getData(*ii, round == 0 ? flag : Galois::NONE);
            Graph::edge_iterator begin = g.edge_begin(*ii, Galois::NONE);
            no_deref_iterator nbegin(round == 0 ? no_deref_iterator(begin) : starts[index]);
            no_deref_iterator nend(no_deref_iterator(g.edge_end(*ii, Galois::NONE)));
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
      const int innerCount = std::numeric_limits<int>::max(); // XXX
      const LatentValue stepSize = steps[updatesPerEdge - maxUpdates + task.updates];
      GetDst fn { &g };
      double error = 0.0;

      // TODO modify edge data to support agnostic edge blocking
      for (int phase = makeSerializable ? 0 : 1; phase < 2; ++phase) {
        Galois::MethodFlag flag = phase == 0 ? Galois::ALL : Galois::NONE;
        for (auto ii = task.start1; ii != task.end1; ++ii) {
          Node& nn = g.getData(*ii, round == 0 ? flag : Galois::NONE);
          Graph::edge_iterator begin = g.edge_begin(*ii, Galois::NONE);
          no_deref_iterator nbegin(begin);
          no_deref_iterator nend(g.edge_end(*ii, Galois::NONE));
          edge_dst_iterator dbegin(nbegin, fn);
          edge_dst_iterator dend(nend, fn);
          for (auto jj = std::lower_bound(dbegin, dend, task.start2); jj != dend; ++jj) {
            Graph::edge_iterator edge = *jj.base();
            if (g.getEdgeDst(edge) > task.end2)
              break;
            Node& mm = g.getData(g.getEdgeDst(edge), flag);
            if (phase == 1) {
              LatentValue e = doGradientUpdate(nn.latentVector, mm.latentVector, static_cast<LatentValue>(g.getEdgeData(edge)), stepSize);
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

    void operator()(Task& task, Galois::UserContext<Task>& ctx) {
#if 1
      if (std::try_lock(xLocks[task.x], yLocks[task.y]) >= 0) {
        //Galois::Runtime::forceAbort();
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
        updateBlock(*t);

        xLocks[t->x].unlock();
        yLocks[t->y].unlock();
      }
    }
  };

  struct Inspect {
    Graph& g;
    Galois::InsertBag<Task>& initial;
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
     * traversals, we read a small square of M^1/4 movies by M^1/4 users/out
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

      const size_t numYBlocks = (NUM_MOVIE_NODES + moviesPerBlock - 1) / moviesPerBlock;
      const size_t numXBlocks = (NUM_USER_NODES + usersPerBlock - 1) / usersPerBlock;
      const size_t numBlocks = numXBlocks * numYBlocks;

      std::cout
        << "moviesPerBlock: " << moviesPerBlock
        << " usersPerBlock: " << usersPerBlock
        << " numBlocks: " << numBlocks
        << " numXBlocks: " << numXBlocks
        << " numYBlocks: " << numYBlocks << "\n";

      xLocks.resize(numXBlocks);
      yLocks.resize(numYBlocks);
      tasks.resize(numBlocks);

      GetDst fn { &g };
      int maxNnz = std::numeric_limits<int>::min();
      int minNnz = std::numeric_limits<int>::max();
      int maxFloats = std::numeric_limits<int>::min();
      int minFloats = std::numeric_limits<int>::max();

      for (size_t i = 0; i < numBlocks; ++i) {
        Task& task = tasks[i];
        task.x = i % numXBlocks;
        task.y = i / numXBlocks;
        task.id = i;
        task.updates = 0;
        task.error = 0.0;
        task.start1 = g.begin();
        std::tie(task.start1, task.end1) = Galois::block_range(g.begin(), g.begin() + NUM_MOVIE_NODES, task.y, numYBlocks);
        task.start2 = task.x * usersPerBlock + NUM_MOVIE_NODES;
        task.end2 = (task.x + 1) * usersPerBlock + NUM_MOVIE_NODES - 1;

        if (false) {
          int nnz = 0;
          std::set<uint64_t> uniqueMovies;
          for (auto ii = task.start1; ii != task.end1; ++ii) {
            edge_dst_iterator start(no_deref_iterator(g.edge_begin(*ii)), fn);
            edge_dst_iterator end(no_deref_iterator(g.edge_end(*ii)), fn);
            for (auto jj = std::lower_bound(start, end, task.start2); jj != end; ++jj) {
              if (g.getEdgeDst(*jj.base()) > task.end2)
                break;
              nnz += 1;
              uniqueMovies.insert(g.getEdgeDst(*jj.base()) - task.start2);
            }
          }
          int floats = nnz + 2* std::distance(task.start1, task.end1) * LATENT_VECTOR_SIZE + 2*uniqueMovies.size() * LATENT_VECTOR_SIZE;
          if (nnz > maxNnz)
            maxNnz = nnz;
          if (nnz < minNnz)
            minNnz = nnz;
          if (floats > maxFloats)
            maxFloats = floats;
          if (floats < minFloats)
            minFloats = floats;
        }

        initial.push(task);
      }

      if (false) {
        for (size_t i = 0; i < numYBlocks; ++i) {
          std::mt19937 gen;
          std::cout << (i * numXBlocks) << " " << (i+1) * numXBlocks << " " << numBlocks << "\n";
          std::shuffle(&tasks[i * numXBlocks], &tasks[(i+1) * numXBlocks], gen);
        }
      }


      if (false) {
        std::cout 
          << "Max NNZ: " << maxNnz 
          << " Min NNZ: " << minNnz
          << " Max floats: " << maxFloats
          << " Min floats: " << minFloats
          << "\n";
      }
    }

    void adaptiveTiling(unsigned tid, unsigned total) {
      Galois::Statistic numTasks("Tasks");
      Galois::Statistic underTasks("UnderTasks");
      Galois::Statistic overTasks("OverTasks");

      size_t totalSize = usersPerBlock * (size_t) moviesPerBlock;
      size_t targetSize = static_cast<size_t>(std::max(std::sqrt(totalSize), 1.0));
      size_t sampleSize = static_cast<size_t>(std::max(std::sqrt(targetSize), 1.0));
      Graph::iterator cur, end;
      std::tie(cur, end) = Galois::block_range(g.begin(), g.begin() + NUM_MOVIE_NODES, tid, total);
      //std::tie(cur, end) = Galois::block_range(g.begin(), g.end(), tid, total);
      std::vector<Graph::edge_iterator> prevStarts;

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
          //nodeBlockSize = moviesPerBlock;
          //edgeBlockSize = usersPerBlock;

          task.end1 = Galois::safe_advance(task.start1, end, nodeBlockSize);
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

  void operator()(const StepFunction& sf) {
    //std::cout
    //  << "targetSize: " << (usersPerBlock * (size_t) moviesPerBlock)
    //  << "\n";

    Galois::StatTimer inspect("InspectTime");
    inspect.start();
    Galois::InsertBag<Task> initial;
    std::vector<SpinLock> xLocks;
    std::vector<SpinLock> yLocks;
    std::vector<Task> tasks;
    Inspect fn1 { g, initial, xLocks, yLocks, tasks };
    Galois::on_each(fn1);
    inspect.stop();

    Galois::Statistic edgesVisited("EdgesVisited");
    Galois::Statistic failures("PopFailures");
    Galois::StatTimer execute("ExecuteTime");
    execute.start();
    executeUntilConverged(sf, g, [&](LatentValue* steps, int maxUpdates, Galois::GAccumulator<double>* errorAccum) {
      Process fn2 { g, edgesVisited, failures, xLocks, yLocks, tasks, steps, maxUpdates, errorAccum };
      // Testing sufficient optimizations by moving towards BlockJump
      //Galois::for_each(initial.begin(), initial.end(), fn2, Galois::wl<Galois::WorkList::dChunkedFIFO<1>>());
      //Galois::for_each_local(initial, fn2, Galois::wl<Galois::WorkList::dChunkedFIFO<1>>());
      //Galois::do_all_local(initial, fn2, Galois::wl<Galois::WorkList::dChunkedLIFO<1>>());
      Galois::on_each(fn2);
    });
    execute.stop();
  }
};

struct BlockJumpAlgo {
  static const bool precomputeOffsets = false;

  std::string name() const { return "BlockAlgo"; }

  struct Node {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  typedef Galois::Graph::LC_CSR_Graph<Node, unsigned int>
    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  Graph& g;

  struct BlockInfo {
    size_t id;
    size_t x;
    size_t y;
    size_t userStart;
    size_t userEnd;
    size_t movieStart;
    size_t movieEnd;
    size_t numMovies;
    size_t updates;
    double error;
    int* userOffsets;

    std::ostream& print(std::ostream& os) {
      os 
        << "id: " << id
        << " x: " << x
        << " y: " << y
        << " userStart: " << userStart
        << " userEnd: " << userEnd
        << " movieStart: " << movieStart
        << " movieEnd: " << movieEnd
        << " updates: " << updates
        << "\n";
      return os;
    }
  };

  struct Process {
    Graph& g;
    SpinLock *xLocks, *yLocks;
    BlockInfo* blocks;
    size_t numXBlocks, numYBlocks;
    LatentValue* steps;
    int maxUpdates;
    Galois::GAccumulator<double>* errorAccum;
    
    struct GetDst: public std::unary_function<Graph::edge_iterator, GNode> {
      Graph* g;
      GetDst() { }
      GetDst(Graph* _g): g(_g) { }
      GNode operator()(Graph::edge_iterator ii) const {
        return g->getEdgeDst(ii);
      }
    };

    /**
     * Preconditions: row and column of slice are locked.
     *
     * Postconditions: increments update count, does sgd update on each movie
     * and user in the slice
     */
    template<bool Enable = precomputeOffsets>
    size_t runBlock(BlockInfo& si, typename std::enable_if<!Enable>::type* = 0) {
      typedef Galois::NoDerefIterator<Graph::edge_iterator> no_deref_iterator;
      typedef boost::transform_iterator<GetDst, no_deref_iterator> edge_dst_iterator;

      LatentValue stepSize = steps[si.updates - maxUpdates + updatesPerEdge];
      size_t seen = 0;
      double error = 0.0;

      // Set up movie iterators
      size_t movieId = 0;
      Graph::iterator mm = g.begin(), em = g.begin();
      std::advance(mm, si.movieStart);
      std::advance(em, si.movieEnd);
      
      GetDst fn { &g };

      // For each movie in the range
      for (; mm != em; ++mm, ++movieId) {  
        GNode movie = *mm;
        Node& movieData = g.getData(movie);
        size_t lastUser = si.userEnd + NUM_MOVIE_NODES;

        edge_dst_iterator start(no_deref_iterator(g.edge_begin(movie, Galois::NONE)), fn);
        edge_dst_iterator end(no_deref_iterator(g.edge_end(movie, Galois::NONE)), fn);

        // For each edge in the range
        for (auto ii = std::lower_bound(start, end, si.userStart + NUM_MOVIE_NODES); ii != end; ++ii) {
          GNode user = g.getEdgeDst(*ii.base());

          if (user >= lastUser)
            break;

          LatentValue e = doGradientUpdate(movieData.latentVector, g.getData(user).latentVector, static_cast<LatentValue>(g.getEdgeData(*ii.base())), stepSize);
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

    template<bool Enable = precomputeOffsets>
    size_t runBlock(BlockInfo& si, typename std::enable_if<Enable>::type* = 0) {
      LatentValue stepSize = steps[si.updates - maxUpdates + updatesPerEdge];
      size_t seen = 0;
      double error = 0.0;

      // Set up movie iterators
      size_t movieId = 0;
      Graph::iterator mm = g.begin(), em = g.begin();
      std::advance(mm, si.movieStart);
      std::advance(em, si.movieEnd);
      
      // For each movie in the range
      for (; mm != em; ++mm, ++movieId) {  
        if (si.userOffsets[movieId] < 0)
          continue;

        GNode movie = *mm;
        Node& movieData = g.getData(movie);
        size_t lastUser = si.userEnd + NUM_MOVIE_NODES;

        // For each edge in the range
        for (auto ii = g.edge_begin(movie) + si.userOffsets[movieId], ei = g.edge_end(movie); ii != ei; ++ii) {
          GNode user = g.getEdgeDst(ii);

          if (user >= lastUser)
            break;

          LatentValue e = doGradientUpdate(movieData.latentVector, g.getData(user).latentVector, static_cast<LatentValue>(g.getEdgeData(ii)), stepSize);
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
      size_t numBlocks = numXBlocks * numYBlocks;
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
      Galois::StatTimer timer("PerThreadTime");
      Galois::Statistic edgesVisited("EdgesVisited");
      Galois::Statistic blocksVisited("BlocksVisited");
      size_t numBlocks = numXBlocks * numYBlocks;
      size_t xBlock = (numXBlocks + total - 1) / total;
      size_t xStart = std::min(xBlock * tid, numXBlocks - 1);
      size_t yBlock = (numYBlocks + total - 1) / total;
      size_t yStart = std::min(yBlock * tid, numYBlocks - 1);
      BlockInfo* sp = &blocks[xStart + yStart + numXBlocks];

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

  void operator()(const StepFunction& sf) {
    const size_t numYBlocks = (NUM_MOVIE_NODES + moviesPerBlock - 1) / moviesPerBlock;
    const size_t numXBlocks = (NUM_USER_NODES + usersPerBlock - 1) / usersPerBlock;
    const size_t numBlocks = numXBlocks * numYBlocks;

    SpinLock* xLocks = new SpinLock[numXBlocks];
    SpinLock* yLocks = new SpinLock[numYBlocks];
    
    std::cout
      << "moviesPerBlock: " << moviesPerBlock
      << " usersPerBlock: " << usersPerBlock
      << " numBlocks: " << numBlocks
      << " numXBlocks: " << numXBlocks
      << " numYBlocks: " << numYBlocks << "\n";
    
    // Initialize
    BlockInfo* blocks = new BlockInfo[numBlocks];
    for (size_t i = 0; i < numBlocks; i++) {
      BlockInfo& si = blocks[i];
      si.id = i;
      si.x = i % numXBlocks;
      si.y = i / numXBlocks;
      si.updates = 0;
      si.error = 0.0;
      si.userStart = si.x * usersPerBlock;
      si.userEnd = std::min((si.x + 1) * usersPerBlock, NUM_USER_NODES);
      si.movieStart = si.y * moviesPerBlock;
      si.movieEnd = std::min((si.y + 1) * moviesPerBlock, NUM_MOVIE_NODES);
      si.numMovies = si.movieEnd - si.movieStart;
      if (precomputeOffsets) {
        si.userOffsets = new int[si.numMovies];
      } else {
        si.userOffsets = nullptr;
      }
    }

    // Partition movie edges in blocks to users according to range [userStart, userEnd)
    if (precomputeOffsets) {
      Galois::do_all(g.begin(), g.begin() + NUM_MOVIE_NODES, [&](GNode movie) {
        size_t sliceY = movie / moviesPerBlock;
        BlockInfo* s = &blocks[sliceY * numXBlocks];

        size_t pos = movie - s->movieStart;
        auto ii = g.edge_begin(movie), ei = g.edge_end(movie);
        size_t offset = 0;
        for (size_t i = 0; i < numXBlocks; ++i, ++s) {
          size_t start = userIdToUserNode(s->userStart);
          size_t end = userIdToUserNode(s->userEnd);

          if (ii != ei && g.getEdgeDst(ii) >= start && g.getEdgeDst(ii) < end) {
            s->userOffsets[pos] = offset;
          } else {
            s->userOffsets[pos] = -1;
          }
          for (; ii != ei && g.getEdgeDst(ii) < end; ++ii, ++offset)
            ;
        }
      });
    }
    
    executeUntilConverged(sf, g, [&](LatentValue* steps, int maxUpdates, Galois::GAccumulator<double>* errorAccum) {
      Process fn { g, xLocks, yLocks, blocks, numXBlocks, numYBlocks, steps, maxUpdates, errorAccum };
      Galois::on_each(fn);
    });
  }
};


//! Initializes latent vector with random values and returns basic graph parameters
template<typename Graph>
std::pair<size_t, size_t> initializeGraphData(Graph& g) {
  double top = 1.0/std::sqrt(LATENT_VECTOR_SIZE);

  std::mt19937 gen;
#if __cplusplus >= 201103L || defined(HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
  std::uniform_real_distribution<LatentValue> dist(0, top);
#else
  std::uniform_real<LatentValue> dist(0, top);
#endif
  //std::ofstream file("out.csv");

  // TODO: parallelize
  size_t numMovieNodes = 0;
  // for all movie and user nodes in the graph
  for (typename Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
    auto& data = g.getData(*ii);

    for (int i = 0; i < LATENT_VECTOR_SIZE; i++)
      data.latentVector[i] = dist(gen);

    // count number of movies we've seen; only movies nodes have edges
    if (std::distance(g.edge_begin(*ii), g.edge_end(*ii)))
      numMovieNodes++;
    //file << std::distance(g.edge_begin(*ii), g.edge_end(*ii)) << "\n";
  }

  return std::make_pair(numMovieNodes, g.size() - numMovieNodes);
}


StepFunction* newStepFunction(Step s) {
  switch (s) {
    case Intel: return new IntelStepFunction;
    case Purdue: return new PurdueStepFunction;
    case Bottou: return new BottouStepFunction;
    case Inv: return new InvStepFunction;
    case Bold: return new BoldStepFunction;
  }
  GALOIS_DIE("unknown step function");
}


template<typename Graph>
void testOut(Graph& g) {
  const size_t numYBlocks = (NUM_MOVIE_NODES + moviesPerBlock - 1) / moviesPerBlock;
  const size_t numXBlocks = (NUM_USER_NODES + usersPerBlock - 1) / usersPerBlock;
  const size_t numBlocks = numXBlocks * numYBlocks;

  std::cout
    << "moviesPerBlock: " << moviesPerBlock
    << " usersPerBlock: " << usersPerBlock
    << " numBlocks: " << numBlocks
    << " numXBlocks: " << numXBlocks
    << " numYBlocks: " << numYBlocks << "\n";

  Galois::Timer timer;
  timer.start();
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_type NodeData;
  // computing Root Mean Square Error
  // Assuming only movie nodes have edges
  Galois::GAccumulator<double> error;
  Galois::GAccumulator<size_t> visited;
  TestFixed2DGraphTiledExecutor<Graph,false> executor(g);
  //executor.execute(g.begin(), g.begin() + NUM_MOVIE_NODES, g.begin() + NUM_MOVIE_NODES, g.end(),
  executor.execute(g.begin() + NUM_MOVIE_NODES, g.end(), g.begin(), g.begin() + NUM_MOVIE_NODES,
      moviesPerBlock, usersPerBlock, [&](NodeData& nn, NodeData& mm, unsigned int edgeData) {
    LatentValue e = predictionError(nn.latentVector, mm.latentVector, -static_cast<LatentValue>(edgeData));

    error += (e * e);
    visited += 1;
  });
  timer.stop();
  std::cout 
    << "ERROR: " << error.reduce()
    << " Time: " << timer.get() 
    << " GFLOP/s: " << (visited.reduce() * (2.0 * LATENT_VECTOR_SIZE + 2)) / timer.get() / 1e6 << "\n";
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

  // Represent bipartite graph in general graph data structure:
  //  * movies are the first m nodes
  //  * users are the next n nodes
  //  * only movies have outedges
  Galois::Graph::readGraph(g, inputFilename);

  std::tie(NUM_MOVIE_NODES, NUM_USER_NODES) = initializeGraphData(g);
  NUM_RATINGS = g.sizeEdges();

  std::unique_ptr<StepFunction> sf { newStepFunction(learningRateFunction) };
  Algo algo { g };

  std::cout 
    << "num users: " << NUM_USER_NODES 
    << " num movies: " << NUM_MOVIE_NODES 
    << " num ratings: " << NUM_RATINGS
    << "\n";
  std::cout
    << "latent vector size: " << LATENT_VECTOR_SIZE
    << " lambda: " << lambda
    << " learning rate: " << learningRate
    << " decay rate: " << decayRate
    << " algo: " << algo.name()
    << " step function: " << sf->name()
    << "\n";
      
  if (true) {
    Galois::StatTimer timer;
    timer.start();
    algo(*sf);
    timer.stop();

    if (!skipVerify) {
      verify(g);
    }
  } else {
    testOut(g);
  }

  if (outputFilename != "") {
    std::cout << "Writing latent vectors to " << outputFilename << "\n";
    switch (outputType) {
      case OutputType::binary: writeBinaryLatentVectors(g, outputFilename); break;
      case OutputType::ascii: writeAsciiLatentVectors(g, outputFilename); break;
      default: abort();
    }
  }
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;

  switch (algo) {
    case Algo::edgeMovie: run<EdgeAlgo>(); break;
    case Algo::blockJump: run<BlockJumpAlgo>(); break;
    default: GALOIS_DIE("unknown algorithm"); break;
  }

  return 0;
}
