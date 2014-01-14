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
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>

#include "Galois/Accumulator.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"

#include "Galois/Graph/Graph.h"
#include "Galois/Graph/LCGraph.h"

#include "Galois/Runtime/ll/PaddedLock.h"

#include "Lonestar/BoilerPlate.h"

static const char* const name = "Stochastic Gradient Descent";
static const char* const desc = "Computes Matrix Decomposition using Stochastic Gradient Descent";
static const char* const url = "sgd";

static const int LATENT_VECTOR_SIZE = 20; //Prad's default: 100, Intel: 20
static const double MINVAL = -1e+100;
static const double MAXVAL = 1e+100;

static const double LEARNING_RATE = 0.001; // GAMMA, Purdue: 0.01 Intel: 0.001
static const double DECAY_RATE = 0.9; // STEP_DEC, Purdue: 0.1 Intel: 0.9
static const double LAMBDA = 0.001; // Purdue: 1.0 Intel: 0.001
static const double BottouInit = 0.1;

enum Algo {
  edgeMovie,
  blockJump
};

enum Learn {
  Intel,
  Purdue,
  Bottou,
  Inv
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> maxUpdatesPerEdge("maxUpdatesPerEdge", cll::desc("maximum number of updates per edge"), cll::init(5));
static cll::opt<int> usersPerBlock("usersPerBlock", cll::desc("users per block"), cll::init(2048));
static cll::opt<int> moviesPerBlock("moviesPerBlock", cll::desc("movies per block"), cll::init(350));
static cll::opt<bool> verifyPerIter("verifyPerIter", cll::desc("compute RMS every iter"), cll::init(false));

static cll::opt<Algo> algo(cll::desc("Choose an algorithm:"),
  cll::values(
    clEnumVal(edgeMovie, "Edgewise"),
    clEnumVal(blockJump, "Block jumping version"),
  clEnumValEnd), 
  cll::init(blockJump));

static cll::opt<Learn> learn(cll::desc("Choose a learning function:"),
  cll::values(
    clEnumVal(Intel, "Intel (default)"),
    clEnumVal(Purdue, "Perdue"),
    clEnumVal(Bottou, "Bottou"),
    clEnumVal(Inv, "Simple Inverse"),
  clEnumValEnd), 
  cll::init(Intel));

typedef Galois::Runtime::LL::PaddedLock<true> SpinLock;

size_t NUM_MOVIE_NODES = 0;
size_t NUM_USER_NODES = 0;
size_t NUM_RATINGS = 0;


double dotProduct(
    double* __restrict__ movieLatent,
    double* __restrict__ userLatent)
{
  double dp = 0.0;
  for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
    dp += userLatent[i] * movieLatent[i];
  assert(std::isnormal(dp));
  return dp;
}

double calcPrediction(
    double* __restrict__ movieLatent,
    double* __restrict__ userLatent)
{
  double pred = dotProduct(movieLatent, userLatent);
  return std::min(MAXVAL, std::max(MINVAL, pred));
}

void doGradientUpdate(
    double* __restrict__ movieLatent,
    double* __restrict__ userLatent,
    unsigned int edgeRating,
    double stepSize) 
{
  double error = edgeRating - dotProduct(movieLatent, userLatent);
  
  // Take gradient step
  for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
      double prevMovie = movieLatent[i];
      double prevUser = userLatent[i];
      movieLatent[i] += stepSize * (error * prevUser  - LAMBDA * prevMovie);
      userLatent[i]  += stepSize * (error * prevMovie - LAMBDA * prevUser);
  }
}

size_t userIdToUserNode(size_t userId) {
  return userId + NUM_MOVIE_NODES;
}

struct LearningFunction {
  virtual double stepSize(int round) const = 0;
};

struct PurdueLearningFunction: public LearningFunction {
  virtual double stepSize(int round) const {
    return LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(round + 1, 1.5));
  }
};

struct IntelLearningFunction: public LearningFunction {
  virtual double stepSize(int round) const {
    return LEARNING_RATE * pow(DECAY_RATE, round);
  }
};

struct BottouLearningFunction: public LearningFunction {
  virtual double stepSize(int round) const {
    return BottouInit / (1.0 + BottouInit*LAMBDA*round);
  }
};

struct InvLearningFunction: public LearningFunction {
  virtual double stepSize(int round) const {
    return 1.0 / (round + 1);
  }
};

template<typename Graph>
void verify(Graph& g) {
  typedef typename Graph::GraphNode GNode;
  // computing Root Mean Square Error
  // Assuming only movie nodes have edges
  Galois::GAccumulator<double> rms;

  Galois::do_all_local(g, [&](GNode n) {
    for (auto ii = g.edge_begin(n), ei = g.edge_end(n); ii != ei; ++ii) {
      GNode dst = g.getEdgeDst(ii);
      double pred = calcPrediction(g.getData(n).latentVector, g.getData(dst).latentVector);
      double rating = g.getEdgeData(ii);
    
      if (!std::isnormal(pred))
        std::cout << "denormal warning\n";

      rms += ((pred - rating) * (pred - rating));
    }
  });

  double totalRms = rms.reduce();
  double normalizedRms = sqrt(totalRms/NUM_RATINGS);
  
  std::cout << "RMSE Total: " << totalRms << " Normalized: " << normalizedRms << "\n";
}

//! Simple edge-wise operator
struct EdgeAlgo {
  static const bool makeSerializable = false;

  struct Node {
    double latentVector[LATENT_VECTOR_SIZE];
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
    double stepSize;
    Galois::Statistic& edgesVisited;
    std::vector<SpinLock>& xLocks;
    std::vector<SpinLock>& yLocks;
    std::vector<Task>& tasks;

    void updateBlock(Task& task) {
      GetDst fn { &g };

      for (int phase = makeSerializable ? 0 : 1; phase < 2; ++phase) {
        Galois::MethodFlag flag = phase == 0 ? Galois::ALL : Galois::NONE;
        for (auto ii = task.start1; ii != task.end1; ++ii) {
          Node& nn = g.getData(*ii, flag);
          edge_dst_iterator start(no_deref_iterator(g.edge_begin(*ii, Galois::NONE)), fn);
          edge_dst_iterator end(no_deref_iterator(g.edge_end(*ii, Galois::NONE)), fn);

          for (auto jj = std::lower_bound(start, end, task.start2); jj != end; ++jj) {
            Graph::edge_iterator edge = *jj.base();
            if (g.getEdgeDst(edge) > task.end2)
              break;
            Node& mm = g.getData(g.getEdgeDst(edge), flag);
            if (phase == 1) {
              doGradientUpdate(nn.latentVector, mm.latentVector, g.getEdgeData(edge), stepSize);
              edgesVisited += 1;
            }
          }
        }
      }
      task.updates += 1;
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
      if (task.updates < maxUpdatesPerEdge)
        ctx.push(task);

#if 1
      xLocks[task.x].unlock();
      yLocks[task.y].unlock();
#endif
    }

    size_t getNextBlock(Task* t, size_t numBlocks) {
      size_t nextBlockId;
      
      if (false) {
      // X direction
      nextBlockId = t->id + 1;
      for (size_t i = 0; i < xLocks.size() - 1; ++i, nextBlockId += 1) {
        // Wrap around
        if (nextBlockId >= numBlocks)
          nextBlockId -= numBlocks;
        Task& nextBlock = tasks[nextBlockId];

        if (nextBlock.updates < maxUpdatesPerEdge) {
          if (std::try_lock(xLocks[nextBlock.x], yLocks[nextBlock.y]) < 0) {
            // Return while holding locks
            return nextBlockId;
          }
        }
      }
      }

      // Y direction
      nextBlockId = t->id + xLocks.size();
      for (size_t i = 0; i < yLocks.size() - 1; ++i, nextBlockId += xLocks.size()) {
        // Wrap around
        if (nextBlockId >= numBlocks)
          nextBlockId -= numBlocks;
        Task& nextBlock = tasks[nextBlockId];

        if (nextBlock.updates < maxUpdatesPerEdge) {
          if (std::try_lock(xLocks[nextBlock.x], yLocks[nextBlock.y]) < 0) {
            // Return while holding locks
            return nextBlockId;
          }
        }
      }

      nextBlockId = t->id + 1; //+ xLocks.size();
      for (size_t i = 0; i < 2 * numBlocks; ++i, ++nextBlockId) {
        // Wrap around
        if (nextBlockId >= numBlocks)
          nextBlockId -= numBlocks;

        Task& nextBlock = tasks[nextBlockId];

        if (nextBlock.updates < maxUpdatesPerEdge) {
          if (std::try_lock(xLocks[nextBlock.x], yLocks[nextBlock.y]) < 0) {
            // Return while holding locks
            return nextBlockId;
          }
        }
      }

      return numBlocks;
    }

    void operator()(unsigned tid, unsigned total) {
      const size_t numYBlocks = yLocks.size();
      const size_t numXBlocks = xLocks.size();
      const size_t numBlocks = numXBlocks * numYBlocks;
      const size_t xBlock = (numXBlocks + total - 1) / total;
      const size_t xStart = std::min(xBlock * tid, numXBlocks - 1);
      const size_t yBlock = (numYBlocks + total - 1) / total;
      const size_t yStart = std::min(yBlock * tid, numYBlocks - 1);
      Task* t = &tasks[xStart + yStart + numXBlocks];

      while (true) {
        t = &tasks[getNextBlock(t, numBlocks)];
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

      for (size_t i = 0; i < numBlocks; ++i) {
        Task& task = tasks[i];
        task.x = i % numXBlocks;
        task.y = i / numXBlocks;
        task.id = i;
        task.updates = 0;
        task.start1 = g.begin();
        std::tie(task.start1, task.end1) = Galois::block_range(g.begin(), g.begin() + NUM_MOVIE_NODES, task.y, numYBlocks);
        task.start2 = task.x * usersPerBlock + NUM_MOVIE_NODES;
        task.end2 = (task.x + 1) * usersPerBlock + NUM_MOVIE_NODES - 1;
        initial.push(task);
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

  void operator()(const LearningFunction& lf) {
    std::cout
      << "targetSize: " << (usersPerBlock * (size_t) moviesPerBlock)
      << "\n";

    Galois::Statistic edgesVisited("EdgesVisited");

    Galois::StatTimer inspect("InspectTime");
    inspect.start();
    Galois::InsertBag<Task> initial;
    std::vector<SpinLock> xLocks;
    std::vector<SpinLock> yLocks;
    std::vector<Task> tasks;
    Inspect fn1 { g, initial, xLocks, yLocks, tasks };
    Galois::on_each(fn1);
    inspect.stop();

    Galois::StatTimer execute("ExecuteTime");
    execute.start();
    // FIXME: No straightforward mapping to "rounds"
    double stepSize = lf.stepSize(0);
    Process fn2 { g, stepSize, edgesVisited, xLocks, yLocks, tasks };
    //Galois::for_each(initial.begin(), initial.end(), fn2, Galois::wl<Galois::WorkList::dChunkedFIFO<1>>());
    //Galois::for_each_local(initial, fn2, Galois::wl<Galois::WorkList::dChunkedFIFO<1>>());
    //Galois::do_all_local(initial, fn2, Galois::wl<Galois::WorkList::dChunkedLIFO<1>>());
    Galois::on_each(fn2);
    execute.stop();
  }
};

struct BlockJumpAlgo {
  static const bool precomputeOffsets = false;

  struct Node {
    double latentVector[LATENT_VECTOR_SIZE];
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
    double stepSize;
    
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

      si.updates++;
      size_t seen = 0;

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

          doGradientUpdate(movieData.latentVector, g.getData(user).latentVector, g.getEdgeData(*ii.base()), stepSize);
          ++seen;
        }
      }
      
      return seen;
    }

    template<bool Enable = precomputeOffsets>
    size_t runBlock(BlockInfo& si, typename std::enable_if<Enable>::type* = 0) {
      si.updates++;
      size_t seen = 0;

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

          doGradientUpdate(movieData.latentVector, g.getData(user).latentVector, g.getEdgeData(ii), stepSize);
          ++seen;
        }
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

        if (nextBlock.updates < maxUpdatesPerEdge && xLocks[nextBlock.x].try_lock()) {
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

  void operator()(const LearningFunction& lf) {
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
    
    // FIXME: No straightforward mapping to "rounds"
    double stepSize = lf.stepSize(0); // FIXME: was lf.stepSize(1)
    Process fn { g, xLocks, yLocks, blocks, numXBlocks, numYBlocks, stepSize };
    Galois::on_each(fn);
  }
};


//! Generates a random double in (-1,1)
static double genRand() {
  return 2.0 * (std::rand() / (double)RAND_MAX) - 1.0;
}


//! Initializes latent vector with random values and returns basic graph parameters
template<typename Graph>
std::pair<size_t, size_t> initializeGraphData(Graph& g) {
  // unsigned int seed = 42;
  // std::default_random_engine eng(seed);
  // std::uniform_real_distribution<double> random_lv_value(0, 0.1);
  const unsigned SEED = 4562727;
  std::srand(SEED);

  size_t numMovieNodes = 0;

  // for all movie and user nodes in the graph
  for (typename Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
    auto& data = g.getData(*ii);

    for (int i = 0; i < LATENT_VECTOR_SIZE; i++)
      data.latentVector[i] = genRand();

    // count number of movies we've seen; only movies nodes have edges
    if (std::distance(g.edge_begin(*ii), g.edge_end(*ii)))
      numMovieNodes++;
  }

  return std::make_pair(numMovieNodes, g.size() - numMovieNodes);
}


LearningFunction* newLearningFunction(Learn l) {
  switch (l) {
    case Intel: return new IntelLearningFunction;
    case Purdue: return new PurdueLearningFunction;
    case Bottou: return new BottouLearningFunction;
    case Inv: return new InvLearningFunction;
  }
  GALOIS_DIE("unknown learning function");
}


template<typename Algo>
void run() {
  typename Algo::Graph g;

  // Represent bipartite graph in general graph data structure:
  //  * movies are the first m nodes
  //  * users are the next n nodes
  //  * only movies have outedges
  Galois::Graph::readGraph(g, inputFile);

  std::tie(NUM_MOVIE_NODES, NUM_USER_NODES) = initializeGraphData(g);
  NUM_RATINGS = g.sizeEdges();

  std::cout 
    << "num users: " << NUM_USER_NODES 
    << " num movies: " << NUM_MOVIE_NODES 
    << " num ratings: " << NUM_RATINGS
    << "\n";
  
  std::unique_ptr<LearningFunction> lf { newLearningFunction(learn) };
  Algo algo { g };

  Galois::StatTimer timer;
  timer.start();
  algo(*lf);
  timer.stop();

  if (!skipVerify)
    verify(g);
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
