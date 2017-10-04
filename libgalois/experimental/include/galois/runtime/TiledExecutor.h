/**  -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 */

#ifndef GALOIS_RUNTIME_TILEDEXECUTOR_H
#define GALOIS_RUNTIME_TILEDEXECUTOR_H

#include "galois/Galois.h"
#include "galois/LargeArray.h"
#include "galois/NoDerefIterator.h"
#include "galois/gstl.h"
#include "galois/runtime/ll/PaddedLock.h"

#include <boost/iterator/transform_iterator.hpp>
#include <atomic>
#include <algorithm>
#include <array>
#include <vector>
#include <cstddef>
#include <functional>
#include <mutex>

namespace galois { namespace runtime {

template<typename Graph, bool UseExp = false>
class Fixed2DGraphTiledExecutor {
  static constexpr int numDims = 2;
  typedef galois::runtime::LL::PaddedLock<true> SpinLock;
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::iterator iterator;
  typedef typename Graph::edge_iterator edge_iterator;
  typedef std::array<size_t, numDims> Point;

  template<typename T>
  struct SimpleAtomic {
    std::atomic<T> value;
    SimpleAtomic(): value(0) { }
    SimpleAtomic(const SimpleAtomic& o): value(o.value.load()) { }
    T relaxedLoad() { return value.load(std::memory_order_relaxed); }
    void relaxedAdd(T delta) { value.store(relaxedLoad() + delta, std::memory_order_relaxed); }
  };

  /**
   * Tasks are 2D ranges [start0, end0) x [start1, end1]
   */
  struct Task {
    iterator start0;
    iterator end0;
    GNode start1;
    GNode end1;
    Point coord;
    SimpleAtomic<int> updates;
  };

  struct GetDst: public std::unary_function<edge_iterator, GNode> {
    Graph* g;
    GetDst() { }
    GetDst(Graph* _g): g(_g) { }
    GNode operator()(edge_iterator ii) const {
      return g->getEdgeDst(ii);
    }
  };

  typedef galois::NoDerefIterator<edge_iterator> no_deref_iterator;
  typedef boost::transform_iterator<GetDst, no_deref_iterator> edge_dst_iterator;

  Graph& g;
  int cutoff; // XXX: UseExp
  galois::runtime::Barrier& barrier; // XXX: UseExp
  //std::array<galois::LargeArray<SpinLock>, numDims> locks;
  //galois::LargeArray<Task> tasks;
  galois::Statistic failures;
  std::array<std::vector<SpinLock>, numDims> locks;
  std::vector<Task> tasks;
  size_t numTasks;
  int maxUpdates;
  bool useLocks;

  void nextPoint(Point& p, int dim, int delta) {
    if (delta == 0)
      return;

    p[dim] += delta;

    while (p[dim] >= locks[dim].size())
      p[dim] -= locks[dim].size();
  }

  Task* getTask(const Point& p) {
    Task *t = &tasks[p[0] + p[1] * locks[0].size()];
    assert(t < &tasks[numTasks]);
    assert(t >= &tasks[0]);
    return t;
  }

  template<typename Function>
  void executeSparseBlock(Function& fn, Task& task) {
    GetDst getDst { &g };

    for (auto ii = task.start0; ii != task.end0; ++ii) {
      edge_iterator begin = g.edge_begin(*ii, galois::MethodFlag::UNPROTECTED);
      no_deref_iterator nbegin(begin);
      no_deref_iterator nend(g.edge_end(*ii, galois::MethodFlag::UNPROTECTED));
      edge_dst_iterator dbegin(nbegin, getDst);
      edge_dst_iterator dend(nend, getDst);

      if (UseExp && cutoff < 0 && std::distance(g.edge_begin(*ii, galois::MethodFlag::UNPROTECTED), g.edge_end(*ii, galois::MethodFlag::UNPROTECTED)) >= -cutoff)
        continue;
      else if (UseExp && cutoff > 0 && std::distance(g.edge_begin(*ii, galois::MethodFlag::UNPROTECTED), g.edge_end(*ii, galois::MethodFlag::UNPROTECTED)) < cutoff)
        continue;

      for (auto jj = std::lower_bound(dbegin, dend, task.start1); jj != dend; ) {
        if (UseExp) {
          constexpr int numTimes = 1;
          constexpr int width = 1;
          bool done = false;
          for (int times = 0; times < numTimes; ++times) {
            for (int i = 0; i < width; ++i) {
              edge_iterator edge = *(jj+i).base();
              if (*(jj + i) > task.end1) {
                done = true;
                break;
              }

              fn(*ii, *(jj+i), edge);
            }
          }
          if (done)
            break;
          for (int i = 0; jj != dend && i < width; ++jj, ++i)
            ;
          if (jj == dend)
            break;
        } else {
          edge_iterator edge = *jj.base();
          if (*jj > task.end1)
            break;

          fn(*ii, *jj, edge);
          ++jj;
        }
      }
    }
  }

  template<typename Function>
  void executeDenseBlock(Function& fn, Task& task) {
    GetDst getDst { &g };

    for (auto ii = task.start0; ii != task.end0; ++ii) {
      for (auto jj = g.begin() + task.start1, ej = g.begin() + task.end1 + 1; jj != ej; ++jj) {
        fn(*ii, *jj);
      }
    }
  }

  template<bool UseDense, typename Function>
  void executeBlock(Function& fn, Task& task, typename std::enable_if<UseDense>::type* = 0) {
    executeDenseBlock(fn, task);
  }

  template<bool UseDense, typename Function>
  void executeBlock(Function& fn, Task& task, typename std::enable_if<!UseDense>::type* = 0) {
    executeSparseBlock(fn, task);
  }

  template<bool UseDense, typename Function>
  void executeLoop(Function fn, unsigned tid, unsigned total) {
    if (false && UseExp)
      executeLoopExp2<UseDense>(fn, tid, total);
    else
      executeLoopOrig<UseDense>(fn, tid, total);
  }

  // bulk synchronous diagonals: static work assignment
  template<bool UseDense, typename Function>
  void executeLoopExp(Function fn, unsigned tid, unsigned total) {
    Point numBlocks { locks[0].size(), locks[1].size() };
    Point block;
    Point start;
    for (int i = 0; i < numDims; ++i) {
      block[i] = (numBlocks[i] + total - 1) / total;
      start[i] = std::min(block[i] * tid, numBlocks[i] - 1);
    }

    // Move diagonal along dim each round
    int dim = numBlocks[0] < numBlocks[1] ? 1 : 0;
    int odim = (dim + 1) % 2;
    size_t maxRounds = numBlocks[dim];

    for (size_t rounds = 0; rounds < maxRounds; ++rounds) {
      Point p { start[0], start[1] };
      nextPoint(p, dim, rounds);
      size_t ntries = std::min(block[odim] * (tid + 1), numBlocks[odim]) - start[odim];
      for (size_t tries = 0; tries < ntries; ++tries) {
        Task *t = probeBlock(p, 0, 1);
        if (t) {
          executeBlock<UseDense>(fn, *t);

          if (useLocks) {
            for (int i = 0; i < numDims; ++i)
              locks[i][t->coord[i]].unlock();
          }
        }
        for (int i = 0; i < numDims; ++i)
          nextPoint(p, i, 1);
      }

      barrier.wait();
    }
  }

  // bulk synchronous diagonals: dynamic assignment within diagonals
  template<bool UseDense, typename Function>
  void executeLoopExp2(Function fn, unsigned tid, unsigned total) {
    Point numBlocks { { locks[0].size(), locks[1].size() } };
    Point block;
    Point start;
    for (int i = 0; i < numDims; ++i) {
      block[i] = (numBlocks[i] + total - 1) / total;
      start[i] = std::min(block[i] * tid, numBlocks[i] - 1);
    }

    // Move diagonal along dim each round
    int dim = numBlocks[0] < numBlocks[1] ? 1 : 0;
    int odim = (dim + 1) % 2;
    size_t maxRounds = numBlocks[dim];

    for (size_t round = 0; round < maxRounds; ++round) {
      Point base { { start[0], start[1] } };
      nextPoint(base, dim, round);
      for (size_t tries = 0; tries < numBlocks[odim]; ++tries) {
        size_t index = tries + base[odim];
        if (index >= numBlocks[odim])
          index -= numBlocks[odim];
        Point p {};
        nextPoint(p, dim, round);
        nextPoint(p, odim, index);
        nextPoint(p, dim, index);

        Task *t = probeBlock(p, 0, 1);
        if (!t)
          continue;
        executeBlock<UseDense>(fn, *t);

        if (useLocks) {
          for (int i = 0; i < numDims; ++i)
            locks[i][t->coord[i]].unlock();
        }
      }

      barrier.wait();
    }
  }

  template<bool UseDense, typename Function>
  void executeLoopOrig(Function fn, unsigned tid, unsigned total) {
    Point numBlocks { { locks[0].size(), locks[1].size() } };
    Point block;
    Point start;
    for (int i = 0; i < numDims; ++i) {
      block[i] = (numBlocks[i] + total - 1) / total;
      start[i] = std::min(block[i] * tid, numBlocks[i] - 1);
    }

    unsigned coresPerPackage = LL::getMaxCores() / LL::getMaxPackages();
    if (useLocks)
      start = { { start[0], std::min(block[1] * LL::getPackageForThread(tid) * coresPerPackage, numBlocks[1] - 1) } };

    Point p = start;

    for (int i = 0; ; ++i) {
      Task *t = nextBlock(p, i == 0);
      // TODO: Replace with sparse worklist, etc.
      if (!t)
        break;

      executeBlock<UseDense>(fn, *t);

      if (useLocks) {
        for (int i = 0; i < numDims; ++i)
          locks[i][t->coord[i]].unlock();
      }
    }
  }

  /**
   * Returns a *locked* task by probing along dim; if task is found start is updated
   * to corresponding coodinate.
   */
  Task* probeBlock(Point& start, int dim, size_t n) {
    if (useLocks)
      return probeBlockWithLock(start, dim, n);
    return probeBlockWithoutLock(start, dim, n);
  }

  Task* probeBlockWithLock(Point& start, int dim, size_t n) {
    Point p = start;

    for (size_t i = 0; i < n; ++i) {
      Task* t = getTask(p);
      assert(p[0] == t->coord[0]);
      assert(p[1] == t->coord[1]);
      assert(t->coord[0] < locks[0].size());
      assert(t->coord[1] < locks[1].size());
      if (t->updates.relaxedLoad() >= maxUpdates) {
        ;
      } else if (std::try_lock(locks[0][t->coord[0]], locks[1][t->coord[1]]) < 0) {
        if (t->updates.relaxedLoad() < maxUpdates) {
          t->updates.relaxedAdd(1);
          start = p;
          return t;
        }
        // TODO add to worklist
        for (int i = 0; i < numDims; ++i)
          locks[i][t->coord[i]].unlock();
      }
      nextPoint(p, dim, 1);
    }

    failures += 1;
    return nullptr;
  }

  Task* probeBlockWithoutLock(Point& start, int dim, size_t n) {
    Point p = start;

    for (size_t i = 0; i < n; ++i) {
      Task* t = getTask(p);
      if (t->updates.relaxedLoad() >= maxUpdates) {
        ;
      } else if (t->updates.value.fetch_add(1) < maxUpdates) {
        start = p;
        return t;
      }
      nextPoint(p, dim, 1);
    }
    failures += 1;
    return nullptr;
  }


  // Nested dim1 then dim2
  Task* nextBlock(Point& start, bool inclusive) {
    Task* t;

    for (int times = 0; times < 2; ++times) {
      Point limit { { locks[0].size(), locks[1].size() } };
      int inclusiveDelta = inclusive && times == 0 ? 0 : 1;

      // First iteration is exclusive of start
      for (int i = 0; i < numDims; ++i) {
        Point p = start;
        nextPoint(p, i, inclusiveDelta);
        if ((t = probeBlock(p, i, limit[i] - inclusiveDelta))) {
          start = p;
          return t;
        }
      }

      Point p = start;
      for (int i = 0; i < numDims; ++i)
        nextPoint(p, i, 1);
      while (std::any_of(limit.begin(), limit.end(), [](size_t x) { return x > 0; })) {
        // Subsequent iterations are inclusive of start
        for (int i = 0; i < numDims; ++i) {
          if (limit[i] > 0 && (t = probeBlock(p, i, limit[i] - 1))) {
            start = p;
            return t;
          }
        }
        for (int i = 0; i < numDims; ++i) {
          if (limit[i] > 0) {
            limit[i] -= 1;
            nextPoint(p, i, 1);
          }
        }
      }
    }

    return nullptr;
  }

  void initializeTasks(iterator first0, iterator last0, iterator first1, iterator last1, size_t size0, size_t size1) {
    const size_t numBlocks0 = (std::distance(first0, last0) + size0 - 1) / size0;
    const size_t numBlocks1 = (std::distance(first1, last1) + size1 - 1) / size1;
    const size_t numBlocks = numBlocks0 * numBlocks1;

    //locks[0].create(numBlocks0);
    //locks[1].create(numBlocks1);
    //tasks.create(numBlocks);
    locks[0].resize(numBlocks0);
    locks[1].resize(numBlocks1);
    tasks.resize(numBlocks);

    GetDst fn { &g };

    for (size_t i = 0; i < numBlocks; ++i) {
      Task& task = tasks[i];
      task.coord = { { i % numBlocks0, i / numBlocks0 } };
      std::tie(task.start0, task.end0) = galois::block_range(first0, last0, task.coord[0], numBlocks0);
      iterator s, e;
      std::tie(s, e) = galois::block_range(first1, last1, task.coord[1], numBlocks1);
      // XXX: Works for CSR graphs
      task.start1 = *s;
      task.end1 = *e - 1;
    }
  }

  template<bool UseDense, typename Function>
  struct Process {
    Fixed2DGraphTiledExecutor* self;
    Function fn;

    void operator()(unsigned tid, unsigned total) {
      self->executeLoop<UseDense>(fn, tid, total);
    }
  };

public:
  Fixed2DGraphTiledExecutor(Graph& g, int cutoff = 0):
    g(g), cutoff(cutoff), barrier(galois::runtime::getSystemBarrier()), failures("PopFailures") { }

  template<typename Function>
  void execute(
      iterator first1, iterator last1,
      iterator first2, iterator last2,
      size_t size1, size_t size2,
      Function fn, bool _useLocks, int numIterations = 1) {
    initializeTasks(first1, last1, first2, last2, size1, size2);
    numTasks = tasks.size();
    maxUpdates = numIterations;
    useLocks = _useLocks;
    Process<false, Function> p { this, fn };
    galois::on_each(p);
    //TODO remove after worklist fix
    if (std::any_of(tasks.begin(), tasks.end(), [this](Task& t) { return t.updates.value < maxUpdates; }))
      galois::runtime::LL::gWarn("Missing tasks");
  }

  template<typename Function>
  void executeDense(
      iterator first1, iterator last1,
      iterator first2, iterator last2,
      size_t size1, size_t size2,
      Function fn, bool _useLocks, int numIterations = 1) {
    initializeTasks(first1, last1, first2, last2, size1, size2);
    numTasks = tasks.size();
    maxUpdates = numIterations;
    useLocks = _useLocks;
    Process<true, Function> p { this, fn };
    galois::on_each(p);
    //TODO remove after worklist fix
    if (std::any_of(tasks.begin(), tasks.end(), [this](Task& t) { return t.updates.value < maxUpdates; }))
      galois::runtime::LL::gWarn("Missing tasks");
  }
};

} } // end namespace
#endif
