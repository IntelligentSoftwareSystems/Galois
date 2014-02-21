#ifndef TILEDEXECUTOR_H
#define TILEDEXECUTOR_H

#include "Galois/Galois.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/NoDerefIterator.h"
#include "Galois/gstl.h"

#include <boost/iterator/transform_iterator.hpp>
#include <atomic>
#include <vector>
#include <cstddef>
#include <functional>
#include <mutex>
//#include <iostream> // XXX

template<typename Graph>
class Fixed2DGraphTiledExecutor {
  typedef Galois::Runtime::LL::PaddedLock<true> SpinLock;
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

  Graph& g;
  std::vector<SpinLock> locks1;
  std::vector<SpinLock> locks2;
  std::vector<Task> tasks;
  size_t maxUpdates;
  bool useLocks;

  template<typename Function>
  void executeSparseBlock(Function& fn, Task& task) {
    GetDst getDst { &g };

    for (auto ii = task.start1; ii != task.end1; ++ii) {
      edge_iterator begin = g.edge_begin(*ii);
      no_deref_iterator nbegin(begin);
      no_deref_iterator nend(g.edge_end(*ii));
      edge_dst_iterator dbegin(nbegin, getDst);
      edge_dst_iterator dend(nend, getDst);

      for (auto jj = std::lower_bound(dbegin, dend, task.start2); jj != dend; ++jj) {
        edge_iterator edge = *jj.base();
        if (*jj > task.end2)
          break;
          
        fn(*ii, *jj, edge);
      }
    }
  }

  template<typename Function>
  void executeDenseBlock(Function& fn, Task& task) {
    GetDst getDst { &g };

    for (auto ii = task.start1; ii != task.end1; ++ii) {
      for (auto jj = g.begin() + task.start2, ej = g.begin() + task.end2 + 1; jj != ej; ++jj) {
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
    const size_t numBlocks1 = locks1.size();
    const size_t numBlocks2 = locks2.size();
    const size_t numBlocks = numBlocks1 * numBlocks2;
    const size_t block1 = (numBlocks1 + total - 1) / total;
    const size_t start1 = std::min(block1 * tid, numBlocks1 - 1);
    const size_t block2 = (numBlocks2 + total - 1) / total;
    const size_t start2 = std::min(block2 * tid, numBlocks2 - 1);

    // TODO get actual cores per package
    size_t start;
    if (useLocks)
      //start = block1 * 10 * (tid / 10) + start2 * numBlocks1;
      start = start1 + block2 * 10 * (tid / 10) * numBlocks1;
    else
      start = start1 + start2 * numBlocks1;

    for (int i = 0; ; ++i) {
      start = nextBlock(start, numBlocks, i == 0);
      Task* t = &tasks[start];
      if (t == &tasks[numBlocks])
        break;
      executeBlock<UseDense>(fn, *t);

      if (useLocks) {
        locks1[t->d1].unlock();
        locks2[t->d2].unlock();
      }
    }
  }

  size_t probeBlock(size_t start, size_t by, size_t n, size_t numBlocks) {
    for (size_t i = 0; i < n; ++i, start += by) {
      while (start >= numBlocks)
        start -= numBlocks;
      Task& b = tasks[start];
      if (b.updates.relaxedLoad() < maxUpdates) {
        if (useLocks) {
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

    //std::cout << "XXX " << numBlocks1 << " " << numBlocks2 << "\n";
    for (size_t i = 0; i < numBlocks; ++i) {
      Task& task = tasks[i];
      task.d1 = i % numBlocks1;
      task.d2 = i / numBlocks1;
      task.id = i;
      std::tie(task.start1, task.end1) = Galois::block_range(first1, last1, task.d1, numBlocks1);
      iterator s, e;
      std::tie(s, e) = Galois::block_range(first2, last2, task.d2, numBlocks2);
      // XXX: Works for CSR graphs
      task.start2 = *s;
      task.end2 = *e - 1;
      //std::cout << "YYY " << *task.start1 << " " << *task.end1 << " " << task.start2 << " " << task.end2 << "\n";
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
  Fixed2DGraphTiledExecutor(Graph& _g): g(_g) { }

  template<typename Function>
  void execute(
      iterator first1, iterator last1,
      iterator first2, iterator last2,
      size_t size1, size_t size2,
      Function fn, bool _useLocks, size_t numIterations = 1) {
    initializeTasks(first1, last1, first2, last2, size1, size2);
    maxUpdates = numIterations;
    useLocks = _useLocks;
    Process<false, Function> p = { this, fn };
    Galois::on_each(p);
  }

  template<typename Function>
  void executeDense(
      iterator first1, iterator last1,
      iterator first2, iterator last2,
      size_t size1, size_t size2,
      Function fn, bool _useLocks, size_t numIterations = 1) {
    initializeTasks(first1, last1, first2, last2, size1, size2);
    maxUpdates = numIterations;
    useLocks = _useLocks;
    Process<true, Function> p = { this, fn };
    Galois::on_each(p);
  }
};
#endif
