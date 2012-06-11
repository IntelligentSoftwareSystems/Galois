/** Simple task pool -*- C++ -*-
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
 * @section Description
 *
 * Simple task pool to collect small tasks that can be run in parallel without
 * complicated support for detecting conflicts, etc. Intended to enbedded inside
 * a runtime system that directs threads to methods of the task pool.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_SIMPLETASKPOOL_H
#define GALOIS_RUNTIME_SIMPLETASKPOOL_H

#include "Galois/Mem.h"
#include "Galois/Runtime/Config.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/WorkList.h"

#include <list>
#include <boost/optional.hpp>

namespace GaloisRuntime {

struct TaskFunction {
  virtual int work() = 0;
  virtual ~TaskFunction() { }
};

template<typename IterTy, typename FunctionTy>
struct Task: public TaskFunction {
  IterTy begin;
  IterTy end;
  FunctionTy fn;
  int flag;
  volatile unsigned& done;

  Task(const IterTy& b, const IterTy& e, FunctionTy f, volatile unsigned& d):
    begin(b), end(e), fn(f), flag(0), done(d) { }

  virtual ~Task() { }

  int run() {
    int iterations = 0;

    if (__sync_bool_compare_and_swap(&flag, 0, 1)) {
      for (; begin != end; ++begin) {
        fn(*begin);
        ++iterations;
      }
      __sync_add_and_fetch(&done, 1);
    }

    return iterations;
  }

  virtual int work() {
    return run();
  }
};

template<typename IterTy,typename FunctionTy> struct TaskContext;

class SimpleTaskPool {
  typedef GaloisRuntime::WorkList::dChunkedLIFO<4, TaskFunction*> Worklist;
  Worklist worklist;

public:
  template<typename IterTy,typename FunctionTy>
  void enqueue(TaskContext<IterTy,FunctionTy>& ctx,
      const IterTy& begin, const IterTy& end, FunctionTy fn);
  
  //! Called by runtime to periodically clear pending tasks
  int work(int count = 0) {
    int iterations = 0;
    int c = 0;

    boost::optional<TaskFunction*> r = worklist.pop();
    while (r) {
      iterations += (*r)->work();
      if (count > 0 && ++c >= count)
        break;
      r = worklist.pop();
    }

    return iterations;
  }
};

//! Holds task data for one invocation
template<typename IterTy,typename FunctionTy>
class TaskContext: public boost::noncopyable {
  friend class SimpleTaskPool;
  typedef Task<IterTy,FunctionTy> MyTask;
  struct Node {
    void *x, *y;
    MyTask z;
  };

  // XXX(ddn): Assumes list nodes are of a certain shape
  typedef Galois::GFixedAllocator<Node> Allocator;
  typedef std::list<MyTask, Allocator> Tasks;

  Tasks tasks;
  volatile unsigned done;
  size_t numBlocks;

  template<typename WorklistTy>
  void enqueue(const IterTy& begin, const IterTy& end, WorklistTy& wl, FunctionTy fn) {
    int numThreads = Galois::getActiveThreads();
//    numBlocks = 4 * numThreads;
    numBlocks = 4 * numThreads;

    size_t n = std::distance(begin, end);
    size_t blockSize = n / numBlocks;

    IterTy b(begin);
    IterTy e(begin);

    for (size_t i = 0; i < numBlocks; i++) {
      if (i == numBlocks - 1) {
        tasks.push_back(MyTask(b, end, fn, done));
      } else {
        std::advance(e, blockSize);
        tasks.push_back(MyTask(b, e, fn, done));
      }
      wl.push(&tasks.back());    
      b = e;
    }
  }

public:
  TaskContext(): done(0) { }

  int run(SimpleTaskPool& pool) {
    int iterations = 0;
    for (typename Tasks::iterator ii = tasks.begin(), ei = tasks.end(); ii != ei; ++ii)
      iterations += ii->work();

    while (done < numBlocks) {
      iterations += pool.work(1);
    }

    return iterations;
  }
};

template<typename IterTy,typename FunctionTy>
void SimpleTaskPool::enqueue(TaskContext<IterTy,FunctionTy>& ctx,
      const IterTy& begin, const IterTy& end, FunctionTy fn) {
  ctx.enqueue(begin, end, worklist, fn);
}

SimpleTaskPool& getSystemTaskPool();

}

#endif
