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

#include "cycle.h"

#include "Galois/Runtime/Config.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/WorkListExperimental.h"

#include <list>

#define XXX_USE_PERLEVEL
//#define XXX_USE_PERCPU
//#define XXX_USE_PERHYBRID

namespace GaloisRuntime {

class SimpleTaskPool {
  struct TaskFunction {
    virtual int work(bool serial = false) = 0;
    virtual ~TaskFunction() { }
  };

  typedef GaloisRuntime::WorkList::FIFO<TaskFunction*> Worklist;
#ifdef XXX_USE_PERLEVEL
  PerLevel<Worklist> worklists;
#endif
#ifdef XXX_USE_PERCPU
  PerCPU<Worklist> worklists;
#endif
#ifdef XXX_USE_PERHYBRID
  PerCPU<Worklist> worklists;
#endif

  template<typename IterTy, typename FunctionTy, typename SFunctionTy>
  struct Task: public TaskFunction {
    IterTy begin;
    IterTy end;
    FunctionTy fn;
    SFunctionTy& sf;
    int flag;
    volatile unsigned int& done;

    Task(const IterTy& b, const IterTy& e, const FunctionTy& f, SFunctionTy& s,
        volatile unsigned int& d):
      begin(b), end(e), fn(f), sf(s), flag(0), done(d) { }

    virtual ~Task() { }

    int run(bool serial) {
      int iterations = 0;

      if (__sync_bool_compare_and_swap(&flag, 0, 1)) {
        for (; begin != end; ++begin) {
          if (serial)
            sf(*begin);
          else
            fn(*begin);
          ++iterations;
        }
        __sync_add_and_fetch(&done, 1);
      }

      return iterations;
    }

    virtual int work(bool serial = false) {
      return run(serial);
    }
  };

  template<typename IterTy,typename FunctionTy,typename SFunctionTy>
  void enqueueAndRun(const IterTy& begin, const IterTy& end,
      const FunctionTy& f, const SFunctionTy& sf, size_t n, unsigned int numThreads) {

    ticks s1 = getticks();

    typedef Task<IterTy,FunctionTy,SFunctionTy> MyTask;
    typedef std::list<MyTask> Tasks;

    Tasks tasks;
    size_t numBlocks = 4 * numThreads;
    size_t blockSize = n / numBlocks;

    volatile unsigned int done = 0;
    IterTy b(begin);
    IterTy e(begin);

    Worklist& wl = worklists.get();
    SFunctionTy myFn(sf);

    for (size_t i = 0; i < numBlocks; i++) {
      if (i == numBlocks - 1) {
        tasks.push_back(MyTask(b, end, f, myFn, done));
      } else {
        std::advance(e, blockSize);
        tasks.push_back(MyTask(b, e, f, myFn, done));
      }
      wl.push(&tasks.back());    
      b = e;
    }

    ticks s2 = getticks();

    boost::optional<TaskFunction*> r = wl.pop();
    while (r) {
      (*r)->work(true);
      r = wl.pop();
    }

    ticks s3 = getticks();

    // XXX still broken need non-idempotent worklist
    while (done < numBlocks) {
#if defined(__i386__) || defined(__amd64__)
	asm volatile ("pause");
#endif
    }
    ticks send = getticks();

    printticks("enqueue", 4, s1, s2, s3, send);
  }

public:
  template<typename IterTy,typename FunctionTy,typename SFunctionTy>
  void for_each(const IterTy& begin, const IterTy& end, const FunctionTy& f, const SFunctionTy& sf) {
    ticks s1 = getticks();

    bool serial = false;

    size_t n;
    if (!serial) {
      n = std::distance(begin, end);
      if (n < 16*1024)
        serial = true;
    }

    if (serial) {
      std::for_each(begin, end, sf);
    } else {
      unsigned int numThreads = getSystemThreadPool().getActiveThreads();
      enqueueAndRun(begin, end, f, sf, n, numThreads);
    }

    ticks send = getticks();

    printticks("total", 2, s1, send);
  }

  int work() {
    ticks s1 = getticks();

    int iterations = 0;
#ifdef XXX_USE_PERLEVEL
    Worklist& wl = worklists.get();
    boost::optional<TaskFunction*> r = wl.pop();
    while (r) {
      iterations += (*r)->work();
      r = wl.pop();
    }
#endif
#ifdef XXX_USE_PERHYBRID
    unsigned int numThreads = getSystemThreadPool().getActiveThreads();
    unsigned int myId = worklists.myEffectiveID();
    unsigned int threads_per_package = 4;
    unsigned int base = myId / threads_per_package * threads_per_package;
    unsigned int top = base + threads_per_package;
    bool didWork = false;

    for (unsigned int i = base; i < numThreads && i < top; ++i) {
      Worklist& wl = worklists.get(i);
      std::pair<bool,TaskFunction*> r = wl.pop();
      while (r.first) {
        iterations += r.second->work();
        r = wl.pop();
        didWork = true;
      }
      if (didWork)
        break;
    }
#endif
#ifdef XXX_USE_PERCPU
    unsigned int numThreads = getSystemThreadPool().getActiveThreads();

    bool didWork = false;

    for (unsigned int i = 0; i < numThreads ; ++i) {
      Worklist& wl = worklists.get(i);
      std::pair<bool,TaskFunction*> r = wl.pop();
      while (r.first) {
        iterations += r.second->work();
        r = wl.pop();
        didWork = true;
      }
      if (didWork)
        break;
    }
#endif

    ticks end = getticks();

    if (iterations > 0) {
      printticks("work", 2, s1, end);
    }

    return iterations;
  }
};

SimpleTaskPool& getSystemTaskPool();

}

#endif
