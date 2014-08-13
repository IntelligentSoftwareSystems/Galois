/** Simple thread related classes -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_THREADPOOL_H
#define GALOIS_RUNTIME_THREADPOOL_H

#include "Galois/config.h"

#include "Galois/Runtime/ll/CacheLineStorage.h"

#include <functional>
#include <atomic>
#include <vector>
#include <cassert>

namespace Galois {
namespace Runtime {

namespace detail {
template<typename tpl, int s, int r>
struct exTupleImpl {
  static inline void execute(tpl& cmds) {
    std::get<s>(cmds)();
    exTupleImpl<tpl,s+1,r-1>::execute(cmds);
  }
};
template<typename tpl, int s>
struct exTupleImpl<tpl, s, 0> {
  static inline void execute(tpl& f) { }
};
}

class ThreadPool {
protected:
  unsigned maxThreads;
  ThreadPool(unsigned m);

  //!destroy all threads
  void destroyCommon();

  //! sleep this thread
  virtual void threadWait(unsigned tid) = 0;

  //! wake up thread
  virtual void threadWakeup(unsigned tid) = 0;

  //Common implementation stuff

  //Data passed to threads through run
  std::function<void(void)> work; //active work command
  std::atomic<unsigned> starting; // number of threads
  unsigned masterFastmode; // use fastmode //only seen by thread 0

  //Data used in run loop
  struct per_signal {
    std::atomic<int> done;
    std::atomic<int> fastRelease;
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1310
    per_signal (void): done (), fastRelease () {}
#endif
  };
  std::vector<LL::CacheLineStorage<per_signal>> signals; // signal loop

  struct shutdown_ty {}; //! type for shutting down thread
  struct fastmode_ty {bool mode;}; //! type for setting fastmode

  //! Initialize TID and PTS
  void initThread(unsigned tid);

  //!main thread loop
  void threadLoop(unsigned tid);

  //! spin up for run
  void cascade(int tid, bool fastmode);

  //! spin down after run
  void decascade(int tid);

  //! execute work on num threads
  void runInternal(unsigned num);

public:
  virtual ~ThreadPool();

  //! execute work on all threads
  //! a simple wrapper for run
  template<typename... Args>
  void run(unsigned num, Args&&... args) {
    struct exTuple {
      using Ty = std::tuple<Args...>;
      Ty cmds;

      void operator() () {
        detail::exTupleImpl<Ty, 0, std::tuple_size<Ty>::value>::execute(this->cmds);
      }
      exTuple(Args&&... args) :cmds(std::forward<Args>(args)...) {}
    };
    //paying for an indirection in work allows small-object optimization in std::function to kick in and avoid a heap allocation
    exTuple lwork(std::forward<Args>(args)...);
    work = std::ref(lwork);
    //work = std::function<void(void)>(exTuple(std::forward<Args>(args)...));
    runInternal(num);

  }

  void burnPower(unsigned num);
  void beKind();

  //!return the number of threads supported by the thread pool on the current machine
  unsigned getMaxThreads() const { return maxThreads; }
};

//!Returns or creates the appropriate thread pool for the system
ThreadPool& getSystemThreadPool();

} //Runtime
} //Galois

#endif
