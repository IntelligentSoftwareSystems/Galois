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

  //Data used in run loop
  std::vector<LL::CacheLineStorage<std::atomic<int>>> done; // signal loop done

  struct shutdown_ty {}; //! type for shutting down thread

  //! Initialize TID and PTS
  void initThread(unsigned tid);

  //!main thread loop
  void threadLoop(unsigned tid);

  //! spin up for run
  void cascade(int tid);

  //! spin down after run
  void decascade(int tid);

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
        detail::exTupleImpl<Ty, 0, std::tuple_size<Ty>::value>::execute(cmds);
      }
      exTuple(Args&&... args) :cmds(std::forward<Args>(args)...) {}
    };
    work = std::function<void(void)>(exTuple(std::forward<Args>(args)...));
    //sanitize num
    //seq write to starting should make work safe
    starting = std::min(std::max(1U,num), maxThreads);
    //launch threads
    cascade(0);
    // Do master thread work
    work();
    //wait for children
    decascade(0);
    // Clean up
    work = nullptr;
  }

  //!return the number of threads supported by the thread pool on the current machine
  unsigned getMaxThreads() const { return maxThreads; }
};

//!Returns or creates the appropriate thread pool for the system
ThreadPool& getSystemThreadPool();

} //Runtime
} //Galois

#endif
