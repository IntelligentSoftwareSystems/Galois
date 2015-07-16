/** Simple thread related classes -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @section Description
 *
 * Thread pool definition. Agnostic to type of threads.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_SUBSTRATE_THREADPOOL_H
#define GALOIS_SUBSTRATE_THREADPOOL_H

#include "Galois/config.h"
#include "CacheLineStorage.h"
#include "HWTopo.h"

#include <functional>
#include <atomic>
#include <vector>
#include <cassert>

namespace Galois {
namespace Substrate {

namespace detail {

template<typename tpl, int s, int r>
struct ExecuteTupleImpl {
  static inline void execute(tpl& cmds) {
    std::get<s>(cmds)();
    ExecuteTupleImpl<tpl,s+1,r-1>::execute(cmds);
  }
};

template<typename tpl, int s>
struct ExecuteTupleImpl<tpl, s, 0> {
  static inline void execute(tpl& f) { }
};

}

class ThreadPool {
protected:
  struct shutdown_ty {}; //! type for shutting down thread
  struct fastmode_ty {bool mode;}; //! type for setting fastmode


  //! Per-thread mailboxes for notification
  struct per_signal {
    std::atomic<int> done;
    std::atomic<int> fastRelease;
    std::atomic<per_signal*> next;
    HWTopo::threadInfo topo;
  };

  thread_local static per_signal my_box;

  unsigned maxThreads;
  std::function<void(void)> work; 
  std::atomic<unsigned> starting;
  unsigned masterFastmode;
  std::atomic<per_signal*> signals;
  bool running;

  ThreadPool(unsigned m);

  //!destroy all threads
  void destroyCommon();

  //! sleep this thread
  virtual void threadWait(unsigned tid) = 0;

  //! wake up thread
  virtual void threadWakeup(unsigned tid) = 0;

  //! Initialize a thread
  void initThread();

  //!main thread loop
  void threadLoop();

  //! spin up for run
  void cascade(bool fastmode);

  //! spin down after run
  void decascade();

  //! execute work on num threads
  void runInternal(unsigned num);

public:

  virtual ~ThreadPool();

  //! execute work on all threads
  //! a simple wrapper for run
  template<typename... Args>
  void run(unsigned num, Args&&... args) {
    struct ExecuteTuple {
      using Ty = std::tuple<Args...>;
      Ty cmds;

      void operator()(){
        detail::ExecuteTupleImpl<Ty, 0, std::tuple_size<Ty>::value>::execute(this->cmds);
      }
      ExecuteTuple(Args&&... args) :cmds(std::forward<Args>(args)...) {}
    };
    //paying for an indirection in work allows small-object optimization in std::function
    //to kick in and avoid a heap allocation
    ExecuteTuple lwork(std::forward<Args>(args)...);
    work = std::ref(lwork);
    //work = std::function<void(void)>(ExecuteTuple(std::forward<Args>(args)...));
    runInternal(num);
  }

  //experimental: busy wait for work
  void burnPower(unsigned num);
  //experimental: leave busy wait
  void beKind();

  //!return the number of threads supported by the thread pool on the current machine
  unsigned getMaxThreads() const { return maxThreads; }

  bool isRunning() const { return running; }

  static unsigned getTID() {
    return my_box.topo.tid;
  }
};

//!Returns or creates the appropriate thread pool for the system
ThreadPool& getSystemThreadPool();

} //Substrate
} //Galois

#endif
