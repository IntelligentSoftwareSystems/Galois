/** Simple thread related classes -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * @section Description
 *
 * Thread pool definition. Agnostic to type of threads.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_SUBSTRATE_THREADPOOL_H
#define GALOIS_SUBSTRATE_THREADPOOL_H

#include "CacheLineStorage.h"
#include "HWTopo.h"

#include <condition_variable>
#include <thread>
#include <functional>
#include <atomic>
#include <vector>
#include <cassert>
#include <cstdlib>

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

class Semaphore { //not copy or movable
  std::mutex m;
  std::condition_variable cv;
  int count;

public:
  explicit Semaphore() : count(0) {}
  ~Semaphore() {}

  void release() {
    std::lock_guard<std::mutex> lg(m);
    ++count;
    cv.notify_one();
  }

  void acquire() {
    std::unique_lock<std::mutex> lg(m);
    cv.wait(lg, [=]{ return 0 < count; });
    --count;
  }
};

}

class ThreadPool {
protected:
  struct shutdown_ty {}; //! type for shutting down thread
  struct fastmode_ty {bool mode;}; //! type for setting fastmode


  //! Per-thread mailboxes for notification
  struct per_signal {
    detail::Semaphore start;
    std::atomic<int> done;
    std::atomic<int> fastRelease;
    threadTopoInfo topo;

    void wakeup() {
      start.release();
    }

    void wait() {
      start.acquire();
    }
  };

  thread_local static per_signal my_box;

  std::vector<per_signal*> signals;
  std::vector<std::thread>  threads;

  machineTopoInfo mi;

  std::function<void(void)> work; 
  std::atomic<unsigned> starting;
  unsigned masterFastmode;
  bool running;

  //!destroy all threads
  void destroyCommon();

  //! Initialize a thread
  void initThread(unsigned tid);

  //!main thread loop
  void threadLoop(unsigned tid);

  //! spin up for run
  void cascade(bool fastmode);

  //! spin down after run
  void decascade();

  //! execute work on num threads
  void runInternal(unsigned num);

  ThreadPool();

public:
  ~ThreadPool();

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
    assert(num <= getMaxThreads());
    runInternal(num);
  }

  //experimental: busy wait for work
  void burnPower(unsigned num);
  //experimental: leave busy wait
  void beKind();

  bool isRunning() const { return running; }


  //!return the number of threads supported by the thread pool on the current machine
  unsigned getMaxThreads() const { return mi.maxThreads; }
  unsigned getMaxCores() const { return mi.maxCores; }
  unsigned getMaxPackages() const { return mi.maxPackages; }
  unsigned getMaxNumaNodes() const { return mi.maxNumaNodes; }

  unsigned getLeaderForPackage(unsigned pid) const {
    for (unsigned i = 0; i < getMaxThreads(); ++i)
      if (getPackage(i) == pid && isLeader(i))
        return i;
    abort();
  }
  
  bool isLeader(unsigned tid) const { return signals[tid]->topo.socketLeader == tid; }
  unsigned getPackage(unsigned tid) const { return signals[tid]->topo.socket; }
  unsigned getLeader(unsigned tid) const { return signals[tid]->topo.socketLeader; }
  unsigned getCumulativeMaxPackage(unsigned tid) const { return signals[tid]->topo.cumulativeMaxSocket; }
  unsigned getNumaNode(unsigned tid) const { return signals[tid]->topo.numaNode; }

  static unsigned getTID() { return my_box.topo.tid; }
  static bool isLeader() { return my_box.topo.tid == my_box.topo.socketLeader; }
  static unsigned getLeader() { return my_box.topo.socketLeader; }
  static unsigned getPackage() { return my_box.topo.socket; }
  static unsigned getCumulativeMaxPackage() { return my_box.topo.cumulativeMaxSocket; }
  static unsigned getNumaNode() { return my_box.topo.numaNode; }

  static ThreadPool& getThreadPool();
};

} //Substrate
} //Galois

#endif
