/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_SUBSTRATE_THREADPOOL_H
#define GALOIS_SUBSTRATE_THREADPOOL_H

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

#include "galois/substrate/CacheLineStorage.h"
#include "galois/substrate/HWTopo.h"

namespace galois::substrate::internal {

template <typename tpl, int s, int r>
struct ExecuteTupleImpl {
  static inline void execute(tpl& cmds) {
    std::get<s>(cmds)();
    ExecuteTupleImpl<tpl, s + 1, r - 1>::execute(cmds);
  }
};

template <typename tpl, int s>
struct ExecuteTupleImpl<tpl, s, 0> {
  static inline void execute(tpl&) {}
};

} // namespace galois::substrate::internal

namespace galois::substrate {

class ThreadPool {
  friend class SharedMem;

protected:
  struct shutdown_ty {}; //! type for shutting down thread
  struct fastmode_ty {
    bool mode;
  }; //! type for setting fastmode
  struct dedicated_ty {
    std::function<void(void)> fn;
  }; //! type to switch to dedicated mode

  //! Per-thread mailboxes for notification
  struct per_signal {
    std::condition_variable cv;
    std::mutex m;
    unsigned wbegin, wend;
    std::atomic<int> done;
    std::atomic<int> fastRelease;
    ThreadTopoInfo topo;

    void wakeup(bool fastmode) {
      if (fastmode) {
        done        = 0;
        fastRelease = 1;
      } else {
        std::lock_guard<std::mutex> lg(m);
        done = 0;
        cv.notify_one();
        // start.release();
      }
    }

    void wait(bool fastmode) {
      if (fastmode) {
        while (!fastRelease.load(std::memory_order_relaxed)) {
          asmPause();
        }
        fastRelease = 0;
      } else {
        std::unique_lock<std::mutex> lg(m);
        cv.wait(lg, [=] { return !done; });
        // start.acquire();
      }
    }
  };

  thread_local static per_signal my_box;

  MachineTopoInfo mi;
  std::vector<per_signal*> signals;
  std::vector<std::thread> threads;
  unsigned reserved;
  unsigned masterFastmode;
  bool running;
  std::function<void(void)> work;

  //! destroy all threads
  void destroyCommon();

  //! Initialize a thread
  void initThread(unsigned tid);

  //! main thread loop
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

  ThreadPool(const ThreadPool&)            = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  ThreadPool(ThreadPool&&)            = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  //! execute work on all threads
  //! a simple wrapper for run
  template <typename... Args>
  void run(unsigned num, Args&&... args) {
    struct ExecuteTuple {
      //      using Ty = std::tuple<Args...>;
      std::tuple<Args...> cmds;

      void operator()() {
        internal::ExecuteTupleImpl<
            std::tuple<Args...>, 0,
            std::tuple_size<std::tuple<Args...>>::value>::execute(this->cmds);
      }
      ExecuteTuple(Args&&... args) : cmds(std::forward<Args>(args)...) {}
    };
    // paying for an indirection in work allows small-object optimization in
    // std::function to kick in and avoid a heap allocation
    ExecuteTuple lwork(std::forward<Args>(args)...);
    work = std::ref(lwork);
    // work =
    // std::function<void(void)>(ExecuteTuple(std::forward<Args>(args)...));
    assert(num <= getMaxThreads());
    runInternal(num);
  }

  //! run function in a dedicated thread until the threadpool exits
  void runDedicated(std::function<void(void)>& f);

  // experimental: busy wait for work
  void burnPower(unsigned num);
  // experimental: leave busy wait
  void beKind();

  bool isRunning() const { return running; }

  //! return the number of non-reserved threads in the pool
  unsigned getMaxUsableThreads() const { return mi.maxThreads - reserved; }
  //! return the number of threads supported by the thread pool on the current
  //! machine
  unsigned getMaxThreads() const { return mi.maxThreads; }
  unsigned getMaxCores() const { return mi.maxCores; }
  unsigned getMaxSockets() const { return mi.maxSockets; }
  unsigned getMaxNumaNodes() const { return mi.maxNumaNodes; }

  unsigned getLeaderForSocket(unsigned pid) const {
    for (unsigned i = 0; i < getMaxThreads(); ++i)
      if (getSocket(i) == pid && isLeader(i))
        return i;
    abort();
  }

  bool isLeader(unsigned tid) const {
    return signals[tid]->topo.socketLeader == tid;
  }
  unsigned getSocket(unsigned tid) const { return signals[tid]->topo.socket; }
  unsigned getLeader(unsigned tid) const {
    return signals[tid]->topo.socketLeader;
  }
  unsigned getCumulativeMaxSocket(unsigned tid) const {
    return signals[tid]->topo.cumulativeMaxSocket;
  }
  unsigned getNumaNode(unsigned tid) const {
    return signals[tid]->topo.numaNode;
  }

  static unsigned getTID() { return my_box.topo.tid; }
  static bool isLeader() { return my_box.topo.tid == my_box.topo.socketLeader; }
  static unsigned getLeader() { return my_box.topo.socketLeader; }
  static unsigned getSocket() { return my_box.topo.socket; }
  static unsigned getCumulativeMaxSocket() {
    return my_box.topo.cumulativeMaxSocket;
  }
  static unsigned getNumaNode() { return my_box.topo.numaNode; }
};

/**
 * return a reference to system thread pool
 */
ThreadPool& getThreadPool(void);

} // namespace galois::substrate

namespace galois::substrate::internal {

void setThreadPool(ThreadPool* tp);

} // namespace galois::substrate::internal

#endif
