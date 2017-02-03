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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Thread pool definition. Agnostic to type of threads.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_THREADPOOL_H
#define GALOIS_RUNTIME_THREADPOOL_H

#include "HWTopo.h"
#include "CompilerSpecific.h"

#include <condition_variable>
#include <thread>
#include <functional>
#include <atomic>
#include <cassert>

namespace Galois {
namespace Runtime {

void initPTS(unsigned max);


class ThreadPool {
protected:
  struct local_state {
    std::condition_variable cv;
    std::mutex m;
    
    //null when done, set by thread
    //not null set by signaller
    std::function<void(void)> work;
    std::atomic<int> state;
    std::atomic<bool> fastmode;
    
    threadTopoInfo topo;

    void wakeup_remote(const std::function<void(void)>& w, unsigned num);
    void detach_remote(const std::function<void(void)>& w);
    void wait_local();
    //waits until thread is idle
    void wait_remote();
    bool try_wait_remote();
  };

  thread_local static local_state tcb;

  machineTopoInfo mi;
  std::unique_ptr<std::atomic<local_state*>[]> signals;
  unsigned sizeSignal;
  std::vector<std::thread>  threads;
  bool running;

  //! Initialize a thread
  void initThread(unsigned tid, bool bindCore);

  //!main thread loop
  void threadLoop(unsigned tid, bool bindCore);

  //! spin up for run
  void cascade(const std::function<void(void)>& fn, unsigned num);

  //! spin down after run
  void decascade(unsigned num);

  ThreadPool();

public:
  ~ThreadPool();

  //! execute work on num threads
  void run(unsigned num, const std::function<void(void)>& fn);

  //! run function in a dedicated thread until the threadpool exits
  void runDedicated(const std::function<void(void)>&& f);

  //experimental: busy wait for work
  void burnPower(unsigned num);
  //experimental: leave busy wait
  void beKind();

  bool isRunning() const { return running; }


  //!return the number of non-reserved threads in the pool
  unsigned getMaxUsableThreads() const { return sizeSignal; }
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
  
  bool isLeader(unsigned tid) const { return signals[tid].load(std::memory_order_relaxed)->topo.socketLeader == tid; }
  unsigned getPackage(unsigned tid) const { return signals[tid].load(std::memory_order_relaxed)->topo.socket; }
  unsigned getLeader(unsigned tid) const { return signals[tid].load(std::memory_order_relaxed)->topo.socketLeader; }
  unsigned getCumulativeMaxPackage(unsigned tid) const { return signals[tid].load(std::memory_order_relaxed)->topo.cumulativeMaxSocket; }
  unsigned getNumaNode(unsigned tid) const { return signals[tid].load(std::memory_order_relaxed)->topo.numaNode; }

  static unsigned getTID() { return tcb.topo.tid; }
  static bool isLeader() { return tcb.topo.tid == tcb.topo.socketLeader; }
  static unsigned getLeader() { return tcb.topo.socketLeader; }
  static unsigned getPackage() { return tcb.topo.socket; }
  static unsigned getCumulativeMaxPackage() { return tcb.topo.cumulativeMaxSocket; }
  static unsigned getNumaNode() { return tcb.topo.numaNode; }

  static ThreadPool& getThreadPool();
};

} //Substrate
} //Galois

#endif
