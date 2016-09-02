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

#include <condition_variable>
#include <thread>
#include <functional>
#include <atomic>
//#include <vector>
#include <cassert>
//#include <cstdlib>

namespace Galois {
namespace Runtime {

class ThreadPool {
protected:
  struct shutdown_ty {}; //! type for shutting down thread
  struct fastmode_ty {bool mode;}; //! type for setting fastmode
  struct dedicated_ty {std::function<void(void)> fn;}; //! type to switch to dedicated mode


  //! Per-thread mailboxes for notification
  struct per_signal {
    std::condition_variable cv;
    std::mutex m;
    unsigned wbegin, wend;
    std::atomic<int> done;
    std::atomic<int> fastRelease;
    threadTopoInfo topo;

    void wakeup(bool fastmode);
    void wait(bool fastmode);
  };

  thread_local static per_signal my_box;

  bool no_bind;
  bool no_bind_main;
  machineTopoInfo mi;
  std::vector<per_signal*> signals;
  std::vector<std::thread>  threads;
  unsigned reserved;
  unsigned masterFastmode;
  bool running;
  std::function<void(void)> work;



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

  ThreadPool(bool _no_bind, bool _no_bind_main);

public:
  ~ThreadPool();

  //! execute work on num threads
  void run(unsigned num, std::function<void(void)>&& fn);

  //! run function in a dedicated thread until the threadpool exits
  void runDedicated(std::function<void(void)>& f);

  //experimental: busy wait for work
  void burnPower(unsigned num);
  //experimental: leave busy wait
  void beKind();

  bool isRunning() const { return running; }


  //!return the number of non-reserved threads in the pool
  unsigned getMaxUsableThreads() const { return mi.maxThreads - reserved; }
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

  static ThreadPool& getThreadPool(bool no_bind=false, bool no_bind_main=false);
};

} //Substrate
} //Galois

#endif
