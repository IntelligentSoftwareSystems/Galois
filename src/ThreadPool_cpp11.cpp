/** pthread thread pool implementation -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/ll/EnvCheck.h"
#include "Galois/Runtime/ll/HWTopo.h"
#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include "boost/utility.hpp"

#include <thread>
#include <condition_variable>
#include <algorithm>
#include <atomic>

// Forward declare this to avoid including PerThreadStorage.
// We avoid this to stress that the thread Pool MUST NOT depend on PTS.
namespace Galois {
namespace Runtime {
extern void initPTS();
}
}

using namespace Galois::Runtime;

namespace {

class Semaphore: private boost::noncopyable {
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
    cv.wait(lg, [=]{ return 0 != count; });
    --count;
  }
};

class ThreadPool_cpp11 : public ThreadPool {
  std::vector<std::thread> threads;
  std::vector<Semaphore> starts;  // Signal to release threads to run
  std::atomic<unsigned> started;
  std::atomic<bool> shutdown; // Set and start threads to have them exit
  std::atomic<unsigned> starting; // Each run call uses this to control num threads
  std::atomic<RunCommand*> workBegin; // Begin iterator for work commands
  std::atomic<RunCommand*> workEnd; // End iterator for work commands

  void initThread(unsigned tid) {
    // Initialize TID
    Galois::Runtime::LL::initTID(tid);
    Galois::Runtime::initPTS();
    if (!LL::EnvCheck("GALOIS_DO_NOT_BIND_THREADS"))
      if (tid != 0 || !LL::EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD"))
	Galois::Runtime::LL::bindThreadToProcessor(tid);
    // Use a simple pthread or atomic to avoid depending on Galois
    // too early in the initialization process
    ++started;
  }

  void cascade(int tid) {
    const unsigned multiple = 2;
    for (unsigned i = 1; i <= multiple; ++i) {
      unsigned n = tid * multiple + i;
      if (n < starting)
        starts[n].release();
    }
  }

  void doWork() {
    RunCommand* workPtr = workBegin;
    RunCommand* workEndL = workEnd;
    while (workPtr != workEndL) {
      (*workPtr)();
      ++workPtr;
    }
  }

  void mlaunch(unsigned tid) {
    initThread(tid);
    while (!shutdown) {
      starts[tid].acquire();
      cascade(tid);
      doWork();
      --started;
    }
  }

public:
  ThreadPool_cpp11():
    ThreadPool(Galois::Runtime::LL::getMaxThreads()),
    starts(maxThreads),
    started(0), shutdown(false), workBegin(0), workEnd(0)
  {
    initThread(0);

    for (unsigned i = 1; i < maxThreads; ++i) {
      std::thread t(&ThreadPool_cpp11::mlaunch, this, i);
      threads.emplace_back(std::move(t));
    }
    while(started < maxThreads)
      std::this_thread::yield();
  }

  virtual ~ThreadPool_cpp11() {
    shutdown = true;
    workBegin = nullptr;
    workEnd = nullptr;
    for (auto& s : starts)
      s.release();
    for(auto& t : threads)
      t.join();
  }

  virtual void run(RunCommand* begin, RunCommand* end, unsigned num) {
    // Sanitize num
    num = std::min(std::max(num,1U), maxThreads);
    starting = num;
    // Setup work
    workBegin = begin;
    workEnd = end;
    //atomic ensures seq consistency for workBegin,End
    //launch threads
    started = num - 1;
    cascade(0);
    // Do master thread work
    doWork();
    //wait for children
    while (started) { LL::asmPause(); }
    // Clean up
    workBegin = 0;
    workEnd = 0;
  }
};

} // end namespace

#if 1
//! Implement the global threadpool
ThreadPool& Galois::Runtime::getSystemThreadPool() {
  static ThreadPool_cpp11 pool;
  return pool;
}
#endif
