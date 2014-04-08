/** std::thread thread pool implementation -*- C++ -*-
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
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/ll/CacheLineStorage.h"
#include "Galois/Runtime/ll/HWTopo.h"

#include "boost/utility.hpp"

#include <thread>
#include <condition_variable>
#include <algorithm>
#include <atomic>
#include <vector>

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
  std::atomic<unsigned> starting; // Each run call uses this to control num threads
  std::function<void(void)> work; // active work command
  std::vector<LL::CacheLineStorage<std::atomic<int>>> done; // signal loop done

  struct shutdown_ty {};

  void initThread(unsigned tid, bool wait = true) {
    ThreadPool::initThreadCommon(tid);
    if (wait)
      decascade(tid);
  }

  void cascade(int tid) {
    const unsigned multiple = 2;
    for (unsigned i = 1; i <= multiple; ++i) {
      unsigned n = tid * multiple + i;
      if (n < starting) {
        done[n].get() = 0;
        starts[n].release();
      }
    }
  }

  void decascade(int tid) {
    const unsigned multiple = 2;
    for (unsigned i = 1; i <= multiple; ++i) {
      unsigned n = tid * multiple + i;
      if (n < starting)
        while (!done[n].get()) { LL::asmPause(); }
    }
    done[tid].get() = 1;
  }


  void mlaunch(unsigned tid) {
    initThread(tid);
    try {
      while (true) {
        starts[tid].acquire();
        cascade(tid);
        work();
        decascade(tid);
      }
    } catch (const shutdown_ty&) {
    }
  }

public:
  ThreadPool_cpp11():
    ThreadPool(Galois::Runtime::LL::getMaxThreads()),
    starts(maxThreads), starting(maxThreads),
    work(), done(maxThreads)
  {   
    initThread(0, false);

    for (unsigned i = 1; i < maxThreads; ++i) {
      std::thread t(&ThreadPool_cpp11::mlaunch, this, i);
      threads.emplace_back(std::move(t));
    }
    decascade(0);
  }

  virtual ~ThreadPool_cpp11() {
    std::function<void(void)> f = []() { throw shutdown_ty(); };
    work = f;
    starting = starts.size();
    cascade(0);
    for(auto& t : threads)
      t.join();
  }

  virtual void runInternal(unsigned num, std::function<void(void)>& cmd) {
    // Sanitize num
    num = std::min(std::max(num,1U), maxThreads);
    starting = num;
    // Setup work
    work = cmd;
    //atomic ensures seq consistency for workBegin,End
    //launch threads
    cascade(0);
    // Do master thread work
    cmd();
    //wait for children
    decascade(0);
    // Clean up
    work = nullptr;
  }
};

} // end namespace

//! Implement the global threadpool
ThreadPool& Galois::Runtime::getSystemThreadPool() {
  static ThreadPool_cpp11 pool;
  return pool;
}
