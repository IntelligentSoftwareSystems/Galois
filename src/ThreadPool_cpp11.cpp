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
#include "Galois/Runtime/ll/HWTopo.h"

#include "boost/utility.hpp"

#include <thread>
#include <condition_variable>

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

  virtual void threadWakeup(unsigned n) {
    starts[n].release();
  }

  virtual void threadWait(unsigned tid) {
    starts[tid].acquire();
  }

public:
  ThreadPool_cpp11():
    ThreadPool(Galois::Runtime::LL::getMaxThreads()),
    starts(maxThreads)
  {   
    for (unsigned i = 1; i < maxThreads; ++i) {
      std::thread t(&ThreadPool_cpp11::threadLoop, this, i);
      threads.emplace_back(std::move(t));
    }
    decascade(0);
  }

  virtual ~ThreadPool_cpp11() {
    destroyCommon();
    for(auto& t : threads)
      t.join();
  }
};

} // end namespace

//! Implement the global threadpool
ThreadPool& Galois::Runtime::getSystemThreadPool() {
  static ThreadPool_cpp11 pool;
  return pool;
}
