/** std::thread thread pool implementation -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/CompilerSpecific.h"

#include <thread>
#include <algorithm>
#include <condition_variable>

namespace {

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

class ThreadPool_cpp11 : public Galois::Substrate::ThreadPool {
  std::vector<std::thread> threads;
  std::vector<Galois::Substrate::CacheLineStorage<Semaphore>> starts;  // Signal to release threads to run

  virtual void threadWakeup(unsigned n) {
    starts[n].get().release();
  }

  virtual void threadWait(unsigned tid) {
    starts[tid].get().acquire();
  }

public:
  ThreadPool_cpp11():
    ThreadPool(),
    starts(getMaxThreads())
  {   
    for (unsigned i = 1; i < getMaxThreads(); ++i) {
      std::thread t(&ThreadPool_cpp11::threadLoop, this, i);
      threads.emplace_back(std::move(t));
    }
    //we don't want signals to have to contain atomics, since they are set once
    while (std::any_of(signals.begin(), signals.end(), [](per_signal* p) { return !p || !p->done; })) {
      std::atomic_thread_fence(std::memory_order_seq_cst);
    }
  }

  virtual ~ThreadPool_cpp11() {
    destroyCommon();
    for(auto& t : threads)
      t.join();
  }
};

} // end namespace

//! Implement the global threadpool
Galois::Substrate::ThreadPool& Galois::Substrate::getThreadPool() {
  static ThreadPool_cpp11 pool;
  return pool;
}
