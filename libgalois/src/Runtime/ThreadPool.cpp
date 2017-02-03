/** Thread pool common implementation -*- C++ -*-
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/ErrorFeedBack.h"

#include <algorithm>
#include <iostream>

using namespace Galois::Runtime;

thread_local ThreadPool::local_state ThreadPool::tcb;

void ThreadPool::local_state::wakeup_remote(const std::function<void(void)>& w, unsigned num) {
  //wait for thread to be done
  wait_remote();
  work = w;
  state = num;
  if (!fastmode.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lg(m);
    cv.notify_one();
  }
}

void ThreadPool::local_state::detach_remote(const std::function<void(void)>& w) {
  //wait for thread to be done
  wait_remote();
  work = w;
  state = ~0;
  if (!fastmode.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lg(m);
    cv.notify_one();
  }
}

void ThreadPool::local_state::wait_local() {
  work = nullptr;
  state.store(0, std::memory_order_release);
  if (fastmode.load(std::memory_order_acquire)) {
    while(!state.load(std::memory_order_acquire)) { asmPause(); }
  } else {
    std::unique_lock<std::mutex> lg(m);
    cv.wait(lg, [=] { return state.load(); });
  }
}

//waits until thread is idle
void ThreadPool::local_state::wait_remote() {
  while (state.load(std::memory_order_acquire)) { asmPause(); }
}

bool ThreadPool::local_state::try_wait_remote() {
  return !state.load(std::memory_order_acquire);
}

ThreadPool::ThreadPool()
  : mi(getHWTopo().first),
    running(false)
{
  bool no_bind = false;
  bool no_bind_main = false;
  TRACE("Init Begin ThreadPool");
  sizeSignal = mi.maxThreads;
  signals.reset(new std::atomic<local_state*>[sizeSignal]);
  initThread(0, !no_bind && !no_bind_main);

  for (unsigned i = 1; i < mi.maxThreads; ++i) {
    signals[i].store(nullptr);
    threads.emplace_back(&ThreadPool::threadLoop, this, i, !no_bind);
  }
  TRACE("Init End ThreadPool");
}

ThreadPool::~ThreadPool() {
  TRACE("Destroy Begin ThreadPool");
  for (unsigned i = 0; i < sizeSignal; ++i)
    signals[i].load()->detach_remote([]() {});
  TRACE("Destroy Mid ThreadPool");
  for(auto& t : threads)
    t.join();
  TRACE("Destroy End ThreadPool");
}

void ThreadPool::burnPower(unsigned num) {
  run(num, [this]() -> void { 
      this->signals[ThreadPool::getTID()].load()->fastmode = true;
    });
//     masterFastmode = num;
//   }
}

void ThreadPool::beKind() {
  run(sizeSignal, [this]() {
      this->signals[ThreadPool::getTID()].load()->fastmode = false;
    });
}

void ThreadPool::initThread(unsigned tid, 
                            bool bindCore) {
  tcb.topo = getHWTopo().second[tid];
  tcb.work = nullptr;
  tcb.fastmode = false;
  tcb.state = 0;
  
  if (bindCore)
    bindThreadSelf(tcb.topo.osContext);

  initPTS(mi.maxThreads);

  //done with initializing
  signals[tid] = &tcb;
}

void ThreadPool::threadLoop(unsigned tid, bool bindCore) {
  initThread(tid, bindCore);

  do {
    tcb.wait_local();
    if (tcb.state != ~0)
      cascade(tcb.work, tcb.state);
    try {
      TRACE("TP D tl ", getTID()," ",  tcb.work.target_type().name());
      tcb.work();
    } catch (const std::bad_function_call& bfc) {
      TRACE("TP E tl ", getTID(), " ", tcb.work.target_type().name());
      gDie("Thread Pool bad function call: ", bfc.what());
    } catch (const std::exception &exc) {
      // catch anything thrown within try block that derives from std::exception
      gDie("Thread Pool caught: ", exc.what());
    }
    catch (...) {
      abort();
    }
    if (tcb.state != ~0)
      decascade(tcb.state);
  } while (tcb.state != ~0);
}


void ThreadPool::decascade(unsigned num) {
    auto tid = getTID();

    for (unsigned i = tid * 3 + 1; i < tid * 3 + 4 && i < num; ++i) {
      while (!signals[i].load()) {} // wait for threads to be loaded.  Fixes DSO/TLS problems
      signals[i].load(std::memory_order_relaxed)->wait_remote();
    }

  // //nothing to wake up
  // if (me.wbegin != me.wend) {
  //   auto midpoint = me.wbegin + (1 + me.wend - me.wbegin)/2;
  //   auto& c1done = signals[me.wbegin]->done;
  //   while (!c1done) { asmPause(); }
  //   if (midpoint < me.wend) {
  //     auto& c2done = signals[midpoint]->done;
  //     while (!c2done) { asmPause(); }
  //   }
  // }
  // me.done = 1;
}

void ThreadPool::cascade(const std::function<void(void)>& fn, unsigned num) {
  // assert(me.wbegin <= me.wend);

  auto tid = getTID();

  TRACE("TP Cascade ", tid, " ", num);
  
  for (unsigned i = tid * 3 + 1; i < tid * 3 + 4 && i < num; ++i) {
    TRACE("TP Cascade2 ", tid, " ", i);
    signals[i].load(std::memory_order_relaxed)->wakeup_remote(fn, num);
  }
}

void ThreadPool::run(unsigned num, const std::function<void(void)>& fn) {
  TRACE("Run Begin ThreadPool ", num);
  
  //sanitize num
  assert(num <= getMaxThreads() && "too many threads");
  assert(!running && "recursive thread pool execution not supported");

  //seq write to starting should make work safe
  running = true;
  num = std::min(std::max(1U,num), sizeSignal);

  //launch threads
  cascade(fn, num);
  // Do master thread work
  fn();
  //wait for children
  decascade(num);
  // Clean up
  running = false;
  TRACE("Run End ThreadPool ", num);
}

void ThreadPool::runDedicated(const std::function<void(void)>&& f) {
  if (threads.empty()) {
    assert(0 && "out of threads for dedicated function");
    abort();
  }
  
  if (running) {
    assert(0 && "can't start dedicated thread durring parallel section");
    abort();
  }

  
  signals[sizeSignal-1].load(std::memory_order_relaxed)->detach_remote(f);
  signals[sizeSignal-1].store(0);
  --sizeSignal;
  threads.rbegin()->detach();
  threads.pop_back();

  //FIXME: Galois::setActiveThreads(Galois::getActiveThreads());
}

ThreadPool& ThreadPool::getThreadPool() {
  static ThreadPool p;
  return p;
}
