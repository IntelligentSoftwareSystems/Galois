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
#include "Galois/Runtime/CompilerSpecific.h"
#include "Galois/Runtime/ErrorFeedBack.h"

#include <algorithm>
#include <iostream>

using namespace Galois::Runtime;

thread_local ThreadPool::per_signal ThreadPool::my_box;


ThreadPool::ThreadPool(bool _no_bind, bool _no_bind_main) 
  : no_bind(_no_bind),
    no_bind_main(_no_bind_main),
    mi(getHWTopo().first),
    reserved(0),
    masterFastmode(false), 
    running(false)
{
  TRACE("Init Begin ThreadPool");
  signals.resize(mi.maxThreads);
  initThread(0);

  for (unsigned i = 1; i < mi.maxThreads; ++i) {
    std::thread t(&ThreadPool::threadLoop, this, i);
    threads.emplace_back(std::move(t));
  }

  //we don't want signals to have to contain atomics, since they are set once
  while (std::any_of(signals.begin(), signals.end(), [](per_signal* p) { return !p || !p->done; })) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }
  TRACE("Init End ThreadPool");
}

ThreadPool::~ThreadPool() {
  TRACE("Destroy Begin ThreadPool");
  destroyCommon();
  for(auto& t : threads)
    t.join();
  TRACE("Destroy End ThreadPool");
}

void ThreadPool::destroyCommon() {
  run(mi.maxThreads, []() { throw shutdown_ty(); });
}

void ThreadPool::burnPower(unsigned num) {
  //changing number of threads?  just do a reset
  if (masterFastmode && masterFastmode != num)
    beKind();
  if (!masterFastmode) {
    run(num, []() { throw fastmode_ty{true}; });
    masterFastmode = num;
  }
}

void ThreadPool::beKind() {
  if (masterFastmode) {
    run(masterFastmode, []() { throw fastmode_ty{false}; });
    masterFastmode = 0;
  }
}

void ThreadPool::initThread(unsigned tid) {
  signals[tid] = &my_box;
  my_box.topo = getHWTopo().second[tid];

  if (!no_bind)
    if (my_box.topo.tid != 0 || !no_bind_main)
      bindThreadSelf(my_box.topo.osContext);
  my_box.done = true;
}

void ThreadPool::threadLoop(unsigned tid) {
  initThread(tid);
  bool fastmode = false;
  auto& me = my_box;
  do {
    me.wait(fastmode);
    cascade(fastmode);
    std::atomic_thread_fence(std::memory_order_acquire);
    try {
      work();
    } catch (const shutdown_ty&) {
      return;
    } catch (const fastmode_ty& fm) {
      fastmode = fm.mode;
    } catch (const dedicated_ty& dt) {
      me.done = 1;
      dt.fn();
      return;
    } catch (const std::bad_function_call& bfc) {
      gDie("Thread Pool bad function call: ", bfc.what());
    } catch (const std::exception &exc) {
      // catch anything thrown within try block that derives from std::exception
      gDie("Thread Pool caught: ", exc.what());
    }
    catch (...) {
      abort();
    }
    decascade();
  } while (true);
}


void ThreadPool::decascade() {
  auto& me = my_box;
  //nothing to wake up
  if (me.wbegin != me.wend) {
    auto midpoint = me.wbegin + (1 + me.wend - me.wbegin)/2;
    auto& c1done = signals[me.wbegin]->done;
    while (!c1done) { asmPause(); }
    if (midpoint < me.wend) {
      auto& c2done = signals[midpoint]->done;
      while (!c2done) { asmPause(); }
    }
  }
  me.done = 1;
}

void ThreadPool::cascade(bool fastmode) {
  auto& me = my_box;
  assert(me.wbegin <= me.wend);

  //nothing to wake up
  if (me.wbegin == me.wend)
    return;

  auto midpoint = me.wbegin + (1 + me.wend - me.wbegin)/2;
  // static std::mutex m;
  // {
  //   std::lock_guard<std::mutex> lg(m);
  //   std::cout << getTID() << "\t" << me.wbegin << ": (" << me.wbegin + 1 << " " << midpoint << ")\t" << midpoint << ": (" << midpoint+1 << " " << me.wend << ")\n";
  // }

  auto child1 = signals[me.wbegin];
  child1->wbegin = me.wbegin+1;
  child1->wend = midpoint;
  child1->wakeup(fastmode);
  
  if (midpoint < me.wend) {
    auto child2 = signals[midpoint];
    child2->wbegin = midpoint + 1;
    child2->wend = me.wend;
    child2->wakeup(fastmode);
  }
}

void ThreadPool::run(unsigned num, std::function<void(void)>&& fn) {
  TRACE("Run Begin ThreadPool ", num);
  work = std::ref(fn);
  std::atomic_thread_fence(std::memory_order_release);

  //sanitize num
  assert(num <= getMaxThreads() && "too many threads");
  assert(!running && "recursive thread pool execution not supported");

  //seq write to starting should make work safe
  running = true;
  num = std::min(std::max(1U,num), mi.maxThreads - reserved);
  //my_box is tid 0
  auto& me = my_box;
  me.wbegin = 1;
  me.wend = num;

  assert(!masterFastmode || masterFastmode == num);
  //launch threads
  cascade(masterFastmode);
  // Do master thread work
  try {
    work();
  } catch (const shutdown_ty&) {
    return;
  } catch (const fastmode_ty& fm) {
  }
  //wait for children
  decascade();
  // Clean up
  work = nullptr;
  running = false;
  TRACE("Run End ThreadPool ", num);
}

void ThreadPool::runDedicated(std::function<void(void)>& f) {
  assert(!running && "can't start dedicated thread durring parallel section");
  ++reserved;
  assert(reserved < mi.maxThreads && "Too many dedicated threads");
  work = [&f] () { throw dedicated_ty {f}; };
  auto child = signals[mi.maxThreads - reserved];
  child->wbegin = 0;
  child->wend = 0;
  child->done = 0;
  child->wakeup(masterFastmode);
  while (!child->done) { asmPause(); }
  work = nullptr;
  //FIXME: Galois::setActiveThreads(Galois::getActiveThreads());
}


void ThreadPool::per_signal::wakeup(bool fastmode) {
  if (fastmode) {
    done = 0;
    fastRelease = 1;
  } else {
    std::lock_guard<std::mutex> lg(m);
    done = 0;
    cv.notify_one();
    //start.release();
  }
}

void ThreadPool::per_signal::wait(bool fastmode) {
  if (fastmode) {
    while(!fastRelease.load(std::memory_order_relaxed)) { asmPause(); }
    fastRelease = 0;
  } else {
    std::unique_lock<std::mutex> lg(m);
    cv.wait(lg, [=] { return !done; });
    //start.acquire();
  }
}


ThreadPool& ThreadPool::getThreadPool(bool nb, bool nbm) {
  static ThreadPool p(nb,nbm);
  return p;
}
