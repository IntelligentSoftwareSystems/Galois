/** Thread pool common implementation -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/EnvCheck.h"
#include "Galois/Substrate/HWTopo.h"
#include "Galois/Substrate/gio.h"

#include <cstdlib>

// Forward declare this to avoid including PerThreadStorage.
// We avoid this to stress that the thread Pool MUST NOT depend on PTS.
namespace Galois {
namespace Runtime {

extern void initPTS();

}
}


using namespace Galois::Substrate;

ThreadPool::ThreadPool(): mi(getHWTopo()->getMachineInfo()), starting(mi.maxThreads), masterFastmode(false), signals(nullptr), running(false) {
  initThread();
}

ThreadPool::~ThreadPool() { }

void ThreadPool::destroyCommon() {
  beKind(); // reset fastmode
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

//inefficient append
template<typename T>
static void atomic_append(std::atomic<T*>& headptr, T* newnode) {
  T* n = nullptr;
  if (!headptr.compare_exchange_strong(n, newnode))
    atomic_append(headptr.load()->next, newnode);
}

//find id
template<typename T> 
static unsigned findID(std::atomic<T*>& headptr, T* node, unsigned off) {
  T* n = headptr.load();
  assert(n);
  if (n == node)
    return off;
  else
    return findID(n->next, node, off+1);
}

template<typename T>
static T* getNth(std::atomic<T*>& headptr, unsigned off) {
  T* n = headptr.load();
  if (!off)
    return n;
  else
    return getNth(n->next, off - 1);
}

void ThreadPool::initThread() {
  atomic_append(signals, &my_box);
  //my_box.id = findID(signals, &my_box, 0);

  // Initialize 
  Runtime::initPTS();

  if (!EnvCheck("GALOIS_DO_NOT_BIND_THREADS"))
    if (topo.tid != 0 || !EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD"))
      getHWTopo()->bindThreadToProcessor(topo.tid);
}

void ThreadPool::threadLoop() {
  initThread();
  decascade();
  bool fastmode = false;
  do {
    if (fastmode) {
      while (!signals[topo.tid].fastRelease.load(std::memory_order_relaxed)) {
        asmPause();
      }
      signals[topo.tid].fastRelease = 0;
    } else {
      threadWait(topo.tid);
    }
    cascade(fastmode);
    
    try {
      work();
    } catch (const shutdown_ty&) {
      return;
    } catch (const fastmode_ty& fm) {
      fastmode = fm.mode;
    } catch (...) {
      abort();
    }
    decascade();
  } while (true);
}


void ThreadPool::decascade() {
  assert(topo.tid == 0 || my_box.done == 0);
  const unsigned multiple = 3;
  unsigned limit = starting;
  for (unsigned i = 1; i <= multiple; ++i) {
    unsigned n = topo.tid * multiple + i;
    if (n < limit) {
      auto& done_flag = getNth(signals, n)->done;
      while (!done_flag) { asmPause(); }
    }
  }
  my_box.done = 1;
}

void ThreadPool::cascade(bool fastmode) {
  unsigned limit = starting;
  const unsigned multiple = 3;
  for (unsigned i = 1; i <= multiple; ++i) {
    unsigned n = topo.tid * multiple + i;
    if (n < limit) {
      auto nid = getNth(signals, n);
      nid->done = 0;
      if (fastmode)
        nid->fastRelease = 1;
      else
        threadWakeup(n);
    }
  }
}

void ThreadPool::runInternal(unsigned num) {
  //sanitize num
  //seq write to starting should make work safe
  GALOIS_ASSERT(!running, "recursive thread pool execution not supported");
  running = true;
  num = std::min(std::max(1U,num), mi.maxThreads);
  starting = num;
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
}
