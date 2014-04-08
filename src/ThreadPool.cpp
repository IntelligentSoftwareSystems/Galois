/** Thread pool common implementation -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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

// Forward declare this to avoid including PerThreadStorage.
// We avoid this to stress that the thread Pool MUST NOT depend on PTS.
namespace Galois {
namespace Runtime {
extern void initPTS();
}
}


using namespace Galois::Runtime;

ThreadPool::ThreadPool(unsigned m): maxThreads(m), starting(m), done(m) {
  initThread(0);
}

ThreadPool::~ThreadPool() { }

void ThreadPool::destroyCommon() {
  std::function<void(void)> f = []() { throw shutdown_ty(); };
  work = f;
  starting = maxThreads;
  cascade(0);
}

void ThreadPool::initThread(unsigned tid) {
  // Initialize TID
  LL::initTID(tid);
  initPTS();
  if (!LL::EnvCheck("GALOIS_DO_NOT_BIND_THREADS"))
    if (tid != 0 || !LL::EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD"))
      LL::bindThreadToProcessor(tid);
}

void ThreadPool::threadLoop(unsigned tid) {
  initThread(tid);
  decascade(tid);
  try {
    while (true) {
      threadWait(tid);
      cascade(tid);
      work();
      decascade(tid);
    }
  } catch (const shutdown_ty&) {
  }
}

void ThreadPool::decascade(int tid) {
  const unsigned multiple = 2;
  for (unsigned i = 1; i <= multiple; ++i) {
    unsigned n = tid * multiple + i;
    if (n < starting)
      while (!done[n].get()) { LL::asmPause(); }
  }
  done[tid].get() = 1;
}

void ThreadPool::cascade(int tid) {
  const unsigned multiple = 2;
  for (unsigned i = 1; i <= multiple; ++i) {
    unsigned n = tid * multiple + i;
    if (n < starting) {
      done[n].get() = 0;
      threadWakeup(n);
    }
  }
}
