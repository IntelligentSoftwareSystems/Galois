/** Common thread pool implementations -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Galois.h"

using namespace GaloisRuntime;

typedef boost::intrusive::list<ThreadAware, boost::intrusive::constant_time_size<false>, boost::intrusive::base_hook<HIDDEN::ThreadAwareHook> > ListTy;

//pointer because of undefined construct ordering
//should be a pod type so is always constructed
static PtrLock<ListTy*,true> allObjects;

static void createAllObjects(void) {
  if (!allObjects.getValue())
    allObjects.setValue(new ListTy());
}

//! This, once initialized by a thread, stores an dense index/label for that thread
__thread unsigned int ThreadPool::LocalThreadID = ~0;

//! This stores the next thread id
int ThreadPool::nextThreadID = 0;

ThreadAware::ThreadAware() {
  allObjects.lock();
  createAllObjects();
  allObjects.getValue()->push_front(*this);
  allObjects.unlock();
}

ThreadAware::~ThreadAware() {
  allObjects.lock();
  allObjects.getValue()->erase(allObjects.getValue()->iterator_to(*this));
  allObjects.unlock();
}

void ThreadAware::NotifyOfChange(bool starting) {
  allObjects.lock();
  if (ListTy* LV = allObjects.getValue())
    for (ListTy::iterator ii = LV->begin(), ee = LV->end(); ii != ee; ++ii)
      ii->ThreadChange(starting);
  allObjects.unlock();
}

void ThreadPool::NotifyAware(bool starting) {
  ThreadAware::NotifyOfChange(starting);
}

unsigned int ThreadPool::getMyID() {
  unsigned int retval = LocalThreadID;
  if (retval == 0)
    return 0;
  if (retval == ~(unsigned int)0) {
    retval = __sync_add_and_fetch(&nextThreadID, 1);
    LocalThreadID = retval;
  }
  return retval;
}

initMainThread::initMainThread() {
  ThreadPool::LocalThreadID = 0;
  //  GaloisRuntime::getSystemThreadPolicy();
}

static initMainThread mainThreadIDSetter;

unsigned int Galois::setMaxThreads(unsigned int num) {
  return GaloisRuntime::getSystemThreadPool().setActiveThreads(num);
}
