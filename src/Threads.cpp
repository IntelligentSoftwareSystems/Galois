#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Galois.h"

using namespace GaloisRuntime;

typedef boost::intrusive::list<ThreadAware, boost::intrusive::constant_time_size<false>, boost::intrusive::base_hook<HIDDEN::ThreadAwareHook> > ListTy;

static ListTy allObjects;
static SimpleLock<int, true> allObjectsLock;

//This, once initialized by a thread, stores an dense index/label for that thread
__thread int ThreadPool::LocalThreadID = -1;

//This stores the next thread id
int ThreadPool::nextThreadID = 0;

ThreadAware::ThreadAware() {
  allObjectsLock.lock();
  allObjects.push_front(*this);
  allObjectsLock.unlock();
}

ThreadAware::~ThreadAware() {
  allObjectsLock.lock();
  allObjects.erase(allObjects.iterator_to(*this));
  allObjectsLock.unlock();
}

void ThreadAware::NotifyOfChange(bool starting) {
  allObjectsLock.lock();
  for (ListTy::iterator ii = allObjects.begin(), ee = allObjects.end(); ii != ee; ++ii) {
    ii->ThreadChange(starting);
  }
  allObjectsLock.unlock();
}

void ThreadPool::NotifyAware(bool starting) {
  ThreadAware::NotifyOfChange(starting);
}

unsigned int ThreadPool::getMyID() {
  unsigned int retval = LocalThreadID;
  if (retval == 0)
    return 0;
  if (retval == -1) {
    retval = __sync_add_and_fetch(&nextThreadID, 1);
    LocalThreadID = retval;
  }
  return retval;
}

initMainThread::initMainThread() {
  ThreadPool::LocalThreadID = 0;
}

static initMainThread mainThreadIDSetter;

void Galois::setMaxThreads(unsigned int num) {
  GaloisRuntime::getSystemThreadPool().setMaxThreads(num);
}
