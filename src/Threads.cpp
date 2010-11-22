#include "Galois/Runtime/Threads.h"

#include "Support/ThreadSafe/simple_lock.h"

using namespace GaloisRuntime;

typedef boost::intrusive::list<ThreadAware, boost::intrusive::constant_time_size<false>, boost::intrusive::base_hook<HIDDEN::ThreadAwareHook> > ListTy;

static ListTy allObjects;
static threadsafe::simpleLock<int, true> allObjectsLock;

static int numThreads = 0;

//This, once initialized by a thread, stores an dense index/label for that thread
static __thread int LocalThreadID = -1;

//This stores the next thread id
static int nextThreadID = 0;

//This stores the minimum below which we don't consider an ID valid anymore
static int minThreadID = 0;

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

void ThreadAware::NotifyOfChange(int num) {
  allObjectsLock.lock();
  if (numThreads != num) {
    numThreads = num;
    for (ListTy::iterator ii = allObjects.begin(), ee = allObjects.end(); ii != ee; ++ii) {
      ii->ThreadChange(num);
    }
  }
  allObjectsLock.unlock();
}

void ThreadAware::init() {
  if (numThreads)
    ThreadChange(numThreads);
}

void ThreadPool::NotifyAware(int n) {
  ThreadAware::NotifyOfChange(n);
}

void ThreadPool::ResetThreadNumbers() {
  minThreadID = nextThreadID;
}

int ThreadPool::getMyID() {
  int retval = LocalThreadID;
  if (retval == 0)
    return 0;
  if (retval == -1) {
    retval = __sync_fetch_and_add(&nextThreadID, 1);
    LocalThreadID = retval;
  }
  return retval;
}

namespace {
struct initMainThread {
  initMainThread() {
    LocalThreadID = 0;
  }
};
}

initMainThread mainThreadIDSetter;

  
