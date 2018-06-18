#include "Exp/Parallel.h"

#include "boost/utility.hpp"
#include <cassert>
#include <cstdio>

#ifdef GALOIS_USE_VTUNE
#include "ittnotify.h"
#endif

thread_local unsigned Exp::TID = 0;
unsigned Exp::nextID       = 0;

//! Generic check for pthread functions
void checkResults(int val) {
  if (val) {
    perror("PTHREAD: ");
    assert(0 && "PThread check");
    abort();
  }
}

void Exp::PthreadBarrier::checkResults(int val) {
  if (val) {
    perror("PTHREADS: ");
    assert(0 && "PThread check");
    abort();
  }
}

Exp::PthreadBarrier::PthreadBarrier() {
  // uninitialized barriers block a lot of threads to help with debugging
  int rc = pthread_barrier_init(&bar, 0, ~0);
  checkResults(rc);
}

Exp::PthreadBarrier::PthreadBarrier(unsigned int val) {
  int rc = pthread_barrier_init(&bar, 0, val);
  checkResults(rc);
}

Exp::PthreadBarrier::~PthreadBarrier() {
  int rc = pthread_barrier_destroy(&bar);
  checkResults(rc);
}

void Exp::PthreadBarrier::reinit(int val) {
  int rc = pthread_barrier_destroy(&bar);
  checkResults(rc);
  rc = pthread_barrier_init(&bar, 0, val);
  checkResults(rc);
}

void Exp::PthreadBarrier::wait() {
  int rc = pthread_barrier_wait(&bar);
  if (rc && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    checkResults(rc);
}

#if 1
namespace {

class Semaphore : private boost::noncopyable {
  pthread_mutex_t lock;
  pthread_cond_t cond;
  int val;

public:
// Explicit init/destroy because MakeDeterministic loops infinitely otherwise
#if 0
  explicit Semaphore(int v = 0): val(v) {
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&cond, NULL);
  }
  ~Semaphore() {
    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&lock);
  }
#else
  void init(int v = 0) {
    val = v;
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&cond, NULL);
  }

  void destroy() {
    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&lock);
  }
#endif
  void release(int n = 1) {
    pthread_mutex_lock(&lock);
    val += n;
    if (val > 0)
      pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&lock);
  }

  void acquire(int n = 1) {
    pthread_mutex_lock(&lock);
    while (val < n) {
      pthread_cond_wait(&cond, &lock);
    }
    val -= n;
    pthread_mutex_unlock(&lock);
  }
};

class ThreadPool_pthread {
  pthread_t* threads; // set of threads
  Semaphore* starts;  // signal to release threads to run
  Semaphore started;
  Semaphore idlock;
  unsigned maxThreads;
  volatile bool shutdown; // Set and start threads to have them exit
  volatile Exp::RunCommand* workBegin; // Begin iterator for work commands
  volatile Exp::RunCommand* workEnd;   // End iterator for work commands

  void initThread() {
    // we use a simple pthread or atomic to avoid depending on Galois
    // stuff too early in the initialization process
    idlock.acquire();
    Exp::getTID();
    idlock.release();
    started.release();
  }

  void cascade(int tid) {
    const unsigned multiple = 2;
    for (unsigned i = 1; i <= multiple; ++i) {
      unsigned n = tid * multiple + i;
      if (n < maxThreads)
        starts[n].release();
    }
  }

  void doWork(unsigned LocalThreadID) {
    cascade(LocalThreadID);
    Exp::RunCommand* workPtr  = (Exp::RunCommand*)workBegin;
    Exp::RunCommand* workEndL = (Exp::RunCommand*)workEnd;

    while (workPtr != workEndL) {
      (*workPtr)();
      ++workPtr;
    }
  }

  void launch(void) {
    unsigned LocalThreadID = Exp::getTID();
    while (!shutdown) {
      starts[LocalThreadID].acquire();
      doWork(LocalThreadID);
    }
  }

  static void* slaunch(void* V) {
    ThreadPool_pthread* TP = (ThreadPool_pthread*)V;
    TP->initThread();
    TP->launch();
    return 0;
  }

public:
  ThreadPool_pthread() : shutdown(false), workBegin(0), workEnd(0) {
    started.init(0);
    idlock.init(1);
    maxThreads = Exp::getNumThreads();
    initThread();

    starts  = new Semaphore[maxThreads];
    threads = new pthread_t[maxThreads];

    for (unsigned i = 0; i < maxThreads; ++i)
      starts[i].init(0);

    for (unsigned i = 1; i < maxThreads; ++i) {
      int rc = pthread_create(&threads[i], 0, &slaunch, this);
      checkResults(rc);
    }
    started.acquire(maxThreads);
  }

  ~ThreadPool_pthread() {
    shutdown  = true;
    workBegin = workEnd = 0;
    __sync_synchronize();
    for (unsigned i = 1; i < maxThreads; ++i)
      starts[i].release();
    for (unsigned i = 1; i < maxThreads; ++i) {
      int rc = pthread_join(threads[i], NULL);
      checkResults(rc);
    }
    // delete [] starts;
    for (unsigned i = 0; i < maxThreads; ++i)
      starts[i].destroy();
    idlock.destroy();
    started.destroy();
    delete[] threads;
  }

  void run(Exp::RunCommand* begin, Exp::RunCommand* end) {
    // setup work
    workBegin = begin;
    workEnd   = end;
    // ensure stores happen before children are spawned
    __sync_synchronize();
    // Do master thread work
    doWork(0);
    // clean up
    workBegin = workEnd = 0;
  }
};
} // namespace
#endif

unsigned Exp::getNumThreads() {
  char* p = getenv("GALOIS_NUM_THREADS");
  if (p) {
    int n = atoi(p);
    if (n > 0)
      return n;
  }
  return 1;
}

int Exp::getNumRounds() {
  char* p = getenv("EXP_NUM_ROUNDS");
  if (p) {
    int n = atoi(p);
    if (n > 0)
      return n;
  }
  return -1;
}

static bool started = 0;

void Exp::beginSampling() {
#ifdef GALOIS_USE_VTUNE
  if (!started) {
    __itt_resume();
    started = true;
  }
#endif
  char* p = getenv("GALOIS_EXIT_BEFORE_SAMPLING");
  if (p) {
    int n = atoi(p);
    exit(n);
  }
}

void Exp::endSampling() {
#ifdef GALOIS_USE_VTUNE
  __itt_pause();
  started = 0;
#endif
  char* p = getenv("GALOIS_EXIT_AFTER_SAMPLING");
  if (p) {
    int n = atoi(p);
    exit(n);
  }
}

#ifdef USE_SIMPLE_RUNTIME
static void* slaunch(void* ptr) {
  std::pair<Exp::RunCommand*, Exp::RunCommand*>* work =
      (std::pair<Exp::RunCommand*, Exp::RunCommand*>*)ptr;
  Exp::RunCommand* begin = work->first;
  Exp::RunCommand* end   = work->second;

  while (begin != end) {
    (*begin)();
    ++begin;
  }
  return NULL;
}
#endif

static ThreadPool_pthread* pool;

void Exp::do_all_init() {
#ifndef USE_SIMPLE_RUNTIME
  pool = new ThreadPool_pthread();
#endif
}

void Exp::do_all_finish() {
#ifndef USE_SIMPLE_RUNTIME
  delete pool;
#endif
}

void Exp::do_all_impl(RunCommand* begin, RunCommand* end) {
#ifdef USE_SIMPLE_RUNTIME
  unsigned maxThreads = Exp::getNumThreads();
  pthread_t* threads  = new pthread_t[maxThreads];

  std::pair<Exp::RunCommand*, Exp::RunCommand*> work(begin, end);

  for (unsigned i = 1; i < maxThreads; ++i) {
    int rc = pthread_create(&threads[i], 0, &slaunch, &work);
    checkResults(rc);
  }
  slaunch(&work);
  for (unsigned i = 1; i < maxThreads; ++i) {
    int rc = pthread_join(threads[i], NULL);
    checkResults(rc);
  }
#else
  //  static ThreadPool_pthread pool;
  pool->run(begin, end);
#endif
}
