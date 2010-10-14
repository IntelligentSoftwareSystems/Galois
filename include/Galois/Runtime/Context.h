// simple galois context and contention manager -*- C++ -*-

#ifndef _GALOIS_RUNTIME_CONTEXT_H
#define _GALOIS_RUNTIME_CONTEXT_H

#include "Support/ThreadSafe/simple_lock.h"
#include <set>

namespace GaloisRuntime {
  class Lockable {
    threadsafe::simpleLock L;
  public:
    bool try_lock() {
      return L.try_write_lock();
    }
    void unlock() {
      L.write_unlock();
    }
  };

  class SimpleRuntimeContext {

    std::set<Lockable*> locks;

    void rollback() {
      throw -1;
    }
  public:
    void acquire(Lockable& L) {
      acquire(&L);
    }
    void acquire(Lockable* C) {
      if (!locks.count(C)) {
	bool suc = C->try_lock();
	if (suc) {
	  locks.insert(C);
	} else {
	  rollback();
	}
      }
    }

    SimpleRuntimeContext() {}
    ~SimpleRuntimeContext() {
      for (std::set<Lockable*>::iterator ii = locks.begin(), ee = locks.end(); ii != ee; ++ii)
	(*ii)->unlock();
    }
  };

  SimpleRuntimeContext* getThreadContext();
  void setThreadContext(SimpleRuntimeContext* n);
  void acquire(Lockable&);
  void acquire(Lockable*);

}

#endif
