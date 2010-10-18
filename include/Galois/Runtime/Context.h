// simple galois context and contention manager -*- C++ -*-

#ifndef _GALOIS_RUNTIME_CONTEXT_H
#define _GALOIS_RUNTIME_CONTEXT_H

#include "Support/ThreadSafe/simple_lock.h"
#include <vector>

namespace GaloisRuntime {

  typedef threadsafe::ptrLock Lockable;

  class SimpleRuntimeContext {

    std::vector<Lockable*> locks;

    void rollback() {
      throw -1;
    }
  public:
    void acquire(Lockable& L) {
      acquire(&L);
    }
    void acquire(Lockable* C) {
      if (C->getValue() != this) {
	bool suc = C->try_lock(this);
	if (suc) {
	  locks.push_back(C);
	} else {
	  rollback();
	}
      }
    }

    SimpleRuntimeContext() {}
    ~SimpleRuntimeContext() {
      for (std::vector<Lockable*>::iterator ii = locks.begin(), ee = locks.end(); ii != ee; ++ii)
	(*ii)->unlock();
    }
  };

  extern __thread SimpleRuntimeContext* thread_cnx;

  static SimpleRuntimeContext* getThreadContext() {
    return thread_cnx;
  }

  static void setThreadContext(SimpleRuntimeContext* n) {
    thread_cnx = n;
  }

  static void acquire(Lockable* C) {
    SimpleRuntimeContext* cnx = getThreadContext();
    if (cnx)
      cnx->acquire(C);
  }

  static void acquire(Lockable& L) {
    acquire(&L);
  }

}

#endif
