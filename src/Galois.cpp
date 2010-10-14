#include <cassert>
#include "Galois/Galois.h"

int GaloisRuntime::numThreads;

static __thread GaloisRuntime::SimpleRuntimeContext* thread_cnx;

void Galois::setMaxThreads(int T)
{
  GaloisRuntime::numThreads = T;
}


GaloisRuntime::SimpleRuntimeContext* GaloisRuntime::getThreadContext() {
  return thread_cnx;
}

void GaloisRuntime::setThreadContext(GaloisRuntime::SimpleRuntimeContext* n) {
  thread_cnx = n;
}

void GaloisRuntime::acquire(Lockable& L) {
  acquire(&L);
}

void GaloisRuntime::acquire(Lockable* C) {
  SimpleRuntimeContext* cnx = getThreadContext();
  if (cnx)
    cnx->acquire(C);
}

