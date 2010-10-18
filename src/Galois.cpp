#include <cassert>
#include "Galois/Galois.h"

int GaloisRuntime::numThreads;

void Galois::setMaxThreads(int T)
{
  GaloisRuntime::numThreads = T;
}

__thread GaloisRuntime::SimpleRuntimeContext* GaloisRuntime::thread_cnx;

