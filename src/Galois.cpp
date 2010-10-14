#include <cassert>
#include "Galois/Galois.h"

int GaloisRuntime::numThreads;

__thread GaloisRuntime::GaloisWorkContextCausious* GaloisRuntime::thread_cnx;

void Galois::setMaxThreads(int T)
{
  GaloisRuntime::numThreads = T;
}
