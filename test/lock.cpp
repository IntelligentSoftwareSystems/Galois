#include "galois/Substrate/SimpleLock.h"

#include <cstdlib>

volatile int V;

int main(int argc, char** argv) {
  unsigned M = 1;
  if (argc > 1)
    M = atoi(argv[1]);
  if (!M)
    M = 1000000000;
  galois::substrate::SimpleLock L;
  for (unsigned x = 0; x < M; ++x) {
    V = 0;
    L.lock();
    V = 1;
    L.unlock();
    V = 2;
  }
  return 0;
}
