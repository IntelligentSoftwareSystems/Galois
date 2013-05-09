#include "Galois/Runtime/ll/SimpleLock.h"

volatile int V;

int main(int argc, char** argv) {
  Galois::Runtime::LL::SimpleLock<true> L;
  for (unsigned x = 0; x < 1000000000; ++x) {
    V = 0;
    L.lock();
    V = 1;
    L.unlock();
    V = 2;
  }
  return 0;
}
