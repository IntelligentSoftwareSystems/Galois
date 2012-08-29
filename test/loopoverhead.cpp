#include "Galois/Timer.h"
#include "Galois/Galois.h"

#include <iostream>
#include <cstdlib>

int RandomNumber () { return (rand()%1000000); }

struct emp {
  template<typename T>
  void operator()(const T& t) {
  }
};

int main() {

  std::vector<unsigned> V(1024);
  unsigned M = GaloisRuntime::LL::getMaxThreads();

  const unsigned iter = 16*1024;

  std::cout << "IterxSize\n";

  while (M) {
    
    Galois::setActiveThreads(M); //GaloisRuntime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";
   
    Galois::Timer t;
    t.start();
    for (unsigned x = 0; x < iter; ++x)
      Galois::do_all(V.begin(), V.end(), emp());
    t.stop();

    std::cout << "Galois(" << iter << "x" << V.size() << "): " << t.get() << "\n";

    M >>= 1;
  }

  return 0;
}
