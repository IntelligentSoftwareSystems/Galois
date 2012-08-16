#include "Galois/Timer.h"
#include "Galois/Galois.h"

#include <iostream>
#include <cstdlib>

int RandomNumber () { return (rand()%10000); }

int main() {

  unsigned M = GaloisRuntime::LL::getMaxThreads();

  while (M) {
    
    Galois::setActiveThreads(M); //GaloisRuntime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";
    
    std::vector<unsigned> V(1024*1024);
    std::generate (V.begin(), V.end(), RandomNumber);
    std::vector<unsigned> C = V;

    Galois::Timer t;
    t.start();
    Galois::sort(V.begin(), V.end());
    t.stop();
    
    Galois::Timer t2;
    t2.start();
    std::sort(C.begin(), C.end());
    t2.stop();

    std::cout << t.get() << " (" << V.size() << ") "
	      << t2.get() << " (" << C.size() << ") "
	      << std::equal(C.begin(), C.end(), V.begin()) << "\n";
    M >>= 1;
  }

  return 0;
}
