#include "Galois/Timer.h"
#include "Galois/Galois.h"

#include <iostream>
#include <cstdlib>

int RandomNumber () { return (rand()%1000000); }

int main() {

  unsigned M = GaloisRuntime::LL::getMaxThreads();

  while (M) {
    
    Galois::setActiveThreads(M); //GaloisRuntime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";
    
    std::vector<unsigned> V(1024*1024*16);
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

    bool eq = std::equal(C.begin(), C.end(), V.begin());

    std::cout << "Galois: " << t.get()
	      << " STL: " << t2.get()
	      << " Equal: " << eq << "\n";
    
    if (!eq) {
      std::vector<unsigned> R = V;
      std::sort(R.begin(), R.end());
      if (!std::equal(C.begin(), C.end(), R.begin()))
	std::cout << "Cannot be made equal, sort mutated array\n";
      for (int x = 0; x < V.size() ; ++x) {
	std::cout << x << "\t" << V[x] << "\t" << C[x];
	if (V[x] != C[x]) std::cout << "\tDiff";
	if (V[x] < C[x]) std::cout << "\tLT";
	if (V[x] > C[x]) std::cout << "\tGT";
	std::cout << "\n";
      }
      return 1;
    }

    M >>= 1;
  }

  return 0;
}
