#include "Galois/Timer.h"
#include "Galois/Galois.h"

#include <iostream>
#include <cstdlib>

int RandomNumber () { return (rand()%1000000); }
const unsigned iter = 16*1024;

struct emp {
  template<typename T>
  void operator()(const T& t) { GaloisRuntime::LL::compilerBarrier(); }
  template<typename T, typename C>
  void operator()(const T& t, const C& c) { GaloisRuntime::LL::compilerBarrier(); }
};

void t_stl() {
  std::vector<unsigned> V(1024);
  //unsigned M = GaloisRuntime::LL::getMaxThreads();

  std::cout << "stl:\nIterxSize\n";

  std::cout << "Using " << 1 << " threads\n";
   
  Galois::Timer t;
  t.start();
  for (unsigned x = 0; x < iter; ++x)
    std::for_each(V.begin(), V.end(), emp());
  t.stop();
  
  std::cout << "STL(" << iter << "x" << V.size() << "): " << t.get() << "\n";
}

void t_doall() {

  std::vector<unsigned> V(1024);
  unsigned M = GaloisRuntime::LL::getMaxThreads();

  std::cout << "doall:\nIterxSize\n";

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
}

void t_foreach() {

  std::vector<unsigned> V(1024);
  unsigned M = GaloisRuntime::LL::getMaxThreads();

  std::cout << "foreach:\nIterxSize\n";

  while (M) {
    
    Galois::setActiveThreads(M); //GaloisRuntime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";
   
    Galois::Timer t;
    t.start();
    for (unsigned x = 0; x < iter; ++x)
      Galois::for_each(V.begin(), V.end(), emp());
    t.stop();

    std::cout << "Galois(" << iter << "x" << V.size() << "): " << t.get() << "\n";

    M >>= 1;
  }
}

int main() {
  t_stl();
  t_doall();
  t_foreach();
  return 0;
}
