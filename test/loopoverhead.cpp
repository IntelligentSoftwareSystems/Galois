#include "Galois/Timer.h"
#include "Galois/Galois.h"

#include <iostream>
#include <cstdlib>

int RandomNumber () { return (rand()%1000000); }
const unsigned iter = 16*1024;

struct emp {
  template<typename T>
  void operator()(const T& t) { Galois::Runtime::LL::compilerBarrier(); }
  template<typename T, typename C>
  void operator()(const T& t, const C& c) { Galois::Runtime::LL::compilerBarrier(); }
};

void t_inline() {
  std::vector<unsigned> V(1024);
  //unsigned M = Galois::Runtime::LL::getMaxThreads();

  std::cout << "inline:\nIterxSize\n";

  std::cout << "Using " << 1 << " threads\n";
   
  Galois::Timer t;
  t.start();
  emp e;
  for (unsigned x = 0; x < iter; ++x)
    for (auto i : V)
      e(i);
  t.stop();
  
  std::cout << "Inline(" << iter << "x" << V.size() << "): " << t.get() << "\n";
}

void t_stl() {
  std::vector<unsigned> V(1024);
  //unsigned M = Galois::Runtime::LL::getMaxThreads();

  std::cout << "stl:\nIterxSize\n";

  std::cout << "Using " << 1 << " threads\n";
   
  Galois::Timer t;
  t.start();
  for (unsigned x = 0; x < iter; ++x)
    std::for_each(V.begin(), V.end(), emp());
  t.stop();
  
  std::cout << "STL(" << iter << "x" << V.size() << "): " << t.get() << "\n";
}

void t_doall(bool burn) {

  std::vector<unsigned> V(1024);
  unsigned M = Galois::Runtime::LL::getMaxThreads();

  std::cout << "doall:\nIterxSize\n";

  while (M) {
    Galois::setActiveThreads(M); //Galois::Runtime::LL::getMaxThreads());
    if (burn)
      Galois::Runtime::getSystemThreadPool().burnPower(M);
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

void t_foreach(bool burn) {

  std::vector<unsigned> V(1024);
  unsigned M = Galois::Runtime::LL::getMaxThreads();

  std::cout << "foreach:\nIterxSize\n";

  while (M) {
    
    Galois::setActiveThreads(M); //Galois::Runtime::LL::getMaxThreads());
    if (burn)
      Galois::Runtime::getSystemThreadPool().burnPower(M);

    std::cout << "Using " << M << " threads\n";
   
    Galois::Timer t;
    t.start();
    for (unsigned x = 0; x < iter; ++x)
      Galois::for_each(V.begin(), V.end(), emp(), Galois::wl<Galois::WorkList::StableIterator<std::vector<unsigned>::iterator>>());
    t.stop();

    std::cout << "Galois(" << iter << "x" << V.size() << "): " << t.get() << "\n";

    M >>= 1;
  }
}

int main() {
  t_inline();
  t_stl();
  t_doall(false);
  t_foreach(false);
  t_doall(true);
  t_foreach(true);
  return 0;
}
