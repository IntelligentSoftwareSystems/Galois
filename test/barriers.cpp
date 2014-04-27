#include "Galois/Timer.h"
#include "Galois/Galois.h"
#include "Galois/Runtime/Barrier.h"

#include <iostream>
#include <cstdlib>
#include <unistd.h>

const unsigned iter = 16*1024;
char bname[100];

struct emp {
  Galois::Runtime::Barrier& b;

  void go() {
    for (int i = 0; i < iter; ++i)
      b.wait();
  }

  template<typename T>
  void operator()(const T& t) { 
    go();
  }

  template<typename T, typename C>
  void operator()(const T& t, const C& c) {
    go();
  }
};

void test(Galois::Runtime::Barrier& b) {

  unsigned M = Galois::Runtime::LL::getMaxThreads();
  while (M) {   
    Galois::setActiveThreads(M); //Galois::Runtime::LL::getMaxThreads());
    b.reinit(M);
    Galois::Timer t;
    t.start();
    Galois::on_each(emp{b});
    t.stop();
    std::cout << bname << "," << b.name() << "," << M << "," << t.get() << "\n";
    M -= 1;
  }
}

int main() {
  gethostname(bname, sizeof(bname));
  using namespace Galois::Runtime::benchmarking;
  test(getPthreadBarrier());
  test(getCountingBarrier());
  test(getMCSBarrier());
  test(getTopoBarrier());
  return 0;
}
