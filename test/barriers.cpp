#include "Galois/Timer.h"
#include "Galois/Galois.h"
#include "Galois/Runtime/Barrier.h"

#include <iostream>
#include <cstdlib>
#include <unistd.h>

unsigned iter = 1;
unsigned numThreads = 2;

char bname[100];

struct emp {
  Galois::Runtime::Barrier& b;

  void go() {
    for (unsigned i = 0; i < iter; ++i) {
      // if (Galois::Runtime::LL::getTID() == 0)
      //   usleep(100);
      b.wait();
    }
    // std::cout << Galois::Runtime::LL::getTID() << " ";
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
  unsigned M = numThreads;
  if (M > 16) M /= 2;
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

int main(int argc, char** argv) {
  if (argc > 1)
    iter = atoi(argv[1]);
  if (!iter)
    iter = 16*1024;
  if (argc > 2)
    numThreads = Galois::Runtime::LL::getMaxThreads();

  gethostname(bname, sizeof(bname));
  using namespace Galois::Runtime::benchmarking;
  test(getPthreadBarrier());
  test(getCountingBarrier());
  test(getMCSBarrier());
  test(getTopoBarrier());
  test(getDisseminationBarrier());
  return 0;
}
