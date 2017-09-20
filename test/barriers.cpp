#include "Galois/Timer.h"
#include "Galois/Galois.h"
#include "Galois/Substrate/BarrierImpl.h"

#include <iostream>
#include <cstdlib>
#include <unistd.h>

unsigned iter = 1;
unsigned numThreads = 2;

char bname[100];

struct emp {
  galois::Substrate::Barrier& b;

  void go() {
    for (unsigned i = 0; i < iter; ++i) {
      // if (galois::Runtime::LL::getTID() == 0)
      //   usleep(100);
      b.wait();
    }
    // std::cout << galois::Runtime::LL::getTID() << " ";
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

void test(std::unique_ptr<galois::Substrate::Barrier> b) {
  unsigned M = numThreads;
  if (M > 16) M /= 2;
  while (M) {   
    galois::setActiveThreads(M); //galois::Runtime::LL::getMaxThreads());
    b->reinit(M);
    galois::Timer t;
    t.start();
    galois::on_each(emp{*b.get()});
    t.stop();
    std::cout << bname << "," << b->name() << "," << M << "," << t.get() << "\n";
    M -= 1;
  }
}

int main(int argc, char** argv) {
  if (argc > 1)
    iter = atoi(argv[1]);
  if (!iter)
    iter = 16*1024;
  if (argc > 2)
    numThreads = galois::Substrate::getThreadPool().getMaxThreads();

  gethostname(bname, sizeof(bname));
  using namespace galois::Substrate;
  test(createPthreadBarrier(1));
  test(createCountingBarrier(1));
  test(createMCSBarrier(1));
  test(createTopoBarrier(1));
  test(createDisseminationBarrier(1));
  return 0;
}
