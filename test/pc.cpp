#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Timer.h"
#include "Galois/Galois.h"

#include <iostream>

GaloisRuntime::MCSBarrier mbarrier;
GaloisRuntime::PthreadBarrier pbarrier;
GaloisRuntime::FastBarrier fbarrier;
GaloisRuntime::FasterBarrier ffbarrier;
GaloisRuntime::TopoBarrier tbarrier;

template<typename BarTy>
struct test {
  BarTy& b;
  test(BarTy& B) :b(B) {}
  void operator()(unsigned t, unsigned n) {
    for (int x = 0; x < 128 * 1024; ++x) {
      b.wait();
    }
    std::cout << ".";
  }
};

template<typename T>
void testf(T& b, const char* str) {
  std::cout << "\nRunning: " << &b << " " << str << "\n";
  b.reinit(Galois::getActiveThreads());
  Galois::Timer t;
  t.start();
  Galois::on_each(test<T>(b));
  t.stop();
  std::cout << str << ": " << t.get() << '\n';
}

int main() {
  unsigned M = GaloisRuntime::LL::getMaxThreads();

  while (M) {
    Galois::setActiveThreads(M); //GaloisRuntime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";

    if (0) {
      int count = 128 * 1024 * 1024;

      Galois::Timer t;
      t.start();
      GaloisRuntime::PerCPU<int> v;
      for (int i = 0; i < count; ++i)
        v.get()++;
      t.stop();

      Galois::Timer t2;
      t2.start();
      GaloisRuntime::PerThreadStorage<int> v2;
      for (int i = 0; i < count; ++i)
        (*v2.getLocal())++;
      t2.stop();

      Galois::Timer t3;
      t3.start();
      GaloisRuntime::PerCPU<int> v3;
      for (int i = 0; i < count; ++i)
        v3.get(1)++;
      t3.stop();

      Galois::Timer t4;
      t4.start();
      GaloisRuntime::PerThreadStorage<int> v4;
      for (int i = 0; i < count; ++i)
        (*v4.getRemote(1))++;
      t4.stop();

      std::cout << t.get() << " " << t2.get() << "\n";
      std::cout << t3.get() << " " << t4.get() << "\n";
    }

    // testf(pbarrier, "pthread");
    // testf(fbarrier, "fast");
    testf(mbarrier, "mcs");
    //  testf(ffbarrier, "faster");
    testf(tbarrier, "topo");

    M /= 2;
  }

  return 0;
}
