#include "galois/substrate/PerThreadStorage.h"
#include "galois/Timer.h"
#include "galois/Galois.h"

#include <cstdlib>
#include <iostream>

using namespace galois::substrate;

int num = 1;

template<typename T>
struct testL {
  PerThreadStorage<T>& b;

  testL(PerThreadStorage<T>& B) :b(B) {}
  void operator()(unsigned t, unsigned n) {
    for (int x = 0; x < num; ++x) {
      *b.getLocal() += x;
    }
  }
};

template<typename T>
struct testR {
  PerThreadStorage<T>& b;

  testR(PerThreadStorage<T>& B) :b(B) {}
  void operator()(unsigned t, unsigned n) {
    for (int x = 0; x < num; ++x) {
      *b.getRemote((t + 1) % n) += x;
    }
  }
};

template<typename T>
void testf(const char* str) {
  PerThreadStorage<T> b;
  std::cout << "\nRunning: " << str << " sizeof " << sizeof(PerThreadStorage<T>) << "\n";
  galois::Timer tL;
  tL.start();
  testL<T> L(b);
  galois::on_each(L);
  tL.stop();
  galois::Timer tR;
  tR.start();
  testR<T> R(b);
  galois::on_each(R);
  tR.stop();
  std::cout << str << " L: " << tL.get() << " R: " << tR.get() << '\n';
}

int main(int argc, char** argv) {
  if (argc > 1)
    num = atoi(argv[1]);
  if (num <= 0)
    num = 1024 * 1024 * 1024;

  unsigned M = galois::substrate::getThreadPool().getMaxThreads();

  while (M) {
    galois::setActiveThreads(M); //galois::runtime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";

    testf<int>("int");
    testf<double>("double");

    M /= 2;
  }

  return 0;
}
