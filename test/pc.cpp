#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Timer.h"
#include "Galois/Galois.h"

#include <iostream>

using namespace Galois::Runtime;

const int num = 1024 * 1024 * 1024;

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
  Galois::Timer tL;
  tL.start();
  Galois::on_each(testL<T>(b));
  tL.stop();
  Galois::Timer tR;
  tR.start();
  Galois::on_each(testR<T>(b));
  tR.stop();
  std::cout << str << " L: " << tL.get() << " R: " << tR.get() << '\n';
}

int main() {
  unsigned M = Galois::Runtime::LL::getMaxThreads();

  while (M) {
    Galois::setActiveThreads(M); //Galois::Runtime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";

    testf<int>("int");
    testf<double>("double");

    M /= 2;
  }

  return 0;
}
