#include "Galois/Galois.h"
#include "Galois/Runtime/Serialize.h"

#include <iostream>

using namespace Galois::Runtime;

int main() {

  for (int num = 1; num < 1024; num *= 2) {
    std::vector<double> input(1024*num, 1.0);
    Galois::Timer T;
    T.start();
    for (int i = 0; i < 1000; ++i) {
      SendBuffer b;
      Galois::Runtime::gSerialize(b, input);
    }
    T.stop();
    auto bytes = sizeof(double) * 1024*num * 1000;
    double mbytes = (double)bytes / (1024 * 1024);
    double time = (double)T.get() / 1000;
    std::cout << "Time: " << time << " sec\n";
    std::cout << "Bytes: " << mbytes << " MB\n";
    std::cout << "Throughput: " << mbytes/time
              << " MB/s\n";
  }

  return 0;
}
