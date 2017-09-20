#include "galois/Galois.h"
#include "galois/Runtime/Serialize.h"

#include <iostream>

using namespace galois::Runtime;

int main() {

  std::map<int, double> compare;
  
  std::cout << galois::runtime::gSized(std::vector<int>(100)) << "\n";
  //check by hand that gSized reduces this to a constant
  std::cout << galois::runtime::gSized(1.2, 3.4, '1') << "\n";

  std::cout << "Ser\n\n";
  
  for (int num = 1; num < 1024; num *= 2) {
    std::vector<double> input(1024*num, 1.0);
    galois::Timer T;
    T.start();
    for (int i = 0; i < 1000; ++i) {
      SendBuffer b;
      galois::runtime::gSerialize(b, input);
    }
    T.stop();
    auto bytes = sizeof(double) * 1024*num * 1000;
    double mbytes = (double)bytes / (1024 * 1024);
    double time = (double)T.get() / 1000;
    compare[num] = time;
    std::cout << "Time: " << time << " sec\n";
    std::cout << "Bytes: " << mbytes << " MB\n";
    std::cout << "Throughput: " << mbytes/time
              << " MB/s\n";
  }

  std::cout << "\n\nSer + DeSer\n\n";
  
  for (int num = 1; num < 1024; num *= 2) {
    std::vector<double> input(1024*num, 1.0);
    galois::Timer T;
    T.start();
    for (int i = 0; i < 1000; ++i) {
      SendBuffer b;
      galois::runtime::gSerialize(b, input);
      RecvBuffer r(std::move(b));
      galois::runtime::gDeserialize(r, input);
    }
    T.stop();
    auto bytes = sizeof(double) * 1024*num * 1000;
    double mbytes = (double)bytes / (1024 * 1024);
    double time = (double)T.get() / 1000;
    std::cout << "Time: " << time << " sec\n";
    std::cout << "Bytes: " << mbytes << " MB\n";
    std::cout << "Throughput: " << mbytes/time
              << " MB/s\n";
    std::cout << "Ratio: " << compare[num] / time << " (ser/(ses+des))\n";
  }

  return 0;
}
