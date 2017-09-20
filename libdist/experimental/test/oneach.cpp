#include "Galois/Galois.h"
#include <iostream>

struct Function {
  void operator()(unsigned tid, unsigned total) {
    galois::Runtime::LL::gPrint("host: ", galois::Runtime::NetworkInterface::ID, " tid: ", tid, "\n");
  }
};

int main(int argc, char** argv) {
  int threads = 2;
  if (argc > 1)
    threads = atoi(argv[1]);

  galois::setActiveThreads(threads);
  auto& net = galois::Runtime::getSystemNetworkInterface();
  net.start();

  std::cout << "Hosts: " << galois::Runtime::NetworkInterface::Num << " ";
  std::cout << "Threads: " << threads << "\n";

  galois::on_each(Function());

  net.terminate();

  return 0;
}
