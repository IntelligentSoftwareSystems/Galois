#include "Galois/Galois.h"
#include <iostream>

struct Function {
  void operator()(unsigned tid, unsigned total) {
    galois::runtime::LL::gPrint("host: ", galois::runtime::NetworkInterface::ID, " tid: ", tid, "\n");
  }
};

int main(int argc, char** argv) {
  int threads = 2;
  if (argc > 1)
    threads = atoi(argv[1]);

  galois::setActiveThreads(threads);
  auto& net = galois::runtime::getSystemNetworkInterface();
  net.start();

  std::cout << "Hosts: " << galois::runtime::NetworkInterface::Num << " ";
  std::cout << "Threads: " << threads << "\n";

  galois::on_each(Function());

  net.terminate();

  return 0;
}
