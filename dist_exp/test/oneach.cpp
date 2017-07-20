#include "Galois/Galois.h"
#include <iostream>

struct Function {
  void operator()(unsigned tid, unsigned total) {
    Galois::Runtime::LL::gPrint("host: ", Galois::Runtime::NetworkInterface::ID, " tid: ", tid, "\n");
  }
};

int main(int argc, char** argv) {
  int threads = 2;
  if (argc > 1)
    threads = atoi(argv[1]);

  Galois::setActiveThreads(threads);
  auto& net = Galois::Runtime::getSystemNetworkInterface();
  net.start();

  std::cout << "Hosts: " << Galois::Runtime::NetworkInterface::Num << " ";
  std::cout << "Threads: " << threads << "\n";

  Galois::on_each(Function());

  net.terminate();

  return 0;
}
