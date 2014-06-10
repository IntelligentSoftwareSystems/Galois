#include "Galois/Galois.h"
#include <chrono>
#include <thread>
#include <iostream>

struct Function {
  void operator()(unsigned tid, unsigned total) {
    Galois::Runtime::LL::gPrint("host: ", Galois::Runtime::NetworkInterface::ID, " tid: ", tid, "\n");
  }
};

int main(int argc, char** argv) {
  int threads = 1;
  if (argc > 1)
    threads = atoi(argv[1]);

  Galois::setActiveThreads(threads);
  Galois::Runtime::getSystemNetworkInterface().start();

  std::cout << "Hosts: " << Galois::Runtime::NetworkInterface::Num << " ";
  std::cout << "Threads: " << threads << "\n";

  Galois::on_each(Function());

  std::this_thread::sleep_for(std::chrono::seconds(5));

  Galois::Runtime::getSystemNetworkInterface().handleReceives();
  Galois::Runtime::getSystemNetworkInterface().terminate();

  return 0;
}
