#include "Galois/Substrate/HWTopo.h"

#include <iostream>

int main(int argc, char** argv) {
  auto t = galois::substrate::getHWTopo();
  std::cout << "T,C,P,N: " << t.first.maxThreads << " " << t.first.maxCores << " " << t.first.maxPackages << " " << t.first.maxNumaNodes << "\n";
  for (unsigned i = 0; i < t.first.maxThreads; ++i) {
    auto& c = t.second[i];
    std::cout << "tid: " << c.tid << " leader: " << c.socketLeader << " socket: " << c.socket << " numaNode: " << c.numaNode << " cumulativeMaxSocket: " << c.cumulativeMaxSocket << " osContext: " << c.osContext << " osNumaNode: " << c.osNumaNode << "\n";
  }
  return 0;
}
