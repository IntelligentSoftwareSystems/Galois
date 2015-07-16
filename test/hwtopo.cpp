#include "Galois/Substrate/HWTopo.h"

#include <iostream>

int main(int argc, char** argv) {
  auto t = Galois::Substrate::getHWTopo();
  std::cout << "T,C,P: " << t->getMaxThreads() << " " << t->getMaxCores() << " " << t->getMaxPackages() << "\n";
  for (unsigned i = 0; i < t->getMaxThreads(); ++i) {
    auto& c = t->getThreadInfo(i);
    std::cout << "l,L,p,os: " << c.isPackageLeader << " " << c.packageLeader << " " << c.package << " " << c.hwContext << "\n";
  }
  return 0;
}
