#include <iostream>
#include <cstring>

#include "Galois/Runtime/Network.h"

using namespace Galois::Runtime;

void testFunc(uint32_t src) {
  std::cout << "Called at " << getSystemNetworkInterface().ID << " by " << src << "\n";
}
void testFunc2(uint32_t src, uint32_t x) {
  std::cout << "Called at " << getSystemNetworkInterface().ID << " by " << src << " with " << x << "\n";
}

int main(int argc, char** argv) {
  NetworkInterface& net = getSystemNetworkInterface();
  
  for (uint32_t i = 0; i < net.Num; ++i) {
    if (i != net.ID) {
      net.sendAlt(i, testFunc, net.ID);
      net.sendAlt(i, testFunc2, net.ID, net.ID);
   }
  }

  net.flush();

  do {
    net.handleReceives();
  } while (true);

  return 0;
}
