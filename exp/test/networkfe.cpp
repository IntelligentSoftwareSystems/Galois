#include <iostream>
#include <cstring>

#include "Galois/Runtime/Network.h"

using namespace Galois::Runtime;

void testFunc(uint32_t src) {
  std::cout << "Called at " << getSystemNetworkInterface().ID << " by " << src << "\n";
}

int main(int argc, char** argv) {
  NetworkInterface& net = getSystemNetworkInterface();
  
  for (uint32_t i = 0; i < net.Num; ++i) {
    if (i != net.ID) {
      net.sendAlt(i, testFunc, net.ID);
    }
  }

  do {
    net.handleReceives();
  } while (true);

  return 0;
}
