#include <iostream>

#include "Galois/Runtime/Network.h"

std::atomic<int> num;

void say(uint32_t i) {
  std::cout << i;
  ++num;
}

int main() {
  galois::runtime::NetworkInterface& net = galois::runtime::makeNetworkBuffered();

  std::cout << net.ID << " " << net.Num << "\n";
  
  num = 0;

  for (int x = 1; x <= 100; ++x)
    for (int i = 0; i < net.Num; ++i)
      net.sendAlt(i, say, net.ID); 
  while (num != 100*net.Num) { net.handleReceives(); }
   
  return 0;
  
}
