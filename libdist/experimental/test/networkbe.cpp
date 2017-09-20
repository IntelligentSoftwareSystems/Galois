#include <iostream>
#include <cstring>

#include "galois/Runtime/NetworkBackend.h"

using namespace galois::Runtime;

int main(int argc, char** argv) {
  NetworkBackend& net = getSystemNetworkBackend();
  
  for (int i = 1; i < net.Num(); ++i) {
    auto* sb = net.allocSendBlock();
    sb->size = net.size();
    sb->dest = (net.ID() + i) % net.Num();
    if (net.ID() == 0)
      std::strcpy((char*)sb->data, "Hi there minions");
    else
      std::strcpy((char*)sb->data, "Hello");
    net.send(sb);
  }
  int seen = 1;
  while (seen < net.Num()) {
    NetworkBackend::SendBlock* rb = nullptr;
    while (!rb)
      rb = net.recv();
    
    std::cout << rb->dest << ":" << rb->size << "|" << rb->data << "| at " << net.ID() << "\n";
    net.freeSendBlock(rb);
    ++seen;
  }

  return 0;
}
