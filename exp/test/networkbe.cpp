#include <iostream>
#include <cstring>

#include "Galois/Runtime/NetworkBackend.h"

using namespace Galois::Runtime;

int main(int argc, char** argv) {
  NetworkBackend& net = getSystemNetworkBackend();
  
  for (int i = 1; i < NetworkBackend::Num; ++i) {
    auto* sb = net.allocSendBlock();
    sb->size = net.size();
    sb->dest = (NetworkBackend::ID + i) % NetworkBackend::Num;
    if (NetworkBackend::ID == 0)
      std::strcpy((char*)sb->data, "Hi there minions");
    else
      std::strcpy((char*)sb->data, "Hello");
    net.send(sb);
  }
  int seen = 1;
  while (seen < NetworkBackend::Num) {
    NetworkBackend::SendBlock* rb = nullptr;
    while (!rb)
      rb = net.recv();
    
    std::cout << rb->dest << ":" << rb->size << "|" << rb->data << "| at " << NetworkBackend::ID << "\n";
    net.freeSendBlock(rb);
    ++seen;
  }

  return 0;
}
