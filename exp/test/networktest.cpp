#include "Galois/Runtime/Network.h"

#include <iostream>

using namespace Galois::Runtime::Distributed;

bool didbcast = false;

void landingPad(RecvBuffer& foo) {
  int val;
  foo.deserialize(val);
  std::cout << "Landed on " << networkHostID << " from " << val << "\n";
  if (!didbcast) {
    didbcast = true;
    SendBuffer buf;
    buf.serialize((int) networkHostID);
    getSystemNetworkInterface().broadcastMessage(&landingPad, buf);
  }
}

int main(int argc, char** argv) {
  NetworkInterface& net = getSystemNetworkInterface();
  
  std::cout << "testing " << networkHostID << " " << networkHostNum << "\n";

  if (networkHostID == 0) {
    SendBuffer buf;
    buf.serialize((int) networkHostID);
    net.broadcastMessage(&landingPad, buf);
  }
  while (true) {
    net.handleRecieves();
  }

  return 0;
}
