#include "Galois/Runtime/Network.h"

#include <iostream>

using namespace Galois::Runtime::Distributed;

bool didbcast = false;

void landingPad(RecvBuffer& foo) {
  int val;
  gDeserialize(foo,val);
  std::cout << "Landed on " << networkHostID << " from " << val << "\n";
  if (!didbcast) {
    didbcast = true;
    SendBuffer buf;
    gSerialize(buf,(int) networkHostID);
    getSystemNetworkInterface().broadcastMessage(&landingPad, buf);
  }
}

int main(int argc, char** argv) {
  NetworkInterface& net = getSystemNetworkInterface();
  
  std::cout << "testing " << networkHostID << " " << networkHostNum << "\n";

  if (networkHostID == 0) {
    SendBuffer buf;
    gSerialize(buf,(int) networkHostID);
    net.broadcastMessage(&landingPad, buf);
  }
  while (true) {
    net.handleReceives();
  }

  return 0;
}
