#include "Galois/Runtime/Network.h"
#include "Galois/Timer.h"

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

void lp2(RecvBuffer&) {}

int main(int argc, char** argv) {
  NetworkInterface& net = getSystemNetworkInterface();
  
  std::cout << "testing " << networkHostID << " " << networkHostNum << "\n";

  if (networkHostID == 0) {
    Galois::Timer T;
    T.start();
    SendBuffer buf;
    gSerialize(buf,(int) networkHostID);
    net.broadcastMessage(&landingPad, buf);
    for (unsigned int i = 0; i < 1000000; ++i) {
      net.handleReceives();
      SendBuffer buf2;
      net.sendMessage(1, &lp2, buf2);
    }
    T.stop();
    std::cout << "Time " << T.get() << "\n";
  }
  while (true) {
    net.handleReceives();
  }

  return 0;
}
