#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/PerHostStorage.h"
#include "Galois/Timer.h"

#include <iostream>
#include <cmath>

using namespace Galois::Runtime::Distributed;

bool didbcast = false;

struct sayHi {
  sayHi(Galois::Runtime::PerHost<sayHi> ptr, DeSerializeBuffer& b) { std::cout << "Hi " << this << "\n"; }
  sayHi(Galois::Runtime::PerHost<sayHi> ptr) { std::cout << "Hi " << this << "\n"; }
  ~sayHi() { std::cout << "Bye\n"; }
  void getInitData(SerializeBuffer& b) {}
};

void landingPad(RecvBuffer& foo) {
  int val;
  gDeserialize(foo,val);
  std::cout << "Landed on " << networkHostID << " from " << val << "\n";
  if (!didbcast) {
    didbcast = true;
    SendBuffer buf;
    gSerialize(buf,(int) networkHostID);
    getSystemNetworkInterface().broadcast(&landingPad, buf);
  }
}

void lp2(RecvBuffer&) {
  static double d;
  d = cos(d);
}

void lp2a() {}

void lp3(unsigned x, unsigned y) {
  std::cout << "alt dispatch " << x << "," << y << "\n";
}

int main(int argc, char** argv) {
  NetworkInterface& net = getSystemNetworkInterface();
  
  std::cout << "testing " << networkHostID << " " << networkHostNum << "\n";

  if (networkHostID == 0) {
    net.sendAlt(1, lp3, 4U, 5U);
    Galois::Runtime::PerHost<sayHi> p = Galois::Runtime::PerHost<sayHi>::allocate();
    p.remote(1).dump(std::cout);
    Galois::Runtime::PerHost<sayHi>::deallocate(p);
  }

  std::cout << "Begin loop classic\n";

  if (networkHostID == 0) {
    Galois::Timer T;
    T.start();
    SendBuffer buf;
    gSerialize(buf,(int) networkHostID);
    //net.broadcastMessage(&landingPad, buf);
    for (unsigned int i = 0; i < 1000000; ++i) {
      net.handleReceives();
      SendBuffer buf2;
      net.send(1, &lp2, buf2);
    }
    T.stop();
    std::cout << "Time " << T.get() << "\n";

    Galois::Timer T2;
    T2.start();
    net.sendAlt(1, lp2a);
    for (unsigned int i = 0; i < 1000000; ++i) {
      net.handleReceives();
      net.sendAlt(1, lp2a);
    }
    T2.stop();
    std::cout << "Time " << T2.get() << "\n";


  }
  while (true) {
    net.handleReceives();
  }

  return 0;
}
