#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>

#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Timer.h"

using namespace Galois::Runtime;

static const int trials = 1000000;
static int num = 0;

void func(RecvBuffer&) {
  ++num;
}

int main(int argc, char** argv) {
  NetworkInterface& net = getSystemNetworkInterface();
  Galois::StatManager sm;

  getSystemBarrier().wait();

  Galois::Timer T;
  T.start();
  for (int j = 0; j < net.Num; ++j) {
    if (j != net.ID) {
      for(int i = 0; i < trials; ++i) {
        SendBuffer buf;
        net.send(j, func, buf);
      }
    }
  }
  net.flush();
  while (num < trials * (net.Num - 1)) { net.handleReceives(); }
  getSystemBarrier().wait();
  T.stop();
  std::cerr << "\n*" << net.ID << " " << T.get() << "!\n";
  return 0;
}
