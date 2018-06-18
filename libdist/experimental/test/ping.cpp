#include <iostream>
#include <cstring>
#include <sys/time.h>

#include "galois/runtime/Network.h"

using namespace galois::runtime;

std::atomic<int> count;

int diff_micro(timeval t1, timeval t2) {
  return (((t1.tv_sec - t2.tv_sec) * 1000000) + (t1.tv_usec - t2.tv_usec));
}

void func2(timeval v) {
  timeval v2;
  gettimeofday(&v2, 0);
  std::cout << diff_micro(v2, v) << " ";
  --count;
}

void func(timeval v) {
  getSystemNetworkInterface().sendAlt(0, func2, v);
  ++count;
}

int main(int argc, char** argv) {
  NetworkInterface& net = getSystemNetworkInterface();

  if (net.ID == 0) {
    for (int i = 0; i < 100; ++i) {
      ++count;
      timeval v;
      gettimeofday(&v, 0);
      net.sendAlt(1, func, v);
      while (count)
        net.handleReceives();
    }
  } else {
    do {
      net.handleReceives();
    } while (count < 100);
  }

  net.terminate();

  return 0;
}
