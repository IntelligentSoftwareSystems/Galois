#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>

#include <mpi.h>

#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Timer.h"

using namespace galois::Runtime;

static std::atomic<int> num;

void func(RecvBuffer&) {
  ++num;
}

volatile int cont = 0;

int main(int argc, char** argv) {
  num = 0;
  int trials = 1000000;
  if (argc > 1)
    trials = atoi(argv[1]);

  auto& net = getSystemNetworkInterface();
  auto& bar = getSystemBarrier();

  if (net.Num != 2) {
    std::cerr << "Just run with 2 hosts\n";
    return 1;
  }

  //while (!cont) {}

  for (int s = 10 ; s < trials; s*=1.1) { //1069 is from 10. 243470 is also
    std::vector<char> vec(s);
    galois::Timer T1, T2, T3;
    bar.wait();
    T3.start();
    T1.start();
    SendBuffer buf;
    gSerialize(buf, vec);
    T1.stop();
    int oldnum = num;
    bar.wait();
    T2.start();
    if (net.ID == 0) {
      net.send(1, func, buf);
      net.flush();
    } else {
      while (num == oldnum) { net.handleReceives(); }
    }
    T2.stop();
    bar.wait();
    T3.stop();
    bar.wait();
    std::cerr << "H" << net.ID 
              << " size " << s 
              << " T1 " << T1.get()
              << " T2 " << T2.get()
              << " T3 " << T3.get()
              << " B " << (T3.get() - T1.get() ? s / (T3.get() - T1.get()) : 0)
              << "\n";
    bar.wait();
  }
  return 0;
}
