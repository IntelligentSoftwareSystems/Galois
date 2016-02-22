#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>

#include <mpi.h>

#include "Galois/Runtime/Network.h"
//#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Substrate.h"
#include "Galois/Timer.h"

using namespace Galois::Runtime;

//tests small message bandwidth

static int num_recv = 0;

void func(RecvBuffer&) {
  ++num_recv;
}

int main(int argc, char** argv) {

  int trials = 1000000;
  if (argc > 1)
    trials = atoi(argv[1]);

  NetworkInterface& net = getSystemNetworkInterface();
  Galois::StatManager sm;

  std::cerr << "Using " << net.Num << "\n";

  for (int num = 0; num < 10; ++num) {

    std::cerr << "Sending " << trials << " between every host.  " << trials * (num - 1) << " total messages per host.\n";
    
    Galois::Runtime::getHostBarrier().wait();
    //getSystemBarrier().wait();
    std::cerr <<" barrier 1\n";

    Galois::Timer T;
    T.start();
    //if (net.ID < num) {
      for (int j = 0; j < net.Num ; ++j) {
        if (j != net.ID) {
          std::cerr << net.ID << " --> " << j << "\n";
          for(int i = 0; i < trials; ++i) {
            SendBuffer buf;
            net.send(j, func, buf);
          }
        }
      }
    //}
    net.flush();
    while (num_recv < trials * (net.Num - 1)) { net.handleReceives(); }
    std::cerr <<" HR done \n";
    //Galois::Runtime::getSystemBarrier().wait();
    num_recv = 0;
    Galois::Runtime::getHostBarrier().wait();
    std::cerr <<" barrier 2\n";
    T.stop();
    std::stringstream os;
    os << net.ID << "@" << num << ": " << T.get() << "ms " << num << " " << (double)num / T.get() << " msg/ms\n";
    std::cout << os.str();
  }
  return 0;
}
