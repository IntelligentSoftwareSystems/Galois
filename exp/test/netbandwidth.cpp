#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>

#include <mpi.h>

#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Timer.h"

using namespace Galois::Runtime;

//tests small message bandwidth

static int num = 0;

void func(RecvBuffer&) {
  ++num;
}

int main(int argc, char** argv) {

  int trials = 1000000;
  if (argc > 1)
    trials = atoi(argv[0]);

  NetworkInterface& net = getSystemNetworkInterface();
  //  Galois::StatManager sm;

  std::cout << "Sending " << trials << " between every host.  " << trials * (net.Num - 1) << " total messages per host.\n";

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
  std::cerr << "\n*" << net.ID << " " << T.get() << "!" << num << "\n";
  std::cout << "Calling MPI_Finaliz\n";
  MPI_Finalize();
  std::stringstream os;
  os << net.ID << ": " << T.get() << "ms " << num << " " << (double)num / T.get() << " msg/ms\n";
  std::cout << os.str();
  return 0;
}
