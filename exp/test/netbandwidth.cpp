#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>

#include "Galois/Runtime/NetworkBackend.h"
#include "Galois/Timer.h"

using namespace Galois::Runtime;

static const int trials = 1000;

int main(int argc, char** argv) {
  NetworkBackend& net = getSystemNetworkBackend();
  
  int count = 0;

  std::ostringstream os;

  for (int i = 1; i <= net.size(); i += 8) {
    Galois::Timer T;
    T.start();
    for (int x = 0; x < trials; ++x) {
      ++count;
      if (net.ID() == 0) {
        auto* sb = net.allocSendBlock();
        sb->size = i;
        sb->dest = 1;
        net.send(sb);
      } else {
        NetworkBackend::SendBlock* rb = nullptr;
        while (!rb)
          rb = net.recv();
        if (rb->size != i)
          std::cerr << "Size error: expected " << i << " got " << rb->size << "\n";
        net.freeSendBlock(rb);
      }
    }
    net.flush(true);
    T.stop();
    //std::cout << "On " << net.ID() << " at " << i << " total " << count << " in " << T.get() << " for " << (i * trials / (double)T.get()) * ((double) 1000 / (1024*1024)) << " MB/s\n";
    os << "," << (i * trials / (double)T.get()) * ((double) 1000 / (1024*1024));
  }
  if (net.ID() == 0) {
    std::cout << "Bytes";
    for (int i = 1; i <= net.size(); i += 8)
      std::cout << "," << i;
    std::cout << "\n";
  }
  std::cout << net.ID() << os.str() << "\n";
  return 0;
}
