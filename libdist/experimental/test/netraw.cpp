#include <iostream>

#include "galois/Runtime/NetworkIO.h"

int main() {
  uint32_t ID, Num;
  std::unique_ptr<galois::runtime::NetworkIO> net;

  std::tie(net,ID,Num) = galois::runtime::makeNetworkIOMPI();

  std::cout << ID << " " << Num << "\n";
  
  for (int x = 1; x <= 100; ++x) {

    for (int i = 0; i < Num; ++i) {
      galois::runtime::NetworkIO::message m;
      m.len = x;
      m.data.reset(new uint8_t[x]);
      m.host = i;
      for (int y = 0; y < x; ++y)
        m.data[y] = ID;
      net->enqueue(std::move(m));
    }
    
    for (int i = 0; i < Num; ++i) {
      galois::runtime::NetworkIO::message m;
      do {
        m = net->dequeue();
      } while (!m.len);
      std::cout << ID << ":" << m.len << ":";
      for (int y = 0; y < m.len; ++y)
        std::cout << " " << (char)(m.data[y] + '0');
      std::cout << "\n";
    }
  }
   
  return 0;
  
}
