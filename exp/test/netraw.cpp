#include <iostream>

#include "Galois/Runtime/NetworkIO.h"

int main() {
  uint32_t ID, Num;
  Galois::Runtime::NetworkIO* net;

  std::tie(net,ID,Num) = Galois::Runtime::makeNetworkIOMPI();

  std::cout << ID << " " << Num << "\n";
  
  for (int x = 1; x <= 100; ++x) {

    for (int i = 0; i < Num; ++i) {
      std::vector<uint8_t> data;
      for (int y = 0; y < x; ++y)
        data.push_back(ID);
      net->enqueue(i, data);
    }
    
    for (int i = 0; i < Num; ++i) {
      std::vector<uint8_t> data;
      do {
        data = net->dequeue();
      } while (data.empty());
      std::cout << ID << ":" << data.size() << ":";
      for (auto d : data)
        std::cout << " " << (char)(d + '0');
      std::cout << "\n";
    }
  }
   
  return 0;
  
}
