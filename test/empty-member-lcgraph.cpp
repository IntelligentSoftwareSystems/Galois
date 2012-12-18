#include "Galois/Graphs/LCGraph.h"

int main() {
  size_t intvoid = sizeof(Galois::Graph::EdgeInfoBase<int,void>);
  size_t intint = sizeof(Galois::Graph::EdgeInfoBase<int,int>);
  std::cout << "sizeof<int,void> = " << intvoid << "\n"
    << "sizeof<int,int> = " << intint << "\n";
  return intvoid < intint ? 0 : 1;
}
