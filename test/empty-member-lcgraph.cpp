#include "Galois/Graphs/LCGraph.h"

int main() {
  constexpr size_t intvoid = sizeof(Galois::Graph::LCGraphImpl::EdgeInfoBase<int,void>);
  constexpr size_t intint = sizeof(Galois::Graph::LCGraphImpl::EdgeInfoBase<int,int>);
  static_assert(intvoid < intint, "Failed to do empty member optimization");
  return intvoid < intint ? 0 : 1;
}
