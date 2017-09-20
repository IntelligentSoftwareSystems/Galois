#include "Galois/Graphs/LCGraph.h"

int main() {
  constexpr size_t intvoid = sizeof(galois::Graph::detail::EdgeInfoBase<int,void>);
  constexpr size_t intint = sizeof(galois::Graph::detail::EdgeInfoBase<int,int>);
  static_assert(intvoid < intint, "Failed to do empty member optimization");
  return intvoid < intint ? 0 : 1;
}
