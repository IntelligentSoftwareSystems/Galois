#include "Galois/Graphs/Graph3.h"

#include <iostream>

using namespace Galois::Graph;

template<typename nd, typename ed, EdgeDirection dir>
using G = ThirdGraph<nd,ed,dir>;


struct op {
  template<typename T, Typename Context>
  void operator()(const T& node, const Context& cnx) {
    node->createEdge(node, node->getData());
  }
};

int main(int argc, const char** argv) {

  G<int, int , EdgeDirection::Out> G;

  for (int x = 0; x < 100; ++x)
    G.createnode(x);

  Galois::foreach<>(G.begin(), G.end(), op());

  for (G::iterator ii = G.begin(), ee = G.end(); ii != ee; ++ii)
    std::cout << ii->getData() << " " << std::distance(ii->begin(), ii->end()) << "\n";

  return 0;
}
