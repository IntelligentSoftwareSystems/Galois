#include "Galois/Galois.h"
#include "Galois/Graphs/Graph3.h"

#include <iostream>

using namespace Galois::Graph;

template<typename nd, typename ed, EdgeDirection dir>
using G = ThirdGraph<nd,ed,dir>;


struct op {
  template<typename T, typename Context>
  void operator()(const T& node, const Context& cnx) {
    node->createEdge(node, node->getData());
  }
};

int main(int argc, const char** argv) {

  G<int, int , EdgeDirection::Out> Gr;

  for (int x = 0; x < 100; ++x)
    Gr.createNode(x);

  Galois::for_each<>(Gr.begin(), Gr.end(), op());

  for (typename G<int,int,EdgeDirection::Out>::iterator ii = Gr.begin(), ee = Gr.end(); ii != ee; ++ii)
    std::cout << (*ii)->getData() << " " << std::distance((*ii)->begin(), (*ii)->end()) << "\n";

  return 0;
}
