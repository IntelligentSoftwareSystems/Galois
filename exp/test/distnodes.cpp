#include "Galois/Galois.h"
#include "Galois/Graphs/Graph3.h"

#include <iostream>

using namespace Galois::Graph;

template<typename nd, typename ed, EdgeDirection dir>
using G = ThirdGraph<nd,ed,dir>;


struct op {
  template<typename T, typename Context>
  void operator()(const T& node, const Context& cnx) {
    node->createEdge(node, -node->getData());
  }
};

int main(int argc, const char** argv) {

  typedef G<int, int , EdgeDirection::Out> GTy;
  GTy Gr;

  for (int x = 0; x < 100; ++x)
    Gr.createNode(x);

  Galois::for_each<>(Gr.begin(), Gr.end(), op());

  for (auto ii = Gr.begin(), ee = Gr.end(); ii != ee; ++ii)
    std::cout << (*ii)->getData() << " " << std::distance((*ii)->begin(), (*ii)->end()) << "\n";

  ///debugging stuff below this point

  Galois::Runtime::Distributed::SerializeBuffer B;
  //serialize the pointer
  auto oldB = *Gr.begin();
  B.serialize(oldB);
  //serialize the node
  B.serialize(*oldB);

  B.print(std::cout);
  cout << "\n";

  typename GTy::NodeHandle foo;
  B.deserialize(foo);

  auto bar = Gr.createNode();
  B.deserialize(*bar);

  B.print(std::cout);
  cout << "\n";

  oldB.dump(std::cout);
  std::cout << "\n";
  foo.dump(std::cout);
  std::cout << "\n";
  (*oldB).dump(std::cout);
  std::cout << "\n";
  (*bar).dump(std::cout);
  std::cout << "\n";

  return 0;
}
