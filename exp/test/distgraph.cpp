#include "Galois/Galois.h"
#include "Galois/Graphs/Graph3.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>

using namespace Galois::Graph;

typedef ThirdGraph<int,int,EdgeDirection::Out> G;

struct op {
  G::pointer graph;

  op(G::pointer g) :graph(g) {}
  op() {}

  template<typename Context>
  void operator()(const int& nodeval, const Context& cnx) {
    G::NodeHandle node = graph->createNode(nodeval);
    node->createEdge(node, node, -node->getData());
    std::cout << ".";
  }
 //Gill
  typedef int tt_is_copyable;

};

int main(int argc, char** argv) {

  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  // check the host id and initialise the network
  Galois::Runtime::getSystemNetworkInterface().start();

  G::pointer Gr = G::allocate();

  Galois::for_each<>(boost::counting_iterator<int>(0), boost::counting_iterator<int>(100), op(Gr));

  std::cout << "done loop\n";

  std::cout << "count: " << std::distance(Gr->begin(), Gr->end()) << "\n";

  for (auto ii = Gr->begin(), ee = Gr->end(); ii != ee; ++ii)
    std::cout << (*ii)->getData() << " " << std::distance((*ii)->begin(), (*ii)->end()) << " ";
  std::cout << "\n";

  std::cout << "done print\n";

  // master_terminate();
  Galois::Runtime::getSystemNetworkInterface().terminate();

  return 0;
}
