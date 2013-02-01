#include "Galois/Galois.h"
#include "Galois/Graphs/Graph3.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>

using namespace Galois::Graph;

typedef ThirdGraph<int,int,EdgeDirection::Out> G;

struct op {
  gptr<G> graph;

  op(gptr<G> g) :graph(g) {}
  op() {}

  template<typename Context>
  void operator()(const int& nodeval, const Context& cnx) {
    G::NodeHandle node = graph->createNode(nodeval);
    node->createEdge(node, -node->getData());
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    s.serialize(graph);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    s.deserialize(graph);
  }

};

int main(int argc, char** argv) {

  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  // check the host id and initialise the network
  Galois::Runtime::Distributed::networkStart();

  Galois::Runtime::Distributed::gptr<G> Gr(new G());

  Galois::for_each<>(boost::counting_iterator<int>(0), boost::counting_iterator<int>(100), op(Gr));

  for (auto ii = Gr->begin(), ee = Gr->end(); ii != ee; ++ii)
    std::cout << (*ii)->getData() << " " << std::distance((*ii)->begin(), (*ii)->end()) << " ";
  std::cout << "\n";

  // master_terminate();
  Galois::Runtime::Distributed::networkTerminate();

  return 0;
}
