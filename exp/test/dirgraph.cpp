#include "Galois/Galois.h"
#include "Galois/Graphs/Graph3.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>

using namespace Galois::Graph;

typedef ThirdGraph<int,void,EdgeDirection::Un> G;

struct op {
  gptr<G> graph;

  op(gptr<G> g) :graph(g) {}
  op() {}

  template<typename Context>
  void operator()(const int& nodeval, const Context& cnx) {
    G::NodeHandle node1 = graph->createNode(nodeval*2);
    G::NodeHandle node2 = graph->createNode((nodeval*2)+1);
    node1->createEdge(node2);
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,graph);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,graph);
  }

};

int main(int argc, char** argv) {

  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  // check the host id and initialise the network
  Galois::Runtime::Distributed::networkStart();

  Galois::Runtime::Distributed::gptr<G> Gr(new G());

  Galois::for_each<>(boost::counting_iterator<int>(0), boost::counting_iterator<int>(5), op(Gr));

  for (auto ii = Gr->begin(), ee = Gr->end(); ii != ee; ++ii)
    std::cout << (*ii)->getData() << " " << std::distance((*ii)->begin(), (*ii)->end()) << " ";
  std::cout << "\n";

  std::cout << "\n" << "Dumping the graph:\n";
  for (auto ii = Gr->begin(), ee = Gr->end(); ii != ee; ++ii) {
    (*ii)->dump(std::cout);
    std::cout << "\n";
  }

  // master_terminate();
  Galois::Runtime::Distributed::networkTerminate();

  return 0;
}
