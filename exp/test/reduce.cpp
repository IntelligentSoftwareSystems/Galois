#include "Galois/Galois.h"
#include "Galois/Graphs/Graph3.h"
#include "Galois/Reduction.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>

using namespace Galois::Graph;
using namespace Galois::Runtime;
using namespace Galois::Runtime::Distributed;

typedef ThirdGraph<int,void,EdgeDirection::Un> G;
typedef Galois::DGReducible<unsigned, std::plus<unsigned> > RD;

struct op {
  gptr<G> graph;
  gptr<RD> count;

  op(gptr<G> g, gptr<RD> c) :graph(g), count(c) {}
  op() {}

  template<typename Context>
  void operator()(const int& nodeval, const Context& cnx) {
    std::cout << networkHostID;
    count->get() += 1;
    //    G::NodeHandle node1 = graph->createNode(nodeval*2);
    //    graph->addNode(node1);
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,graph, count);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,graph,count);
  }
};

int main(int argc, char** argv) {

  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  // check the host id and initialise the network
  networkStart();

  gptr<G> Gr(new G());
  gptr<RD> Cr(new RD());

  std::cout << "Loop\n";
  Galois::for_each<>(boost::counting_iterator<int>(0), boost::counting_iterator<int>(20), op(Gr, Cr));
  std::cout <<"\n";

  for (int i = 0; i < 4; ++i) {
    std::cout << Cr->get() << "\n";
    std::cout << "Reduce\n";
    unsigned& x = Cr->doReduce();
    std::cout << Cr->get() << " " << x << "\n";
    std::cout << "Broadcast\n";
    Cr->doBroadcast(x);
    std::cout << Cr->get() << " " << x << "\n";
  }

  // master_terminate();
  Galois::Runtime::Distributed::networkTerminate();

  return 0;
}
