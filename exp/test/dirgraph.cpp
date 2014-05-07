#include "Galois/Galois.h"
#include "Galois/Graphs/Graph3.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>

using namespace Galois::Graph;
using namespace Galois::Runtime;

typedef ThirdGraph<int,void,EdgeDirection::Un> G;

struct op {
  G::pointer graph;

  op(G::pointer g) :graph(g) {}
  op() {}

  template<typename Context>
  void operator()(const int& nodeval, const Context& cnx) {
    G::NodeHandle node1 = graph->createNode(nodeval*2);
    G::NodeHandle node2 = graph->createNode((nodeval*2)+1);
    graph->addNode(node1);
    graph->addNode(node2);
    graph->addEdge(node1, node2);
    if (nodeval == 3) {
      graph->removeNode(node1);
      if (!graph->containsNode(node1))
	std::cout << "Node: " << graph->getData(node1) << " not found as expected" << std::endl;
    }
    //printf("%d iteration in host %u and thread %u\n", nodeval, Distributed::networkHostID, LL::getTID());
  }
};

struct checking {
  G::pointer graph;

  checking(G::pointer g) :graph(g) {}
  checking() {}

  template<typename Context>
  void operator()(G::NodeHandle n, const Context& cnx) {
    printf("value: %d\n", graph->getData(n));
  }
};

int main(int argc, char** argv) {

  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  // check the host id and initialise the network
  getSystemNetworkInterface().start();

  G::pointer Gr = G::allocate();

  Galois::for_each<>(boost::counting_iterator<int>(0), boost::counting_iterator<int>(15), op(Gr));

  for (auto ii = Gr->begin(), ee = Gr->end(); ii != ee; ++ii)
    std::cout << (*ii)->getData() << " " << std::distance((*ii)->begin(), (*ii)->end()) << " ";
  std::cout << "\n";

  //std::cout << "\n" << "Dumping the graph of size " << NThirdGraphSize(Gr) << std::endl;
  for (auto ii = Gr->begin(), ee = Gr->end(); ii != ee; ++ii) {
    std::cout << *ii << "\n";
  }

  std::cout << "\n" << "Using edge_iterator:\n";
  for (auto ii = Gr->begin(), ee = Gr->end(); ii != ee; ++ii) {
    G::NodeHandle N = *ii;
    std::cout << "Node: " << N->getData() << " Num Edges: ";
    std::cout << std::distance(Gr->edge_begin(N), Gr->edge_end(N)) << " edges ";
    for (auto jj = Gr->edge_begin(N), ff = Gr->edge_end(N); jj != ff; ++jj) {
      G::NodeHandle N = jj->getDst();
      std::cout << N << " ";
    }
    std::cout << std::endl;
  }

  // if (NThirdGraphSize(Gr) != 29) {
  //   printf ("ERROR in the no of nodes!\n");
  // }

  Galois::for_each_local(Gr,checking(Gr));

  // master_terminate();
  getSystemNetworkInterface().terminate();

  return 0;
}
