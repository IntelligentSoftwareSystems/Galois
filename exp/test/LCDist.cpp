#include "Galois/Galois.h"
#include "Galois/Graphs/LC_Dist_Graph.h"

#include <vector>
#include <iostream>

typedef Galois::Graph::LC_Dist<int, int> Graph;
typedef Graph::GraphNode GNode;

struct edge_counter {
  Graph::pointer g;
  
  void operator()(GNode node) {
    GALOIS_ASSERT(std::distance(g->edge_begin(node), g->edge_end(node)) == 2);
  }
  void operator()(GNode node, Galois::UserContext<GNode>&) {
    operator()(node);
  }

  //serialize
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s, g);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s, g);
  }

  //is_trivially_copyable
  typedef int tt_is_copyable;
};

int main(int argc, char *argv[])
{
  Galois::StatManager M;

  int threads = 2;
  if (argc > 1)
    threads = atoi(argv[1]);

  Galois::setActiveThreads(threads);
  auto& net = Galois::Runtime::getSystemNetworkInterface();
  net.start();

  std::vector<unsigned> counts(100, 2); // 100 nodes, 2 edges each

  Graph::pointer g = Graph::allocate(counts);
  std::cout << "Iter dist " << std::distance(g->begin(), g->end()) << "\n";
  std::cout << "Local Iter dist " << std::distance(g->local_begin(), g->local_end()) << "\n";

  for (int i = 0; i < g->size(); ++i) {
    //std::cout << "a " << i << "\n";
    g->addEdge(*(g->begin() + i), *(g->begin() + ((i + 1) % g->size())), 0xDEADBEEF);
  }
  for (int i = 0; i < g->size(); ++i) {
    //std::cout << "b " << i << "\n";
    g->addEdge(*(g->begin() + ((i + 1) % g->size())), *(g->begin() + i));
  }

  //Verify
  std::cout << "\n*\n";
  for (auto ii = g->begin(), ee = g->end(); ii != ee; ++ii)
    edge_counter{g}(*ii);
  std::cout << "\n*\n";
  Galois::for_each_local(g, edge_counter{g});
  std::cout << "\n*\n";
  for (auto ii = g->begin(), ee = g->end(); ii != ee; ++ii)
    edge_counter{g}(*ii);
  std::cout << "\n*\n";

  net.terminate();
  return 0;
};
