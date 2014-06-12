#include "Galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Graphs/LC_Dist_Graph.h"

#include <vector>

static const char *name = "LC_Dist_Graph testcase";
static const char *desc = "do stuff";
static const char *url  = "LCDist";

typedef Galois::Graph::LC_Dist<int, int> Graph;
typedef Graph::GraphNode GNode;

struct edge_counter {
  Graph::pointer g;
  
  void operator()(GNode node) {
    if (std::distance(g->edge_begin(node), g->edge_end(node)) != 2) {
      std::cerr << "FAILURE: Count is " << std::distance(g->edge_begin(node), g->edge_end(node)) << "\n";
    }
    std::cout << Galois::Runtime::NetworkInterface::ID;
  }
  void operator()(GNode node, Galois::UserContext<GNode>& ctx) {
    operator()(node);
  }
};

int main(int argc, char *argv[])
{
  Galois::StatManager M;
  LonestarStart(argc, argv, name, desc, url);

  std::vector<unsigned> counts(100, 2); // 100 nodes, 2 edges each

  Graph::pointer g = Graph::allocate(counts);
  std::cout << "Iter dist " << std::distance(g->begin(), g->end()) << "\n";
  std::cout << "Local Iter dist " << std::distance(g->local_begin(), g->local_end()) << "\n";

  for (int i = 0; i < g->size(); ++i) {
    std::cout << "a " << i << "\n";
    g->addEdge(*(g->begin() + i), *(g->begin() + ((i + 1) % g->size())), 0xDEADBEEF);
  }
  for (int i = 0; i < g->size(); ++i) {
    std::cout << "b " << i << "\n";
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

  Galois::Runtime::getSystemNetworkInterface().terminate();
  return 0;
};
