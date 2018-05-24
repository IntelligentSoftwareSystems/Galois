#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/Galois.h"
#include "galois/graphs/FileGraph.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <set>

#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

static const char* name = "Delta-Stepping";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using the Delta-Stepping algorithm";
static const char* url = NULL;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int> startNode("startNode", cll::desc("Node to start search from"), cll::init(0));
static cll::opt<unsigned int> reportNode("reportNode", cll::desc("Node to report distance to"), cll::init(1));
static cll::opt<int> stepShift("delta", cll::desc("Shift value for the deltastep"), cll::init(10));
static cll::opt<bool> useBFS("bfs", cll::desc("Use BFS"), cll::init(false));

int main(int argc, char **argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  typedef galois::graphs::LC_CSR_Graph<void, unsigned int> Graph;
  Graph g;
  
  galois::graphs::readGraph(g, filename);
  std::cout << "Read " << g.size() << " nodes\n";
  std::cout << "Using delta-step of " << (1 << stepShift) << "\n";
  
  // Copy graph data out into edge and weight vectors
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
          boost::no_property,boost::property<boost::edge_weight_t,int> > BGraph;

  BGraph bg(g.size());
  //int numEdges = g.sizeEdges();

  for (Graph::iterator ii = g.begin(),
      ei = g.end(); ii != ei; ++ii) {
    for (Graph::edge_iterator jj = g.edge_begin(*ii),
        ej = g.edge_end(*ii); jj != ej; ++jj) {
      boost::add_edge(*ii, g.getEdgeDst(jj), useBFS ? 1 : g.getEdgeData(jj), bg);
    }
  }

  typedef boost::property_map<BGraph, boost::vertex_index_t>::const_type IndexMap;
  typedef boost::iterator_property_map<std::vector<int>::iterator, IndexMap> DistanceMap;
  std::vector<int> distanceS(boost::num_vertices(bg), 0);
  DistanceMap distance(distanceS.begin(), boost::get(boost::vertex_index, bg));

  galois::StatTimer T;
  T.start();
  boost::dijkstra_shortest_paths(bg,
      boost::vertex(startNode, bg),
      boost::predecessor_map(boost::dummy_property_map())
      .distance_map(distance));
  T.stop();

  std::cout << reportNode << " " << distanceS[reportNode] << "\n";

  return 0;
}
