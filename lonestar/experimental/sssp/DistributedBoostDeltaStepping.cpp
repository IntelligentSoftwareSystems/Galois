/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/Graph/Graph.h"
#include "galois/Galois.h"
#include "galois/Graph/FileGraph.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <set>

#include <boost/graph/use_mpi.hpp>
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/distributed/adjacency_list.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/distributed/delta_stepping_shortest_paths.hpp>

static const unsigned int DIST_INFINITY =
    std::numeric_limits<unsigned int>::max() - 1;

static const char* name = "Delta-Stepping";
static const char* desc =
    "Computes the shortest path from a source node to all nodes in a directed "
    "graph using the Delta-Stepping algorithm";
static const char* url = NULL;

namespace cll = llvm::cl;
static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int> startNode("startNode",
                                        cll::desc("Node to start search from"),
                                        cll::init(0));
static cll::opt<unsigned int>
    reportNode("reportNode", cll::desc("Node to report distance to"),
               cll::init(1));
static cll::opt<int> stepShift("delta",
                               cll::desc("Shift value for the deltastep"),
                               cll::init(10));
static cll::opt<bool> useBfs("bfs", cll::desc("Use BFS"), cll::init(false));

typedef int weight_type;
struct WeightedEdge {
  WeightedEdge(weight_type weight = 0) : weight(weight) {}
  weight_type weight;

  template <typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar& weight;
  }
};

struct VertexProperties {
  VertexProperties(int d = 0) : distance(d) {}
  int distance;

  template <typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar& distance;
  }
};

template <typename Graph, typename Map>
void verify(const Graph& g, unsigned reportNode, Map map) {
  map.set_max_ghost_cells(0);
  get(map, boost::vertex(reportNode, g));
  synchronize(process_group(g));
  std::cout << get(map, boost::vertex(reportNode, g)) << "\n";
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  typedef galois::graphs::LC_CSR_Graph<void, unsigned int> Graph;
  Graph g;

  galois::graphs::readGraph(g, filename);
  std::cout << "Read " << g.size() << " nodes\n";
  std::cout << "Using delta-step of " << (1 << stepShift) << "\n";

  // Copy graph data out into edge and weight vectors
  typedef std::pair<int, int> Edge;
  typedef boost::adjacency_list<
      boost::vecS,
      boost::distributedS<boost::graph::distributed::mpi_process_group,
                          boost::vecS>,
      boost::directedS, VertexProperties, WeightedEdge>
      BGraph;

  BGraph bg(g.size());
  int numEdges = g.sizeEdges();
  typedef boost::graph_traits<BGraph>::edge_descriptor BEdge;

  for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
    for (Graph::edge_iterator jj = g.edge_begin(*ii), ej = g.edge_end(*ii);
         jj != ej; ++jj) {
      boost::add_edge(boost::vertex(*ii, bg), boost::vertex(*jj, bg),
                      useBFS ? 1 : g.getEdgeData(*ii, *jj), bg);
    }
  }

  galois::StatTimer T;
  T.start();
  boost::graph::distributed::delta_stepping_shortest_paths(
      bg, boost::vertex(startNode, bg), boost::dummy_property_map(),
      boost::get(&VertexProperties::distance, bg),
      boost::get(&WeightedEdge::weight, bg), 1 << stepShift);
  T.stop();

  verify(bg, reportNode, boost::get(&VertexProperties::distance, bg));
  // std::cout << reportNode << " " << distanceS[reportNode] << "\n";

  return 0;
}
