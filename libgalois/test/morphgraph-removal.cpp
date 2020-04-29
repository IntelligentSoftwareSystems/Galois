#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <type_traits>

#include "galois/graphs/MorphGraph.h"
//#include "galois/graphs/Morph_SepInOut_Graph.h"

static std::string graphType;
static unsigned int numNodes;

// only tracks out-going edges
using OutGraph =
    galois::graphs::MorphGraph<unsigned int, unsigned int, true, false>;

// tracks out-going and incoming edges w/ shared edge data
using InOutGraph =
    galois::graphs::MorphGraph<unsigned int, unsigned int, true, true>;

// tracks outgoing edges symmetrically w/ shared edge data
using SymGraph = galois::graphs::MorphGraph<unsigned int, unsigned int, false>;

template<class G>
void traverseGraph(G& g) {
  for (auto n: g) {
    for (auto e: g.edges(n)) {
      auto dst = g.getEdgeDst(e);
      std::cout << "(" << g.getData(n) << " -> " << g.getData(dst) << "): ";
      std::cout << g.getEdgeData(e) << std::endl;
    }
    for (auto ie: g.in_edges(n)) {
      auto src = g.getEdgeDst(ie);
      std::cout << "(" << g.getData(n) << " <- " << g.getData(src) << "): ";
      std::cout << g.getEdgeData(ie) << std::endl;
    }
  }
  std::cout << std::endl;
}

// construct a directed clique w/ (i, j) where i < j
template<class G>
void constructGraph(G& g, std::vector<typename G::GraphNode>& v) {
  // add nodes
  for (unsigned int i = 0; i < numNodes; i++) {
    auto n = g.createNode(i);
    v.push_back(n);
    g.addNode(n);
  }

  // add edges
  for (unsigned int i = 0; i < numNodes; i++) {
    for (unsigned int j = i+1; j < numNodes; j++) {
      g.getEdgeData(g.addEdge(v[i], v[j])) = (i+j);
    }
  }
}

template<class G>
void removeGraphOutEdge(G& g, typename G::GraphNode n1, typename G::GraphNode n2) {
  auto e = g.findEdge(n1, n2);
  if (e != g.edge_end(n1)) {
    g.removeEdge(n1, e);
  }
}

template<class G>
void removeGraphInEdge(G& g, typename G::GraphNode n1, typename G::GraphNode n2) {
  // no incoming edges, do nothing
}

template<>
void removeGraphInEdge(SymGraph& g, SymGraph::GraphNode n1, SymGraph::GraphNode n2) {
  removeGraphOutEdge(g, n2, n1);
}

template<>
void removeGraphInEdge(InOutGraph& g, InOutGraph::GraphNode n1, InOutGraph::GraphNode n2) {
  removeGraphOutEdge(g, n2, n1);
}

template<class G>
bool verifyInEdgeRemovalUptoJI(G& g, std::vector<typename G::GraphNode>& v, unsigned int j, unsigned int i) {
  std::cout << "In-edge removal is done up to (" << j << " <- " << i << ")" << std::endl;
  bool result = true;

  // nodes whose out-edges are all removed
  for (unsigned int ri = 0; ri < i; ri++) {
    for (unsigned int rj = ri+1; rj < numNodes; rj++) {
      if (g.in_edge_end(v[rj]) != g.findInEdge(v[rj], v[ri])) {
        std::cout << "Failed to remove in_edge (" << rj << " <- " << ri << ")" << std::endl;
        result = false;
      }
    }
  }

  // the node whose out-edges are removed up to j
  for (unsigned int rj = i+1; rj <= j; rj++) {
    if (g.in_edge_end(v[rj]) != g.findInEdge(v[rj], v[i])) {
      std::cout << "Failed to remove in_edge (" << rj << " <- " << i << ")" << std::endl;
      result = false;
    }
  }
  for (unsigned int rj = j+1; rj < numNodes; rj++) {
    if (g.in_edge_end(v[rj]) == g.findInEdge(v[rj], v[i])) {
      std::cout << "Should not have removed in_edge (" << rj << " <- " << i << ")" << std::endl;
      result = false;
    }
  }

  // nodes whose out-edges are still there
  for (unsigned int ri = i+1; ri < numNodes; ri++) {
    for (unsigned int rj = ri+1; rj < numNodes; rj++) {
      if (g.in_edge_end(v[rj]) == g.findInEdge(v[rj], v[ri])) {
        std::cout << "Should not have removed edge (" << rj << " <- " << ri << ")" << std::endl;
        result = false;
      }
    }
  }

  return result;
}

template<class G>
bool verifyOutEdgeRemovalUptoIJ(G& g, std::vector<typename G::GraphNode>& v, unsigned int i, unsigned int j) {
  std::cout << "Edge removal is done up to (" << i << " -> " << j << ")" << std::endl;
  bool result = true;

  // nodes whose out-edges are all removed
  for (unsigned int ri = 0; ri < i; ri++) {
    if (std::distance(g.edge_begin(v[ri]), g.edge_end(v[ri]))) {
      std::cout << "Some out-edges are not removed from " << ri << std::endl;
    }
    for (unsigned int rj = ri+1; rj < numNodes; rj++) {
      if (g.edge_end(v[ri]) != g.findEdge(v[ri], v[rj])) {
        std::cout << "Failed to remove edge (" << ri << " -> " << rj << ")" << std::endl;
        result = false;
      }
    }
  }

  // the node whose out-edges are removed up to j
  if ((numNodes - j - 1) != std::distance(g.edge_begin(v[i]), g.edge_end(v[i]))) {
    std::cout << "Error in removing out-edges from " << i << std::endl;
  }
  for (unsigned int rj = i+1; rj <= j; rj++) {
    if (g.edge_end(v[i]) != g.findEdge(v[i], v[rj])) {
      std::cout << "Failed to remove edge (" << i << " -> " << rj << ")" << std::endl;
      result = false;
    }
  }
  for (unsigned int rj = j+1; rj < numNodes; rj++) {
    if (g.edge_end(v[i]) == g.findEdge(v[i], v[rj])) {
      std::cout << "Should not have removed edge (" << i << " -> " << rj << ")" << std::endl;
      result = false;
    }
  }

  // nodes whose out-edges are still there
  for (unsigned int ri = i+1; ri < numNodes; ri++) {
    if ((numNodes - ri - 1) != std::distance(g.edge_begin(v[ri]), g.edge_end(v[ri]))) {
      std::cout << "Some out-edges are removed prematurely from " << ri << std::endl;
    }
    for (unsigned int rj = ri+1; rj < numNodes; rj++) {
      if (g.edge_end(v[ri]) == g.findEdge(v[ri], v[rj])) {
        std::cout << "Should not have removed edge (" << ri << " -> " << rj << ")" << std::endl;
        result = false;
      }
    }
  }

  return result;
}

template<class G>
void testGraphOutEdgeRemoval(G& g, std::vector<typename G::GraphNode>& v) {
  constructGraph(g, v);

  for (unsigned int i = 0; i < numNodes; i++) {
    for (unsigned int j = i+1; j < numNodes; j++) {
      removeGraphOutEdge(g, v[i], v[j]);
      if (verifyOutEdgeRemovalUptoIJ(g, v, i, j)) {
        std::cout << "Normal up to removal of edge (" << i << " -> " << j << ")" << std::endl;
      }
#if 1
      if (std::is_same<G, OutGraph>::value) {
        continue;
      }
      else if (verifyInEdgeRemovalUptoJI(g, v, j, i)) {
        std::cout << "Normal up to removal of in_edge (" << j << " <- " << i << ")" << std::endl;
      }
      traverseGraph(g);
#endif
    }
  }
}

int main(int argc, char* argv[]) {
  galois::SharedMemSys G;

  if (argc < 3) {
    std::cout << "Usage: ./test-morphgraph-removal <num_nodes> "
                 "<out|in-out|symmetric>"
              << std::endl;
    return 0;
  }

  numNodes = std::stoul(argv[1]);
  graphType = argv[2];

  if ("out" == graphType) {
//    OutGraph outG;
//    std::vector<OutGraph::GraphNode> outV;
//    testGraphOutEdgeRemoval(outG, outV);
  }
  else if ("in-out" == graphType) {
    InOutGraph inOutG;
    std::vector<InOutGraph::GraphNode> inOutV;
    testGraphOutEdgeRemoval(inOutG, inOutV);
  }
  else if ("symmetric" == graphType) {
    SymGraph symG;
    std::vector<SymGraph::GraphNode> symV;
    testGraphOutEdgeRemoval(symG, symV);
  }
  else {
    std::cout << "Unrecognized graph type " << graphType << std::endl;
  }

  galois::runtime::reportParam("MorphGraph Removal", "No. Nodes", numNodes);
  galois::runtime::reportParam("MorphGraph Removal", "Graph Type", graphType);
  return 0;
}
