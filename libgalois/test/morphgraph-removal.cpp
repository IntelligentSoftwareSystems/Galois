#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <type_traits>

#include "galois/graphs/MorphGraph.h"

static unsigned int numNodes = 10;
static bool verbose          = false;

// only tracks out-going edges
using OutGraph =
    galois::graphs::MorphGraph<unsigned int, unsigned int, true, false>;

// tracks out-going and incoming edges w/ shared edge data
using InOutGraph =
    galois::graphs::MorphGraph<unsigned int, unsigned int, true, true>;

// tracks outgoing edges symmetrically w/ shared edge data
using SymGraph = galois::graphs::MorphGraph<unsigned int, unsigned int, false>;

template <class G>
void traverseOutGraph(G& g) {
  for (auto n : g) {
    for (auto e : g.edges(n)) {
      auto dst = g.getEdgeDst(e);
      std::cout << "(" << g.getData(n) << " -> " << g.getData(dst) << "): ";
      std::cout << g.getEdgeData(e) << std::endl;
    }
  }
}

template <class G>
void traverseInGraph(G& g) {
  for (auto n : g) {
    for (auto ie : g.in_edges(n)) {
      auto src = g.getEdgeDst(ie);
      std::cout << "(" << g.getData(n) << " <- " << g.getData(src) << "): ";
      std::cout << g.getEdgeData(ie) << std::endl;
    }
  }
}

template <>
void traverseInGraph(OutGraph&) {}

// construct a directed clique w/ (i, j) where i < j
template <class G>
void constructGraph(G& g, std::vector<typename G::GraphNode>& v) {
  // add nodes
  for (unsigned int i = 0; i < numNodes; i++) {
    auto n = g.createNode(i);
    v.push_back(n);
    g.addNode(n);
  }

  // add edges
  for (unsigned int i = 0; i < numNodes; i++) {
    for (unsigned int j = i + 1; j < numNodes; j++) {
      g.getEdgeData(g.addEdge(v[i], v[j])) = (i + j);
    }
  }

  if (verbose) {
    std::cout << "Original" << std::endl;
    traverseOutGraph(g);
    traverseInGraph(g);
  }
}

template <class G>
void removeGraphOutEdge(G& g, typename G::GraphNode n1,
                        typename G::GraphNode n2) {
  auto e = g.findEdge(n1, n2);
  if (e != g.edge_end(n1)) {
    g.removeEdge(n1, e);
  }
}

void removeGraphInEdge(SymGraph& g, SymGraph::GraphNode n1,
                       SymGraph::GraphNode n2) {
  auto e12                            = g.findInEdge(n1, n2);
  auto GALOIS_USED_ONLY_IN_DEBUG(e21) = g.findEdge(n2, n1);

  if (e12 == g.in_edge_end(n1)) {
    assert(e21 == g.edge_end(n1));
  } else {
    assert(e21 != g.edge_end(n1));
    assert(n2 == g.getEdgeDst(e12));
    assert(n1 == g.getEdgeDst(e21));
    assert(g.getEdgeData(e12) == g.getEdgeData(e21));
    g.removeEdge(n1, e12);
    //    g.removeEdge(n2, e21); this is also OK
  }
}

void removeGraphInEdge(InOutGraph& g, InOutGraph::GraphNode n1,
                       InOutGraph::GraphNode n2) {
  auto ie                           = g.findInEdge(n1, n2);
  auto GALOIS_USED_ONLY_IN_DEBUG(e) = g.findEdge(n2, n1);
  if (ie == g.in_edge_end(n1)) {
    assert(e == g.edge_end(n2));
  } else {
    assert(e != g.edge_end(n2));
    assert(n2 == g.getEdgeDst(ie));
    assert(n1 == g.getEdgeDst(e));
    assert(g.getEdgeData(ie) == g.getEdgeData(e));
    //    g.removeEdge(n1, ie); // this leads to compile error
    g.removeEdge(n2, e);
  }
}

unsigned int countUnmatchedEdge(OutGraph& g,
                                std::vector<typename OutGraph::GraphNode>& v,
                                unsigned int i, unsigned int j) {
  unsigned int unmatched = 0;

  // nodes whose out edges are all removed
  for (unsigned int ri = 0; ri < i; ri++) {
    for (unsigned int rj = 0; rj < numNodes; rj++) {
      unmatched += (g.edge_end(v[ri]) != g.findEdge(v[ri], v[rj]));
    }
  }

  // the node whose out edge removed up to j
  for (unsigned int rj = 0; rj < j + 1; rj++) {
    unmatched += (g.edge_end(v[i]) != g.findEdge(v[i], v[rj]));
  }
  for (unsigned int rj = j + 1; rj < numNodes; rj++) {
    unmatched += (g.edge_end(v[i]) == g.findEdge(v[i], v[rj]));
  }

  // nodes whose out edges are kept wholly
  for (unsigned int ri = i + 1; ri < numNodes; ri++) {
    for (unsigned int rj = 0; rj < ri + 1; rj++) {
      unmatched += (g.edge_end(v[ri]) != g.findEdge(v[ri], v[rj]));
    }
    for (unsigned int rj = ri + 1; rj < numNodes; rj++) {
      unmatched += (g.edge_end(v[ri]) == g.findEdge(v[ri], v[rj]));
    }
  }

  return unmatched;
}

unsigned int countUnmatchedEdge(InOutGraph& g,
                                std::vector<typename InOutGraph::GraphNode>& v,
                                unsigned int i, unsigned int j) {
  unsigned int unmatched = 0;

  // nodes whose out edges are all removed
  for (unsigned int ri = 0; ri < i; ri++) {
    for (unsigned int rj = 0; rj < numNodes; rj++) {
      unmatched += (g.edge_end(v[ri]) != g.findEdge(v[ri], v[rj]));
      unmatched += (g.in_edge_end(v[rj]) != g.findInEdge(v[rj], v[ri]));
    }
  }

  // the node whose out edge removed up to j
  for (unsigned int rj = 0; rj < j + 1; rj++) {
    unmatched += (g.edge_end(v[i]) != g.findEdge(v[i], v[rj]));
    unmatched += (g.in_edge_end(v[rj]) != g.findInEdge(v[rj], v[i]));
  }
  for (unsigned int rj = j + 1; rj < numNodes; rj++) {
    unmatched += (g.edge_end(v[i]) == g.findEdge(v[i], v[rj]));
    unmatched += (g.in_edge_end(v[rj]) == g.findInEdge(v[rj], v[i]));
  }

  // nodes whose out edges are kept wholly
  for (unsigned int ri = i + 1; ri < numNodes; ri++) {
    for (unsigned int rj = 0; rj < ri + 1; rj++) {
      unmatched += (g.edge_end(v[ri]) != g.findEdge(v[ri], v[rj]));
      unmatched += (g.in_edge_end(v[rj]) != g.findInEdge(v[rj], v[ri]));
    }
    for (unsigned int rj = ri + 1; rj < numNodes; rj++) {
      unmatched += (g.edge_end(v[ri]) == g.findEdge(v[ri], v[rj]));
      unmatched += (g.in_edge_end(v[rj]) == g.findInEdge(v[rj], v[ri]));
    }
  }

  return unmatched;
}

unsigned int countUnmatchedEdge(SymGraph& g,
                                std::vector<typename SymGraph::GraphNode>& v,
                                unsigned int i, unsigned int j) {
  unsigned int unmatched = 0;

  // no self loops
  for (unsigned int k = 0; k < numNodes; k++) {
    unmatched += (g.edge_end(v[k]) != g.findEdge(v[k], v[k]));
    unmatched += (g.in_edge_end(v[k]) != g.findInEdge(v[k], v[k]));
  }

  // nodes whose out edges are all removed
  for (unsigned int ri = 0; ri < i; ri++) {
    for (unsigned int rj = ri + 1; rj < numNodes; rj++) {
      unmatched += (g.edge_end(v[ri]) != g.findEdge(v[ri], v[rj]));
      unmatched += (g.in_edge_end(v[rj]) != g.findInEdge(v[rj], v[ri]));
    }
  }

  // the node whose out edge removed up to j
  for (unsigned int rj = i; rj < j + 1; rj++) {
    unmatched += (g.edge_end(v[i]) != g.findEdge(v[i], v[rj]));
    unmatched += (g.in_edge_end(v[rj]) != g.findInEdge(v[rj], v[i]));
  }
  for (unsigned int rj = j + 1; rj < numNodes; rj++) {
    unmatched += (g.edge_end(v[i]) == g.findEdge(v[i], v[rj]));
    unmatched += (g.in_edge_end(v[rj]) == g.findInEdge(v[rj], v[i]));
  }

  // nodes whose out edges are kept wholly
  for (unsigned int ri = i + 1; ri < numNodes; ri++) {
    for (unsigned int rj = ri + 1; rj < numNodes; rj++) {
      unmatched += (g.edge_end(v[ri]) == g.findEdge(v[ri], v[rj]));
      unmatched += (g.in_edge_end(v[rj]) == g.findInEdge(v[rj], v[ri]));
    }
  }

  return unmatched;
}

template <class G>
unsigned int testGraphOutEdgeRemoval(G& g,
                                     std::vector<typename G::GraphNode>& v) {
  constructGraph(g, v);
  unsigned int numFailedRemoval = 0;

  for (unsigned int i = 0; i < numNodes; i++) {
    for (unsigned int j = i + 1; j < numNodes; j++) {
      removeGraphOutEdge(g, v[i], v[j]);
      numFailedRemoval += (0 != countUnmatchedEdge(g, v, i, j));

      if (verbose) {
        std::cout << "Removed edge (" << i << " -> " << j << ")" << std::endl;
        traverseOutGraph(g);
        traverseInGraph(g);
      }
    }
  }

  return numFailedRemoval;
}

template <class G>
unsigned int testGraphInEdgeRemoval(G& g,
                                    std::vector<typename G::GraphNode>& v) {
  constructGraph(g, v);
  unsigned int numFailedRemoval = 0;

  for (unsigned int i = 0; i < numNodes; i++) {
    for (unsigned int j = i + 1; j < numNodes; j++) {
      removeGraphInEdge(g, v[j], v[i]);
      numFailedRemoval += (0 != countUnmatchedEdge(g, v, i, j));

      if (verbose) {
        std::cout << "Removed in_edge (" << j << " <- " << i << ")"
                  << std::endl;
        traverseOutGraph(g);
        traverseInGraph(g);
      }
    }
  }

  return numFailedRemoval;
}

int main() {
  galois::SharedMemSys G;
  unsigned int numFailure = 0;

  OutGraph outG;
  std::vector<OutGraph::GraphNode> outV;
  auto num = testGraphOutEdgeRemoval(outG, outV);
  numFailure += num;
  std::cout << "OutGraph: Failed " << num << " edge removals" << std::endl;

  SymGraph symG, symG2;
  std::vector<SymGraph::GraphNode> symV, symV2;
  num = testGraphOutEdgeRemoval(symG, symV);
  numFailure += num;
  std::cout << "SymGraph: Failed " << num << " edge removals" << std::endl;
  num = testGraphInEdgeRemoval(symG2, symV2);
  numFailure += num;
  std::cout << "SymGraph: Failed " << num << " in_edge removals" << std::endl;

  InOutGraph inOutG, inOutG2;
  std::vector<InOutGraph::GraphNode> inOutV, inOutV2;
  num = testGraphOutEdgeRemoval(inOutG, inOutV);
  numFailure += num;
  std::cout << "InOutGraph: Failed " << num << " edge removals" << std::endl;
  num = testGraphInEdgeRemoval(inOutG2, inOutV2);
  numFailure += num;
  std::cout << "InOutGraph: Failed " << num << " in_edge removals" << std::endl;

  return (numFailure > 0) ? -1 : 0;
}
