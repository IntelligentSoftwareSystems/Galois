#ifndef GALOIS_TEST_UTILS
#define GALOIS_TEST_UTILS

#include "../src/utils/ConnectivityManager.h"

std::vector<GNode> generateSampleGraph(Graph& graph) {
  std::vector<GNode> nodes;
  ConnectivityManager connManager{graph};
  nodes.push_back(
      connManager.createNode(NodeData{false, Coordinates{0, 0, 0}, false}));
  nodes.push_back(
      connManager.createNode(NodeData{false, Coordinates{0, 1, 0}, false}));
  nodes.push_back(
      connManager.createNode(NodeData{false, Coordinates{1, 0, 0}, false}));
  nodes.push_back(
      connManager.createNode(NodeData{false, Coordinates{1, 1, 0}, false}));

  connManager.createEdge(nodes[0], nodes[1], true, Coordinates{0, 0.5, 0}, 1);
  connManager.createEdge(nodes[1], nodes[3], true, Coordinates{0.5, 1, 0}, 1);
  connManager.createEdge(nodes[2], nodes[3], true, Coordinates{1, 0.5, 0}, 1);
  connManager.createEdge(nodes[0], nodes[2], true, Coordinates{0.5, 0, 0}, 1);
  connManager.createEdge(nodes[3], nodes[0], false, Coordinates{0.5, 0.5, 0},
                         sqrt(2));

  nodes.push_back(connManager.createInterior(nodes[0], nodes[1], nodes[3]));
  nodes.push_back(connManager.createInterior(nodes[0], nodes[3], nodes[2]));
  return nodes;
}

int countHEdges(Graph& graph) {
  int counter = 0;
  for (auto n : graph) {
    if (graph.containsNode(n) && n->getData().isHyperEdge()) {
      ++counter;
    }
  }
  return counter;
}

int countVertices(Graph& graph) {
  int counter = 0;
  for (auto n : graph) {
    if (!(n->getData().isHyperEdge())) {
      ++counter;
    }
  }
  return counter;
}

#endif