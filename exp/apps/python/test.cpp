#include "PythonGraph.h"

#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
  std::vector<GNode> nodes;

  Graph *g = createGraph();

  nodes.push_back(createNode(g));
  addNode(g, nodes[0]);
  addNodeAttr(g, nodes[0], "color", "red");

  nodes.push_back(createNode(g));
  addNode(g, nodes[1]);
  addNodeAttr(g, nodes[1], "language", "english");
  addNodeAttr(g, nodes[1], "garbage", "to_be_deleted");
  printGraph(g);
  std::cout << "=====" << std::endl;

  removeNodeAttr(g, nodes[1], "garbage");
  printGraph(g);
  std::cout << "=====" << std::endl;

  addMultiEdge(g, nodes[0], nodes[1], "n1n2e1");
  addEdgeAttr(g, nodes[0], nodes[1], "n1n2e1", "weight", "3.0");

  addMultiEdge(g, nodes[0], nodes[1], "n1n2e2");
  addEdgeAttr(g, nodes[0], nodes[1], "n1n2e2", "place", "texas");
  addEdgeAttr(g, nodes[0], nodes[1], "n1n2e2", "garbage", "discard");
  printGraph(g);
  std::cout << "=====" << std::endl;

  removeEdgeAttr(g, nodes[0], nodes[1], "n1n2e2", "garbage");
  removeEdgeAttr(g, nodes[0], nodes[1], "n1n2e2", "galois_id");
  printGraph(g);
  std::cout << "=====" << std::endl;

  analyzeBFS(g, 1);
  printGraph(g);
  std::cout << "=====" << std::endl;

  deleteGraph(g);
  nodes.clear();

  return 0;
}

