#include "PythonGraph.h"

#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
  std::vector<GNode> nodes;
  std::vector<Edge> edges;

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

  edges.push_back(addMultiEdge(g, nodes[0], nodes[1]));
  addEdgeAttr(g, edges[0], "weight", "3.0");

  edges.push_back(addMultiEdge(g, nodes[0], nodes[1]));
  addEdgeAttr(g, edges[1], "place", "texas");
  addEdgeAttr(g, edges[1], "garbage", "discard");
  printGraph(g);
  std::cout << "=====" << std::endl;

  removeEdgeAttr(g, edges[1], "garbage");
  removeEdgeAttr(g, edges[1], "galois_id");
  printGraph(g);
  std::cout << "=====" << std::endl;

  analyzeBFS(g, 1);
  printGraph(g);
  std::cout << "=====" << std::endl;

  deleteGraph(g);
  nodes.clear();
  edges.clear();

  return 0;
}

