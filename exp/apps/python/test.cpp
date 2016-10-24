#include "PythonGraph.h"
#include "AnalyzeBFS.h"

#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
  std::vector<GNode> nodes;
  std::vector<Edge> edges;

  Graph *g = createGraph();

  nodes.push_back(createNode(g));
  addNode(g, nodes[0]);
  setNodeAttr(g, nodes[0], "color", "red");

  nodes.push_back(createNode(g));
  addNode(g, nodes[1]);
  setNodeAttr(g, nodes[1], "language", "english");
  setNodeAttr(g, nodes[1], "garbage", "to_be_deleted");
  printGraph(g);
  std::cout << "=====" << std::endl;

  removeNodeAttr(g, nodes[1], "garbage");
  printGraph(g);
  std::cout << "=====" << std::endl;

  edges.push_back(addEdge(g, nodes[0], nodes[1]));
  setEdgeAttr(g, edges[0], "weight", "3.0");
  printGraph(g);
  std::cout << "=====" << std::endl;

  edges.push_back(addEdge(g, nodes[0], nodes[1]));
  setEdgeAttr(g, edges[1], "place", "texas");
  setEdgeAttr(g, edges[1], "garbage", "discard");
  printGraph(g);
  std::cout << "=====" << std::endl;

  removeEdgeAttr(g, edges[1], "garbage");
  removeEdgeAttr(g, edges[1], "galois_id");
  printGraph(g);
  std::cout << "=====" << std::endl;

  setNumThreads(1);

  analyzeBFS(g, nodes[0], "dist");
  printGraph(g);
  std::cout << "=====" << std::endl;

  deleteGraph(g);
  nodes.clear();
  edges.clear();

  return 0;
}

