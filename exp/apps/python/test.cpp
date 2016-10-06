#include "PythonGraph.h"

#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
  std::vector<GNode> nodes;
  std::vector<edge_iterator> edges;

  Graph *g = createGraph();

  nodes.push_back(createNode(g));
  addNode(g, nodes[0]);
  addNodeAttr(g, nodes[0], "color", "red");

  nodes.push_back(createNode(g));
  addNode(g, nodes[1]);
  addNodeAttr(g, nodes[1], "language", "english");

  edges.push_back(addMultiEdge(g, nodes[0], nodes[1]));
  addEdgeAttr(g, edges[0], "weight", "3.0");

  edges.push_back(addMultiEdge(g, nodes[0], nodes[1]));
  addEdgeAttr(g, edges[1], "place", "texas");

  for(auto n: nodes) {
    for(auto i: g->getData(n)) {
      std::cout << i.first << ": " << i.second << std::endl;
    }
  }

  for(auto e: edges) {
    for(auto i: g->getEdgeData(e)) {
      std::cout << i.first << ": " << i.second << std::endl;
    }
  }

  analyzeBFS(g, 1);

  deleteGraph(g);
  nodes.clear();
  edges.clear();

  return 0;
}

