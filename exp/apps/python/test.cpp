#include "PythonGraph.h"

int main(int argc, char *argv[]) {
  Graph *g = createGraph();
  auto n1 = createNode(g);
  addNode(g, n1);
  addNodeAttr(g, n1, "color", "red");
  auto n2 = createNode(g);
  addNode(g, n2);
  addNodeAttr(g, n2, "language", "english");
  auto n1n2e1 = addMultiEdge(g, n1, n2);
  addEdgeAttr(g, n1n2e1, "weight", "3.0");
  analyzeBFS(g, 1);
  return 0;
}

