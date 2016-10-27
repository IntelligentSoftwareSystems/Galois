#include "PythonGraph.h"

#include <iostream>

/*************************************
 * Node manipulation 
 *************************************/
Node::~Node() {
  using vector_type = std::vector<GNode>;
  if(2 == mode) {
    vVec.~vector_type();
  }
}

/*************************************
 * APIs for PythonGraph
 *************************************/
Graph *createGraph() {
  Graph *g = new Graph();
  return g;
}

void deleteGraph(Graph *g) {
  delete g;
}

void printGraph(Graph* g) {
  size_t nodeID = 0;
  for(auto n: *g) {
    std::cout << "node " << nodeID++ << std::endl;
    for(auto i: g->getData(n).attr) {
      std::cout << "  " << i.first << ": " << i.second << std::endl;
    }
    size_t outEdgeID = 0;
    for(auto e: g->edges(n)) {
      std::cout << "  edge " << outEdgeID++ << std::endl;
      for(auto i: g->getEdgeData(e)) {
        std::cout << "    " << i.first << ": " << i.second << std::endl;
      }
    }
#if !(DIRECTED && !IN_EDGES)
    size_t inEdgeID = 0;
    for(auto e: g->in_edges(n)) {
      std::cout << "  in_edge " << inEdgeID++ << std::endl;
      for(auto i: g->getEdgeData(e)) {
        std::cout << "    " << i.first << ": " << i.second << std::endl;
      }
    }
#endif
  }
}

GNode createNode(Graph *g) {
  return g->createNode();
}

void addNode(Graph *g, const GNode n) {
  g->addNode(n);
}

void setNodeAttr(Graph *g, GNode n, const KeyAltTy key, const ValAltTy val) {
  g->getData(n).attr[key] = val;
}

const ValAltTy getNodeAttr(Graph *g, GNode n, const KeyAltTy key) {
  return const_cast<ValAltTy>(g->getData(n).attr[key].c_str());
}

void removeNodeAttr(Graph *g, GNode n, const KeyAltTy key) {
  g->getData(n).attr.erase(key);
}

Edge addEdge(Graph *g, GNode src, GNode dst) {
  auto ei = g->addEdge(src, dst, Galois::MethodFlag::WRITE);
  return {ei.base(), ei.end()};
}

void setEdgeAttr(Graph *g, Edge e, const KeyAltTy key, const ValAltTy val) {
  g->getEdgeData(edge_iterator(e.base, e.end))[key] = val;
}

const ValAltTy getEdgeAttr(Graph *g, Edge e, const KeyAltTy key) {
  return const_cast<ValAltTy>(g->getEdgeData(edge_iterator(e.base, e.end))[key].c_str());
}

void removeEdgeAttr(Graph *g, Edge e, const KeyAltTy key) {
  g->getEdgeData(edge_iterator(e.base, e.end)).erase(key);
}

void setNumThreads(int numThreads) {
  Galois::setActiveThreads(numThreads < 1 ? 1 : numThreads);
}

