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
  for(auto n: *g) {
    std::cout << "node" << std::endl;
    for(auto i: g->getData(n).attr) {
      std::cout << "  " << i.first << ": " << i.second << std::endl;
    }
    for(auto e: g->edges(n)) {
      std::cout << "  edge" << std::endl;
      for(auto i: g->getEdgeData(e).attr) {
        std::cout << "    " << i.first << ": " << i.second << std::endl;
      }
    }
  }
}

GNode createNode(Graph *g) {
  return g->createNode();
}

void addNode(Graph *g, const GNode n) {
  g->addNode(n);
}

void addNodeAttr(Graph *g, GNode n, const KeyAltTy key, const ValAltTy val) {
  g->getData(n).attr[key] = val;
}

void removeNodeAttr(Graph *g, GNode n, const KeyAltTy key) {
  g->getData(n).attr.erase(key);
}

Edge addEdge(Graph *g, GNode src, GNode dst) {
  auto ei = g->addEdge(src, dst, Galois::MethodFlag::WRITE);
  return {ei.base(), ei.end()};
}

void addEdgeAttr(Graph *g, Edge e, const KeyAltTy key, const ValAltTy val) {
  g->getEdgeData(edge_iterator(e.base, e.end)).attr[key] = val;
}

void removeEdgeAttr(Graph *g, Edge e, const KeyAltTy key) {
  g->getEdgeData(edge_iterator(e.base, e.end)).attr.erase(key);
}

void setNumThreads(int numThreads) {
  Galois::setActiveThreads(numThreads < 1 ? 1 : numThreads);
}

