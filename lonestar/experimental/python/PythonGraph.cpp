#include "PythonGraph.h"

#include <iostream>

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
    std::cout << "node " << n << std::endl;
    auto& nd = g->getData(n);
    std::cout << "node label " << nd.label << std::endl;
    std::cout << "node id " << nd.id << std::endl;
    for(auto e: g->edges(n)) {
      std::cout << "  edge to " << g->getEdgeDst(e) << std::endl;
      auto& ed = g->getEdgeData(e);
      std::cout << "edge label " << ed.label << std::endl;
      std::cout << "edge timestamp " << ed.timestamp << std::endl;
    }
  }
}

void allocate(Graph *g, size_t numNodes, size_t numEdges) {
  g->allocateFrom(numNodes, numEdges);
  g->constructNodes();
}

void fixEndEdge(Graph *g, uint32_t nodeIndex, uint64_t edgeIndex) {
  g->fixEndEdge(nodeIndex, edgeIndex);
}

void setNode(Graph *g, uint32_t nodeIndex, uint32_t label, uint64_t id) {
  auto& nd = g->getData(nodeIndex);
  nd.label = label;
  nd.id = id;
}

void constructEdge(Graph *g, uint64_t edgeIndex, uint32_t dstNodeIndex, uint32_t label, uint64_t timestamp) {
  EdgeData ed;
  ed.label = label;
  ed.timestamp = timestamp;
  g->constructEdge(edgeIndex, dstNodeIndex, ed);
}

//void setNodeAttr(Graph *g, GNode n, const KeyAltTy key, const ValAltTy val) {
//  g->getData(n).attr[key] = val;
//}
//
//const ValAltTy getNodeAttr(Graph *g, GNode n, const KeyAltTy key) {
//  return const_cast<ValAltTy>(g->getData(n).attr[key].c_str());
//}
//
//void removeNodeAttr(Graph *g, GNode n, const KeyAltTy key) {
//  g->getData(n).attr.erase(key);
//}

//void setEdgeAttr(Graph *g, Edge e, const KeyAltTy key, const ValAltTy val) {
//  auto ei = g->findEdge(e.src, e.dst);
//  assert(ei != g.edge_end(e.src));
//  g->getEdgeData(ei)[key] = val;
//}

//const ValAltTy getEdgeAttr(Graph *g, Edge e, const KeyAltTy key) {
//  auto ei = g->findEdge(e.src, e.dst);
//  assert(ei != g.edge_end(e.src));
//  return const_cast<ValAltTy>(g->getEdgeData(ei)[key].c_str());
//}

//void removeEdgeAttr(Graph *g, Edge e, const KeyAltTy key) {
//  auto ei = g->findEdge(e.src, e.dst);
//  assert(ei != g.edge_end(e.src));
//  g->getEdgeData(ei).erase(key);
//}

void setNumThreads(int numThreads) {
  galois::setActiveThreads(numThreads < 1 ? 1 : numThreads);
}

size_t getNumNodes(Graph *g) {
  return g->size();
}

size_t getNumEdges(Graph *g) {
  return g->sizeEdges();
}

