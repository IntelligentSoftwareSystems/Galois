#include "PythonGraph.h"

#include <iostream>

/*************************************
 * APIs for GaloisRuntime
 *************************************/
void initGaloisRuntime() {
  static galois::SharedMemSys *G;
  if (G != NULL) delete G;
  G = new galois::SharedMemSys();
}

void setNumThreads(int numThreads) {
  galois::setActiveThreads(numThreads < 1 ? 1 : numThreads);
}

int getNumThreads() {
  return galois::getActiveThreads();
}

/*************************************
 * APIs for PythonGraph
 *************************************/
AttributedGraph *createGraph() {
  AttributedGraph *g = new AttributedGraph();
  return g;
}

void deleteGraph(AttributedGraph *g) {
  delete g;
}

void printGraph(AttributedGraph* g) {
  Graph& graph = g->graph;
  auto& nodeLabels = g->nodeLabels;
  auto& edgeLabels = g->edgeLabels;
  auto& nodeNames = g->nodeNames;
  for(auto n: graph) {
    auto& nd = graph.getData(n);
    auto& srcLabel = nodeLabels[nd.label];
    auto& srcName = nodeNames[nd.id];
    for(auto e: graph.edges(n)) {
      auto& dst = graph.getData(graph.getEdgeDst(e));
      auto& dstLabel = nodeLabels[dst.label];
      auto& dstName = nodeNames[dst.id];
      auto& ed = graph.getEdgeData(e);
      auto& edgeLabel = edgeLabels[ed.label];
      auto& edgeTimestamp = ed.timestamp;
      std::cout << srcLabel << " " << srcName << " " 
                << edgeLabel << " " << dstLabel << " " 
                << dstName << " at " << edgeTimestamp << std::endl;
    }
  }
}

void allocateGraph(AttributedGraph *g, size_t numNodes, size_t numEdges, size_t numNodeLabels, size_t numEdgeLabels) {
  g->graph.allocateFrom(numNodes, numEdges);
  g->graph.constructNodes();
  g->nodeLabels.resize(numNodeLabels);
  g->edgeLabels.resize(numEdgeLabels);
  g->nodeNames.resize(numNodes);
}

void fixEndEdge(AttributedGraph *g, uint32_t nodeIndex, uint64_t edgeIndex) {
  g->graph.fixEndEdge(nodeIndex, edgeIndex);
}

void setNode(AttributedGraph *g, uint32_t nodeIndex, uint32_t label, char *nodeName) {
  auto& nd = g->graph.getData(nodeIndex);
  nd.label = label;
  nd.id = nodeIndex;
  g->nodeNames[nodeIndex] = nodeName;
}

void setNodeLabel(AttributedGraph *g, uint32_t label, char *name) {
  g->nodeLabels[label] = name;
}

void setEdgeLabel(AttributedGraph *g, uint32_t label, char *name) {
  g->edgeLabels[label] = name;
}

void constructEdge(AttributedGraph *g, uint64_t edgeIndex, uint32_t dstNodeIndex, uint32_t label, uint64_t timestamp) {
  EdgeData ed;
  ed.label = label;
  ed.timestamp = timestamp;
  g->graph.constructEdge(edgeIndex, dstNodeIndex, ed);
}

size_t getNumNodes(AttributedGraph *g) {
  return g->graph.size();
}

size_t getNumEdges(AttributedGraph *g) {
  return g->graph.sizeEdges();
}

//void setNodeAttr(AttributedGraph *g, GNode n, const KeyAltTy key, const ValAltTy val) {
//  g->getData(n).attr[key] = val;
//}
//
//const ValAltTy getNodeAttr(AttributedGraph *g, GNode n, const KeyAltTy key) {
//  return const_cast<ValAltTy>(g->getData(n).attr[key].c_str());
//}
//
//void removeNodeAttr(AttributedGraph *g, GNode n, const KeyAltTy key) {
//  g->getData(n).attr.erase(key);
//}

//void setEdgeAttr(AttributedGraph *g, Edge e, const KeyAltTy key, const ValAltTy val) {
//  auto ei = g->findEdge(e.src, e.dst);
//  assert(ei != g.edge_end(e.src));
//  g->getEdgeData(ei)[key] = val;
//}

//const ValAltTy getEdgeAttr(AttributedGraph *g, Edge e, const KeyAltTy key) {
//  auto ei = g->findEdge(e.src, e.dst);
//  assert(ei != g.edge_end(e.src));
//  return const_cast<ValAltTy>(g->getEdgeData(ei)[key].c_str());
//}

//void removeEdgeAttr(AttributedGraph *g, Edge e, const KeyAltTy key) {
//  auto ei = g->findEdge(e.src, e.dst);
//  assert(ei != g.edge_end(e.src));
//  g->getEdgeData(ei).erase(key);
//}

void runAttributedGraphSimulation(AttributedGraph* queryGraph, AttributedGraph* dataGraph, char* outputFile) {
  runGraphSimulation(queryGraph->graph, dataGraph->graph);
  if (outputFile != NULL) {
    reportGraphSimulation(queryGraph->graph, dataGraph->graph, outputFile);
  }
}

