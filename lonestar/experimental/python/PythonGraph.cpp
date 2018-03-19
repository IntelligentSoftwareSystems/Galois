#include "PythonGraph.h"

#include <iostream>
#include <fstream>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

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

void saveGraph(AttributedGraph *g, char* filename) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  boost::archive::binary_oarchive oarch(file);
  g->graph.serializeGraph(oarch);
  oarch << g->nodeLabels;
  oarch << g->nodeIDs;
  oarch << g->edgeLabels;
  oarch << g->edgeIDs;
  oarch << g->nodeNames;
  size_t size = g->nodeAttributes.size();
  oarch << size;
  for (auto& pair : g->nodeAttributes) {
    oarch << pair.first;
    oarch << pair.second;
  }
  size = g->edgeAttributes.size();
  oarch << size;
  for (auto& pair : g->edgeAttributes) {
    oarch << pair.first;
    oarch << pair.second;
  }
  file.close();
}

void loadGraph(AttributedGraph *g, char* filename) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  boost::archive::binary_iarchive iarch(file);
  g->graph.deSerializeGraph(iarch);
  iarch >> g->nodeLabels;
  iarch >> g->nodeIDs;
  iarch >> g->edgeLabels;
  iarch >> g->edgeIDs;
  iarch >> g->nodeNames;
  size_t size;
  iarch >> size;
  for (size_t i = 0; i < size; ++i) {
    std::string key;
    iarch >> key;
    g->nodeAttributes[key] = std::vector<std::string>();
    iarch >> g->nodeAttributes[key];
  }
  iarch >> size;
  for (size_t i = 0; i < size; ++i) {
    std::string key;
    iarch >> key;
    g->edgeAttributes[key] = std::vector<std::string>();
    iarch >> g->edgeAttributes[key];
  }
  file.close();
}

void printGraph(AttributedGraph* g) {
  Graph& graph = g->graph;
  auto& nodeLabels = g->nodeLabels;
  auto& edgeLabels = g->edgeLabels;
  auto& nodeNames = g->nodeNames;
  for(auto src: graph) {
    auto& srcData = graph.getData(src);
    auto& srcLabel = nodeLabels[srcData.label];
    auto& srcName = nodeNames[src];
    for(auto e: graph.edges(src)) {
      auto dst = graph.getEdgeDst(e);
      auto& dstData = graph.getData(dst);
      auto& dstLabel = nodeLabels[dstData.label];
      auto& dstName = nodeNames[dst];
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
  assert(numNodeLabels <= 32);
  g->nodeLabels.resize(numNodeLabels);
  assert(numEdgeLabels <= 32);
  g->edgeLabels.resize(numEdgeLabels);
  g->nodeNames.resize(numNodes);
}

void fixEndEdge(AttributedGraph *g, uint32_t nodeIndex, uint64_t edgeIndex) {
  g->graph.fixEndEdge(nodeIndex, edgeIndex);
}

void setNode(AttributedGraph *g, uint32_t nodeIndex, uint32_t uuid, uint32_t label, char *nodeName) {
  auto& nd = g->graph.getData(nodeIndex);
  nd.label = label;
  nd.id = uuid;
  g->nodeNames[nodeIndex] = nodeName;
}

void setNodeLabel(AttributedGraph *g, uint32_t label, char *name) {
  g->nodeLabels[label] = name;
  g->nodeIDs[name] = label;
}

void setEdgeLabel(AttributedGraph *g, uint32_t label, char *name) {
  g->edgeLabels[label] = name;
  g->edgeIDs[name] = label;
}

void setNodeAttribute(AttributedGraph *g, uint32_t nodeIndex, char *key, char *value) {
  auto& attributes = g->nodeAttributes;
  if (attributes.find(key) == attributes.end()) {
    attributes[key] = std::vector<std::string>();
    attributes[key].resize(g->graph.size());
  }
  attributes[key][nodeIndex] = value;
}

void constructEdge(AttributedGraph *g, uint64_t edgeIndex, uint32_t dstNodeIndex, uint32_t label, uint64_t timestamp) {
  g->graph.constructEdge(edgeIndex, dstNodeIndex, EdgeData(label, timestamp));
}

void setEdgeAttribute(AttributedGraph *g, uint32_t edgeIndex, char *key, char *value) {
  auto& attributes = g->edgeAttributes;
  if (attributes.find(key) == attributes.end()) {
    attributes[key] = std::vector<std::string>();
    attributes[key].resize(g->graph.sizeEdges());
  }
  attributes[key][edgeIndex] = value;
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

size_t runAttributedGraphSimulation(AttributedGraph* queryGraph, AttributedGraph* dataGraph) {
  runGraphSimulation(queryGraph->graph, dataGraph->graph);
  return countMatchedNodes(dataGraph->graph);
}

size_t findFilesWithMultipleWrites(AttributedGraph* dataGraph) {
  matchNodeWithRepeatedActions(dataGraph->graph,
      dataGraph->nodeIDs["file"],
      dataGraph->edgeIDs["write"]);
  return countMatchedNodes(dataGraph->graph);
}

size_t findProcessesWithReadFileWriteNetflow(AttributedGraph* dataGraph) {
  matchNodeWithTwoActions(dataGraph->graph,
      dataGraph->nodeIDs["process"],
      dataGraph->edgeIDs["read"],
      dataGraph->nodeIDs["file"],
      dataGraph->edgeIDs["write"],
      dataGraph->nodeIDs["netflow"]);
  return countMatchedNodes(dataGraph->graph);
}

size_t findProcessesOriginatingFromNetflow(AttributedGraph* dataGraph) {
  Graph queryGraph;
  queryGraph.allocateFrom(4, 6);
  queryGraph.constructNodes();

  queryGraph.getData(0).label = dataGraph->nodeIDs["netflow"];
  queryGraph.getData(0).id = 0;
  queryGraph.constructEdge(0, 1, EdgeData(dataGraph->edgeIDs["read"], 0));
  queryGraph.fixEndEdge(0, 1);

  queryGraph.getData(1).label = dataGraph->nodeIDs["process"];
  queryGraph.getData(1).id = 1;
  queryGraph.constructEdge(1, 0, EdgeData(dataGraph->edgeIDs["read"], 0));
  queryGraph.constructEdge(2, 2, EdgeData(dataGraph->edgeIDs["write"], 1));
  queryGraph.fixEndEdge(1, 3);

  queryGraph.getData(2).label = dataGraph->nodeIDs["file"];
  queryGraph.getData(2).id = 2;
  queryGraph.constructEdge(3, 1, EdgeData(dataGraph->edgeIDs["write"], 1));
  queryGraph.constructEdge(4, 3, EdgeData(dataGraph->edgeIDs["execute"], 2));
  queryGraph.fixEndEdge(2, 5);

  queryGraph.getData(3).label = dataGraph->nodeIDs["process"];
  queryGraph.getData(3).id = 3;
  queryGraph.constructEdge(5, 2, EdgeData(dataGraph->edgeIDs["execute"], 2));
  queryGraph.fixEndEdge(3, 6);

  runGraphSimulation(queryGraph, dataGraph->graph);
  return countMatchedNodes(dataGraph->graph);
}

size_t findProcessesOriginatingFromNetflowIndirectly(AttributedGraph* dataGraph) {
  Graph queryGraph;
  queryGraph.allocateFrom(6, 10);
  queryGraph.constructNodes();

  queryGraph.getData(0).label = dataGraph->nodeIDs["netflow"];
  queryGraph.getData(0).id = 0;
  queryGraph.constructEdge(0, 1, EdgeData(dataGraph->edgeIDs["read"], 0));
  queryGraph.fixEndEdge(0, 1);

  queryGraph.getData(1).label = dataGraph->nodeIDs["process"];
  queryGraph.getData(1).id = 1;
  queryGraph.constructEdge(1, 0, EdgeData(dataGraph->edgeIDs["read"], 0));
  queryGraph.constructEdge(2, 2, EdgeData(dataGraph->edgeIDs["write"], 1));
  queryGraph.fixEndEdge(1, 3);

  queryGraph.getData(2).label = dataGraph->nodeIDs["file"];
  queryGraph.getData(2).id = 2;
  queryGraph.constructEdge(3, 1, EdgeData(dataGraph->edgeIDs["write"], 1));
  queryGraph.constructEdge(4, 3, EdgeData(dataGraph->edgeIDs["read"], 2));
  queryGraph.fixEndEdge(2, 5);

  queryGraph.getData(3).label = dataGraph->nodeIDs["process"];
  queryGraph.getData(3).id = 3;
  queryGraph.constructEdge(5, 2, EdgeData(dataGraph->edgeIDs["read"], 2));
  queryGraph.constructEdge(6, 4, EdgeData(dataGraph->edgeIDs["write"], 3));
  queryGraph.fixEndEdge(3, 7);

  queryGraph.getData(4).label = dataGraph->nodeIDs["file"];
  queryGraph.getData(4).id = 4;
  queryGraph.constructEdge(7, 3, EdgeData(dataGraph->edgeIDs["write"], 3));
  queryGraph.constructEdge(8, 5, EdgeData(dataGraph->edgeIDs["execute"], 4));
  queryGraph.fixEndEdge(4, 9);

  queryGraph.getData(5).label = dataGraph->nodeIDs["process"];
  queryGraph.getData(5).id = 5;
  queryGraph.constructEdge(9, 4, EdgeData(dataGraph->edgeIDs["execute"], 4));
  queryGraph.fixEndEdge(5, 10);

  runGraphSimulation(queryGraph, dataGraph->graph);
  return countMatchedNodes(dataGraph->graph);
}

size_t processesReadFromFile(AttributedGraph* dataGraph, uint32_t file_uuid) {
  matchNeighbors(dataGraph->graph,
      file_uuid,
      dataGraph->nodeIDs["file"],
      dataGraph->edgeIDs["read"],
      dataGraph->nodeIDs["process"]);
  return countMatchedNeighbors(dataGraph->graph, file_uuid);
}

size_t processesWroteToFile(AttributedGraph* dataGraph, uint32_t file_uuid) {
  matchNeighbors(dataGraph->graph,
      file_uuid,
      dataGraph->nodeIDs["file"],
      dataGraph->edgeIDs["write"],
      dataGraph->nodeIDs["process"]);
  return countMatchedNeighbors(dataGraph->graph, file_uuid);
}

size_t processesReadFromNetflow(AttributedGraph* dataGraph, uint32_t netflow_uuid) {
  matchNeighbors(dataGraph->graph,
      netflow_uuid,
      dataGraph->nodeIDs["netflow"],
      dataGraph->edgeIDs["read"],
      dataGraph->nodeIDs["process"]);
  return countMatchedNeighbors(dataGraph->graph, netflow_uuid);
}

size_t processesWroteToNetflow(AttributedGraph* dataGraph, uint32_t netflow_uuid) {
  matchNeighbors(dataGraph->graph,
      netflow_uuid,
      dataGraph->nodeIDs["netflow"],
      dataGraph->edgeIDs["write"],
      dataGraph->nodeIDs["process"]);
  return countMatchedNeighbors(dataGraph->graph, netflow_uuid);
}

size_t processesReadFromRegistry(AttributedGraph* dataGraph, uint32_t registry_uuid) {
  matchNeighbors(dataGraph->graph,
      registry_uuid,
      dataGraph->nodeIDs["registry"],
      dataGraph->edgeIDs["read"],
      dataGraph->nodeIDs["process"]);
  return countMatchedNeighbors(dataGraph->graph, registry_uuid);
}

size_t processesWroteToRegistry(AttributedGraph* dataGraph, uint32_t registry_uuid) {
  matchNeighbors(dataGraph->graph,
      registry_uuid,
      dataGraph->nodeIDs["registry"],
      dataGraph->edgeIDs["write"],
      dataGraph->nodeIDs["process"]);
  return countMatchedNeighbors(dataGraph->graph, registry_uuid);
}

size_t processesReadFromMemory(AttributedGraph* dataGraph, uint32_t memory_uuid) {
  matchNeighbors(dataGraph->graph,
      memory_uuid,
      dataGraph->nodeIDs["memory"],
      dataGraph->edgeIDs["read"],
      dataGraph->nodeIDs["process"]);
  return countMatchedNeighbors(dataGraph->graph, memory_uuid);
}

size_t processesWroteToMemory(AttributedGraph* dataGraph, uint32_t memory_uuid) {
  matchNeighbors(dataGraph->graph,
      memory_uuid,
      dataGraph->nodeIDs["memory"],
      dataGraph->edgeIDs["write"],
      dataGraph->nodeIDs["process"]);
  return countMatchedNeighbors(dataGraph->graph, memory_uuid);
}

size_t filesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      process_uuid,
      dataGraph->nodeIDs["process"],
      dataGraph->edgeIDs["read"],
      dataGraph->nodeIDs["file"]);
  return countMatchedNeighbors(dataGraph->graph, process_uuid);
}

size_t filesWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      process_uuid,
      dataGraph->nodeIDs["process"],
      dataGraph->edgeIDs["write"],
      dataGraph->nodeIDs["file"]);
  return countMatchedNeighbors(dataGraph->graph, process_uuid);
}

size_t netflowsReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      process_uuid,
      dataGraph->nodeIDs["process"],
      dataGraph->edgeIDs["read"],
      dataGraph->nodeIDs["netflow"]);
  return countMatchedNeighbors(dataGraph->graph, process_uuid);
}

size_t netflowsWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      process_uuid,
      dataGraph->nodeIDs["process"],
      dataGraph->edgeIDs["write"],
      dataGraph->nodeIDs["netflow"]);
  return countMatchedNeighbors(dataGraph->graph, process_uuid);
}

size_t registriesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      process_uuid,
      dataGraph->nodeIDs["process"],
      dataGraph->edgeIDs["read"],
      dataGraph->nodeIDs["registry"]);
  return countMatchedNeighbors(dataGraph->graph, process_uuid);
}

size_t registriesWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      process_uuid,
      dataGraph->nodeIDs["process"],
      dataGraph->edgeIDs["write"],
      dataGraph->nodeIDs["registry"]);
  return countMatchedNeighbors(dataGraph->graph, process_uuid);
}

size_t memoriesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      process_uuid,
      dataGraph->nodeIDs["process"],
      dataGraph->edgeIDs["read"],
      dataGraph->nodeIDs["memory"]);
  return countMatchedNeighbors(dataGraph->graph, process_uuid);
}

size_t memoriesWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      process_uuid,
      dataGraph->nodeIDs["process"],
      dataGraph->edgeIDs["write"],
      dataGraph->nodeIDs["memory"]);
  return countMatchedNeighbors(dataGraph->graph, process_uuid);
}

