#include "PythonGraph.h"

size_t runAttributedGraphSimulation(AttributedGraph* queryGraph, AttributedGraph* dataGraph) {
  runGraphSimulation(queryGraph->graph, dataGraph->graph);
  return countMatchedEdges(dataGraph->graph);
}

size_t findFilesWithMultipleWrites(AttributedGraph* dataGraph) {
  matchNodeWithRepeatedActions(dataGraph->graph,
      dataGraph->nodeLabelIDs["file"],
      dataGraph->edgeLabelIDs["write"]);
  return countMatchedEdges(dataGraph->graph);
}

size_t findProcessesWithReadFileWriteNetwork(AttributedGraph* dataGraph) {
  matchNodeWithTwoActions(dataGraph->graph,
      dataGraph->nodeLabelIDs["process"],
      dataGraph->edgeLabelIDs["read"],
      dataGraph->nodeLabelIDs["file"],
      dataGraph->edgeLabelIDs["write"],
      dataGraph->nodeLabelIDs["network"]);
  return countMatchedEdges(dataGraph->graph);
}

size_t findProcessesOriginatingFromNetwork(AttributedGraph* dataGraph) {
  Graph queryGraph;
  queryGraph.allocateFrom(4, 6);
  queryGraph.constructNodes();

  queryGraph.getData(0).label = dataGraph->nodeLabelIDs["network"];
  queryGraph.getData(0).id = 0;
  queryGraph.constructEdge(0, 1, EdgeData(dataGraph->edgeLabelIDs["read"], 0));
  queryGraph.fixEndEdge(0, 1);

  queryGraph.getData(1).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(1).id = 1;
  queryGraph.constructEdge(1, 0, EdgeData(dataGraph->edgeLabelIDs["read"], 0));
  queryGraph.constructEdge(2, 2, EdgeData(dataGraph->edgeLabelIDs["write"], 1));
  queryGraph.fixEndEdge(1, 3);

  queryGraph.getData(2).label = dataGraph->nodeLabelIDs["file"];
  queryGraph.getData(2).id = 2;
  queryGraph.constructEdge(3, 1, EdgeData(dataGraph->edgeLabelIDs["write"], 1));
  queryGraph.constructEdge(4, 3, EdgeData(dataGraph->edgeLabelIDs["execute"], 2));
  queryGraph.fixEndEdge(2, 5);

  queryGraph.getData(3).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(3).id = 3;
  queryGraph.constructEdge(5, 2, EdgeData(dataGraph->edgeLabelIDs["execute"], 2));
  queryGraph.fixEndEdge(3, 6);

  runGraphSimulation(queryGraph, dataGraph->graph);
  return countMatchedEdges(dataGraph->graph);
}

size_t findProcessesOriginatingFromNetworkIndirectly(AttributedGraph* dataGraph) {
  Graph queryGraph;
  queryGraph.allocateFrom(6, 10);
  queryGraph.constructNodes();

  queryGraph.getData(0).label = dataGraph->nodeLabelIDs["network"];
  queryGraph.getData(0).id = 0;
  queryGraph.constructEdge(0, 1, EdgeData(dataGraph->edgeLabelIDs["read"], 0));
  queryGraph.fixEndEdge(0, 1);

  queryGraph.getData(1).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(1).id = 1;
  queryGraph.constructEdge(1, 0, EdgeData(dataGraph->edgeLabelIDs["read"], 0));
  queryGraph.constructEdge(2, 2, EdgeData(dataGraph->edgeLabelIDs["write"], 1));
  queryGraph.fixEndEdge(1, 3);

  queryGraph.getData(2).label = dataGraph->nodeLabelIDs["file"];
  queryGraph.getData(2).id = 2;
  queryGraph.constructEdge(3, 1, EdgeData(dataGraph->edgeLabelIDs["write"], 1));
  queryGraph.constructEdge(4, 3, EdgeData(dataGraph->edgeLabelIDs["read"], 2));
  queryGraph.fixEndEdge(2, 5);

  queryGraph.getData(3).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(3).id = 3;
  queryGraph.constructEdge(5, 2, EdgeData(dataGraph->edgeLabelIDs["read"], 2));
  queryGraph.constructEdge(6, 4, EdgeData(dataGraph->edgeLabelIDs["write"], 3));
  queryGraph.fixEndEdge(3, 7);

  queryGraph.getData(4).label = dataGraph->nodeLabelIDs["file"];
  queryGraph.getData(4).id = 4;
  queryGraph.constructEdge(7, 3, EdgeData(dataGraph->edgeLabelIDs["write"], 3));
  queryGraph.constructEdge(8, 5, EdgeData(dataGraph->edgeLabelIDs["execute"], 4));
  queryGraph.fixEndEdge(4, 9);

  queryGraph.getData(5).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(5).id = 5;
  queryGraph.constructEdge(9, 4, EdgeData(dataGraph->edgeLabelIDs["execute"], 4));
  queryGraph.fixEndEdge(5, 10);

  runGraphSimulation(queryGraph, dataGraph->graph);
  return countMatchedEdges(dataGraph->graph);
}

size_t processesReadFromFile(AttributedGraph* dataGraph, uint32_t file_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[file_uuid],
      dataGraph->nodeLabelIDs["file"],
      dataGraph->edgeLabelIDs["read"],
      dataGraph->nodeLabelIDs["process"]);
  return countMatchedNeighborEdges(dataGraph->graph, file_uuid);
}

size_t processesWroteToFile(AttributedGraph* dataGraph, uint32_t file_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[file_uuid],
      dataGraph->nodeLabelIDs["file"],
      dataGraph->edgeLabelIDs["write"],
      dataGraph->nodeLabelIDs["process"]);
  return countMatchedNeighborEdges(dataGraph->graph, file_uuid);
}

size_t processesReadFromNetwork(AttributedGraph* dataGraph, uint32_t network_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[network_uuid],
      dataGraph->nodeLabelIDs["network"],
      dataGraph->edgeLabelIDs["read"],
      dataGraph->nodeLabelIDs["process"]);
  return countMatchedNeighborEdges(dataGraph->graph, network_uuid);
}

size_t processesWroteToNetwork(AttributedGraph* dataGraph, uint32_t network_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[network_uuid],
      dataGraph->nodeLabelIDs["network"],
      dataGraph->edgeLabelIDs["write"],
      dataGraph->nodeLabelIDs["process"]);
  return countMatchedNeighborEdges(dataGraph->graph, network_uuid);
}

size_t processesReadFromRegistry(AttributedGraph* dataGraph, uint32_t registry_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[registry_uuid],
      dataGraph->nodeLabelIDs["registry"],
      dataGraph->edgeLabelIDs["read"],
      dataGraph->nodeLabelIDs["process"]);
  return countMatchedNeighborEdges(dataGraph->graph, registry_uuid);
}

size_t processesWroteToRegistry(AttributedGraph* dataGraph, uint32_t registry_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[registry_uuid],
      dataGraph->nodeLabelIDs["registry"],
      dataGraph->edgeLabelIDs["write"],
      dataGraph->nodeLabelIDs["process"]);
  return countMatchedNeighborEdges(dataGraph->graph, registry_uuid);
}

size_t processesReadFromMemory(AttributedGraph* dataGraph, uint32_t memory_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[memory_uuid],
      dataGraph->nodeLabelIDs["memory"],
      dataGraph->edgeLabelIDs["read"],
      dataGraph->nodeLabelIDs["process"]);
  return countMatchedNeighborEdges(dataGraph->graph, memory_uuid);
}

size_t processesWroteToMemory(AttributedGraph* dataGraph, uint32_t memory_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[memory_uuid],
      dataGraph->nodeLabelIDs["memory"],
      dataGraph->edgeLabelIDs["write"],
      dataGraph->nodeLabelIDs["process"]);
  return countMatchedNeighborEdges(dataGraph->graph, memory_uuid);
}

size_t filesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[process_uuid],
      dataGraph->nodeLabelIDs["process"],
      dataGraph->edgeLabelIDs["read"],
      dataGraph->nodeLabelIDs["file"]);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t filesWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[process_uuid],
      dataGraph->nodeLabelIDs["process"],
      dataGraph->edgeLabelIDs["write"],
      dataGraph->nodeLabelIDs["file"]);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t networksReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[process_uuid],
      dataGraph->nodeLabelIDs["process"],
      dataGraph->edgeLabelIDs["read"],
      dataGraph->nodeLabelIDs["network"]);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t networksWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[process_uuid],
      dataGraph->nodeLabelIDs["process"],
      dataGraph->edgeLabelIDs["write"],
      dataGraph->nodeLabelIDs["network"]);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t registriesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[process_uuid],
      dataGraph->nodeLabelIDs["process"],
      dataGraph->edgeLabelIDs["read"],
      dataGraph->nodeLabelIDs["registry"]);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t registriesWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[process_uuid],
      dataGraph->nodeLabelIDs["process"],
      dataGraph->edgeLabelIDs["write"],
      dataGraph->nodeLabelIDs["registry"]);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t memoriesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[process_uuid],
      dataGraph->nodeLabelIDs["process"],
      dataGraph->edgeLabelIDs["read"],
      dataGraph->nodeLabelIDs["memory"]);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t memoriesWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid) {
  matchNeighbors(dataGraph->graph,
      dataGraph->nodeIndices[process_uuid],
      dataGraph->nodeLabelIDs["process"],
      dataGraph->edgeLabelIDs["write"],
      dataGraph->nodeLabelIDs["memory"]);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}
