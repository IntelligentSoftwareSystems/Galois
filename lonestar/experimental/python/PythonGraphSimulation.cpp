/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "PythonGraph.h"

size_t runAttributedGraphSimulation(AttributedGraph* queryGraph,
                                    AttributedGraph* dataGraph,
                                    EventLimit limit, EventWindow window) {
  runGraphSimulation(queryGraph->graph, dataGraph->graph, limit, window, true);
  return countMatchedEdges(dataGraph->graph);
}

size_t findFilesWithMultipleWrites(AttributedGraph* dataGraph,
                                   EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNodeWithRepeatedActions(dataGraph->graph,
                               dataGraph->nodeLabelIDs["file"],
                               dataGraph->edgeLabelIDs["WRITE"], window);
  return countMatchedEdges(dataGraph->graph);
}

size_t findProcessesWithReadFileWriteNetwork(AttributedGraph* dataGraph,
                                             EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("network") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNodeWithTwoActions(dataGraph->graph, dataGraph->nodeLabelIDs["process"],
                          dataGraph->edgeLabelIDs["READ"],
                          dataGraph->nodeLabelIDs["file"],
                          dataGraph->edgeLabelIDs["WRITE"],
                          dataGraph->nodeLabelIDs["network"], window);
  return countMatchedEdges(dataGraph->graph);
}

size_t findProcessesWritingNetworkIndirectly(AttributedGraph* dataGraph,
                                             EventLimit limit,
                                             EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("network") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  Graph queryGraph;
  queryGraph.allocateFrom(4, 6);
  queryGraph.constructNodes();

  queryGraph.getData(0).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(0).id    = 0;
  queryGraph.constructEdge(0, 1, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 0));
  queryGraph.fixEndEdge(0, 1);

  queryGraph.getData(1).label = dataGraph->nodeLabelIDs["file"];
  queryGraph.getData(1).id    = 1;
  queryGraph.constructEdge(1, 0, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 0));
  queryGraph.constructEdge(2, 2, EdgeData(dataGraph->edgeLabelIDs["READ"], 1));
  queryGraph.fixEndEdge(1, 3);

  queryGraph.getData(2).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(2).id    = 2;
  queryGraph.constructEdge(3, 1, EdgeData(dataGraph->edgeLabelIDs["READ"], 1));
  queryGraph.constructEdge(4, 3, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 2));
  queryGraph.fixEndEdge(2, 5);

  queryGraph.getData(3).label = dataGraph->nodeLabelIDs["network"];
  queryGraph.getData(3).id    = 3;
  queryGraph.constructEdge(5, 2, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 2));
  queryGraph.fixEndEdge(3, 6);

  runGraphSimulation(queryGraph, dataGraph->graph, limit, window, false);
  return countMatchedEdges(dataGraph->graph);
}

size_t findProcessesOriginatingFromNetwork(AttributedGraph* dataGraph,
                                           EventLimit limit,
                                           EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("network") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("EXECUTE") ==
       dataGraph->edgeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  Graph queryGraph;
  queryGraph.allocateFrom(4, 6);
  queryGraph.constructNodes();

  queryGraph.getData(0).label = dataGraph->nodeLabelIDs["network"];
  queryGraph.getData(0).id    = 0;
  queryGraph.constructEdge(0, 1, EdgeData(dataGraph->edgeLabelIDs["READ"], 0));
  queryGraph.fixEndEdge(0, 1);

  queryGraph.getData(1).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(1).id    = 1;
  queryGraph.constructEdge(1, 0, EdgeData(dataGraph->edgeLabelIDs["READ"], 0));
  queryGraph.constructEdge(2, 2, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 1));
  queryGraph.fixEndEdge(1, 3);

  queryGraph.getData(2).label = dataGraph->nodeLabelIDs["file"];
  queryGraph.getData(2).id    = 2;
  queryGraph.constructEdge(3, 1, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 1));
  queryGraph.constructEdge(4, 3,
                           EdgeData(dataGraph->edgeLabelIDs["EXECUTE"], 2));
  queryGraph.fixEndEdge(2, 5);

  queryGraph.getData(3).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(3).id    = 3;
  queryGraph.constructEdge(5, 2,
                           EdgeData(dataGraph->edgeLabelIDs["EXECUTE"], 2));
  queryGraph.fixEndEdge(3, 6);

  runGraphSimulation(queryGraph, dataGraph->graph, limit, window, false);
  return countMatchedEdges(dataGraph->graph);
}

size_t findProcessesOriginatingFromNetworkIndirectly(AttributedGraph* dataGraph,
                                                     EventLimit limit,
                                                     EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("network") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("EXECUTE") ==
       dataGraph->edgeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  Graph queryGraph;
  queryGraph.allocateFrom(6, 10);
  queryGraph.constructNodes();

  queryGraph.getData(0).label = dataGraph->nodeLabelIDs["network"];
  queryGraph.getData(0).id    = 0;
  queryGraph.constructEdge(0, 1, EdgeData(dataGraph->edgeLabelIDs["READ"], 0));
  queryGraph.fixEndEdge(0, 1);

  queryGraph.getData(1).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(1).id    = 1;
  queryGraph.constructEdge(1, 0, EdgeData(dataGraph->edgeLabelIDs["READ"], 0));
  queryGraph.constructEdge(2, 2, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 1));
  queryGraph.fixEndEdge(1, 3);

  queryGraph.getData(2).label = dataGraph->nodeLabelIDs["file"];
  queryGraph.getData(2).id    = 2;
  queryGraph.constructEdge(3, 1, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 1));
  queryGraph.constructEdge(4, 3, EdgeData(dataGraph->edgeLabelIDs["READ"], 2));
  queryGraph.fixEndEdge(2, 5);

  queryGraph.getData(3).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(3).id    = 3;
  queryGraph.constructEdge(5, 2, EdgeData(dataGraph->edgeLabelIDs["READ"], 2));
  queryGraph.constructEdge(6, 4, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 3));
  queryGraph.fixEndEdge(3, 7);

  queryGraph.getData(4).label = dataGraph->nodeLabelIDs["file"];
  queryGraph.getData(4).id    = 4;
  queryGraph.constructEdge(7, 3, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 3));
  queryGraph.constructEdge(8, 5,
                           EdgeData(dataGraph->edgeLabelIDs["EXECUTE"], 4));
  queryGraph.fixEndEdge(4, 9);

  queryGraph.getData(5).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(5).id    = 5;
  queryGraph.constructEdge(9, 4,
                           EdgeData(dataGraph->edgeLabelIDs["EXECUTE"], 4));
  queryGraph.fixEndEdge(5, 10);

  runGraphSimulation(queryGraph, dataGraph->graph, limit, window, false);
  return countMatchedEdges(dataGraph->graph);
}

size_t findProcessesExecutingModifiedFile(AttributedGraph* dataGraph,
                                          EventLimit limit,
                                          EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("CHMOD") ==
       dataGraph->edgeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("EXECUTE") ==
       dataGraph->edgeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  Graph queryGraph;
  queryGraph.allocateFrom(4, 6);
  queryGraph.constructNodes();

  queryGraph.getData(0).label = dataGraph->nodeLabelIDs["file"];
  queryGraph.getData(0).id    = 0;
  queryGraph.constructEdge(0, 1, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 0));
  queryGraph.constructEdge(1, 2, EdgeData(dataGraph->edgeLabelIDs["CHMOD"], 1));
  queryGraph.constructEdge(2, 3,
                           EdgeData(dataGraph->edgeLabelIDs["EXECUTE"], 2));
  queryGraph.fixEndEdge(0, 3);

  queryGraph.getData(1).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(1).id    = 1;
  queryGraph.constructEdge(3, 0, EdgeData(dataGraph->edgeLabelIDs["WRITE"], 0));
  queryGraph.fixEndEdge(1, 4);

  queryGraph.getData(2).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(2).id    = 2;
  queryGraph.constructEdge(4, 0, EdgeData(dataGraph->edgeLabelIDs["CHMOD"], 1));
  queryGraph.fixEndEdge(2, 5);

  queryGraph.getData(3).label = dataGraph->nodeLabelIDs["process"];
  queryGraph.getData(3).id    = 3;
  queryGraph.constructEdge(5, 0,
                           EdgeData(dataGraph->edgeLabelIDs["EXECUTE"], 2));
  queryGraph.fixEndEdge(3, 6);

  runGraphSimulation(queryGraph, dataGraph->graph, limit, window, true);
  return countMatchedEdges(dataGraph->graph);
}

size_t processesReadFromFile(AttributedGraph* dataGraph, uint32_t file_uuid,
                             EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[file_uuid],
                 dataGraph->nodeLabelIDs["file"],
                 dataGraph->edgeLabelIDs["READ"],
                 dataGraph->nodeLabelIDs["process"], window);
  return countMatchedNeighborEdges(dataGraph->graph, file_uuid);
}

size_t processesWroteToFile(AttributedGraph* dataGraph, uint32_t file_uuid,
                            EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[file_uuid],
                 dataGraph->nodeLabelIDs["file"],
                 dataGraph->edgeLabelIDs["WRITE"],
                 dataGraph->nodeLabelIDs["process"], window);
  return countMatchedNeighborEdges(dataGraph->graph, file_uuid);
}

size_t processesReadFromNetwork(AttributedGraph* dataGraph,
                                uint32_t network_uuid, EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("network") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[network_uuid],
                 dataGraph->nodeLabelIDs["network"],
                 dataGraph->edgeLabelIDs["READ"],
                 dataGraph->nodeLabelIDs["process"], window);
  return countMatchedNeighborEdges(dataGraph->graph, network_uuid);
}

size_t processesWroteToNetwork(AttributedGraph* dataGraph,
                               uint32_t network_uuid, EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("network") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[network_uuid],
                 dataGraph->nodeLabelIDs["network"],
                 dataGraph->edgeLabelIDs["WRITE"],
                 dataGraph->nodeLabelIDs["process"], window);
  return countMatchedNeighborEdges(dataGraph->graph, network_uuid);
}

size_t processesReadFromRegistry(AttributedGraph* dataGraph,
                                 uint32_t registry_uuid, EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("registry") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[registry_uuid],
                 dataGraph->nodeLabelIDs["registry"],
                 dataGraph->edgeLabelIDs["READ"],
                 dataGraph->nodeLabelIDs["process"], window);
  return countMatchedNeighborEdges(dataGraph->graph, registry_uuid);
}

size_t processesWroteToRegistry(AttributedGraph* dataGraph,
                                uint32_t registry_uuid, EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("registry") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[registry_uuid],
                 dataGraph->nodeLabelIDs["registry"],
                 dataGraph->edgeLabelIDs["WRITE"],
                 dataGraph->nodeLabelIDs["process"], window);
  return countMatchedNeighborEdges(dataGraph->graph, registry_uuid);
}

size_t processesReadFromMemory(AttributedGraph* dataGraph, uint32_t memory_uuid,
                               EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("memory") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[memory_uuid],
                 dataGraph->nodeLabelIDs["memory"],
                 dataGraph->edgeLabelIDs["READ"],
                 dataGraph->nodeLabelIDs["process"], window);
  return countMatchedNeighborEdges(dataGraph->graph, memory_uuid);
}

size_t processesWroteToMemory(AttributedGraph* dataGraph, uint32_t memory_uuid,
                              EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("memory") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[memory_uuid],
                 dataGraph->nodeLabelIDs["memory"],
                 dataGraph->edgeLabelIDs["WRITE"],
                 dataGraph->nodeLabelIDs["process"], window);
  return countMatchedNeighborEdges(dataGraph->graph, memory_uuid);
}

size_t filesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid,
                          EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[process_uuid],
                 dataGraph->nodeLabelIDs["process"],
                 dataGraph->edgeLabelIDs["READ"],
                 dataGraph->nodeLabelIDs["file"], window);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t filesWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid,
                             EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("file") == dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[process_uuid],
                 dataGraph->nodeLabelIDs["process"],
                 dataGraph->edgeLabelIDs["WRITE"],
                 dataGraph->nodeLabelIDs["file"], window);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t networksReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid,
                             EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("network") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[process_uuid],
                 dataGraph->nodeLabelIDs["process"],
                 dataGraph->edgeLabelIDs["READ"],
                 dataGraph->nodeLabelIDs["network"], window);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t networksWrittenByProcess(AttributedGraph* dataGraph,
                                uint32_t process_uuid, EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("network") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[process_uuid],
                 dataGraph->nodeLabelIDs["process"],
                 dataGraph->edgeLabelIDs["WRITE"],
                 dataGraph->nodeLabelIDs["network"], window);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t registriesReadByProcess(AttributedGraph* dataGraph,
                               uint32_t process_uuid, EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("registry") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[process_uuid],
                 dataGraph->nodeLabelIDs["process"],
                 dataGraph->edgeLabelIDs["READ"],
                 dataGraph->nodeLabelIDs["registry"], window);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t registriesWrittenByProcess(AttributedGraph* dataGraph,
                                  uint32_t process_uuid, EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("registry") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[process_uuid],
                 dataGraph->nodeLabelIDs["process"],
                 dataGraph->edgeLabelIDs["WRITE"],
                 dataGraph->nodeLabelIDs["registry"], window);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t memoriesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid,
                             EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("memory") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("READ") == dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[process_uuid],
                 dataGraph->nodeLabelIDs["process"],
                 dataGraph->edgeLabelIDs["READ"],
                 dataGraph->nodeLabelIDs["memory"], window);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}

size_t memoriesWrittenByProcess(AttributedGraph* dataGraph,
                                uint32_t process_uuid, EventWindow window) {
  if ((dataGraph->nodeLabelIDs.find("process") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->nodeLabelIDs.find("memory") ==
       dataGraph->nodeLabelIDs.end()) ||
      (dataGraph->edgeLabelIDs.find("WRITE") ==
       dataGraph->edgeLabelIDs.end())) {
    return 0;
  }

  matchNeighbors(dataGraph->graph, dataGraph->nodeIndices[process_uuid],
                 dataGraph->nodeLabelIDs["process"],
                 dataGraph->edgeLabelIDs["WRITE"],
                 dataGraph->nodeLabelIDs["memory"], window);
  return countMatchedNeighborEdges(dataGraph->graph, process_uuid);
}
