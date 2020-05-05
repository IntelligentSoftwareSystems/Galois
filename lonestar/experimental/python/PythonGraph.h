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

#ifndef GALOIS_PYTHON_GRAPH_H
#define GALOIS_PYTHON_GRAPH_H
#include "../graphsimulation/GraphSimulation.h"

extern "C" {

void initGaloisRuntime();
void setNumThreads(int numThreads);
int getNumThreads();

AttributedGraph* createGraph();
void deleteGraph(AttributedGraph* g);
void saveGraph(AttributedGraph* g, char* filename);
void loadGraph(AttributedGraph* g, char* filename);
void printGraph(AttributedGraph* g);

void allocateGraph(AttributedGraph* g, size_t numNodes, size_t numEdges,
                   size_t numNodeLabels, size_t numEdgeLabels);
void fixEndEdge(AttributedGraph* g, uint32_t nodeIndex, uint64_t edgeIndex);
void setNode(AttributedGraph* g, uint32_t nodeIndex, uint32_t uuid,
             uint32_t label, char* name);
void setNodeLabel(AttributedGraph* g, uint32_t label, char* name);
void setEdgeLabel(AttributedGraph* g, uint32_t label, char* name);
void setNodeAttribute(AttributedGraph* g, uint32_t nodeIndex, char* key,
                      char* value);
void constructEdge(AttributedGraph* g, uint64_t edgeIndex,
                   uint32_t dstNodeIndex, uint32_t label, uint64_t timestamp);
void setEdgeAttribute(AttributedGraph* g, uint32_t edgeIndex, char* key,
                      char* value);

size_t getNumNodes(AttributedGraph* g);
size_t getNumEdges(AttributedGraph* g);

size_t runAttributedGraphSimulation(AttributedGraph* queryGraph,
                                    AttributedGraph* dataGraph,
                                    EventLimit limit, EventWindow window);

size_t findFilesWithMultipleWrites(AttributedGraph* dataGraph,
                                   EventWindow window);
size_t findProcessesWithReadFileWriteNetwork(AttributedGraph* dataGraph,
                                             EventWindow window);
size_t findProcessesWritingNetworkIndirectly(AttributedGraph* dataGraph,
                                             EventLimit limit,
                                             EventWindow window);
size_t findProcessesOriginatingFromNetwork(AttributedGraph* dataGraph,
                                           EventLimit limit,
                                           EventWindow window);
size_t findProcessesOriginatingFromNetworkIndirectly(AttributedGraph* dataGraph,
                                                     EventLimit limit,
                                                     EventWindow window);
size_t findProcessesExecutingModifiedFile(AttributedGraph* dataGraph,
                                          EventLimit limit, EventWindow window);

size_t processesReadFromFile(AttributedGraph* dataGraph, uint32_t file_uuid,
                             EventWindow window);
size_t processesWroteToFile(AttributedGraph* dataGraph, uint32_t file_uuid,
                            EventWindow window);
size_t processesReadFromNetwork(AttributedGraph* dataGraph,
                                uint32_t network_uuid, EventWindow window);
size_t processesWroteToNetwork(AttributedGraph* dataGraph,
                               uint32_t network_uuid, EventWindow window);
size_t processesReadFromRegistry(AttributedGraph* dataGraph,
                                 uint32_t registry_uuid, EventWindow window);
size_t processesWroteToRegistry(AttributedGraph* dataGraph,
                                uint32_t registry_uuid, EventWindow window);
size_t processesReadFromMemory(AttributedGraph* dataGraph, uint32_t memory_uuid,
                               EventWindow window);
size_t processesWroteToMemory(AttributedGraph* dataGraph, uint32_t memory_uuid,
                              EventWindow window);

size_t filesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid,
                          EventWindow window);
size_t filesWrittenByProcess(AttributedGraph* dataGraph, uint32_t process_uuid,
                             EventWindow window);
size_t networksReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid,
                             EventWindow window);
size_t networksWrittenByProcess(AttributedGraph* dataGraph,
                                uint32_t process_uuid, EventWindow window);
size_t registriesReadByProcess(AttributedGraph* dataGraph,
                               uint32_t process_uuid, EventWindow window);
size_t registriesWrittenByProcess(AttributedGraph* dataGraph,
                                  uint32_t process_uuid, EventWindow window);
size_t memoriesReadByProcess(AttributedGraph* dataGraph, uint32_t process_uuid,
                             EventWindow window);
size_t memoriesWrittenByProcess(AttributedGraph* dataGraph,
                                uint32_t process_uuid, EventWindow window);

void reportGraphSimulation(AttributedGraph& queryGraph,
                           AttributedGraph& dataGraph, char* outputFile);

void returnMatchedNodes(AttributedGraph& graph, MatchedNode* matchedNodes);
void reportMatchedNodes(AttributedGraph& graph, char* outputFile);
void returnMatchedNeighbors(AttributedGraph& graph, uint32_t uuid,
                            MatchedNode* matchedNeighbors);
void reportMatchedNeighbors(AttributedGraph& graph, uint32_t uuid,
                            char* outputFile);
void returnMatchedEdges(AttributedGraph& graph, MatchedEdge* matchedEdges);
void reportMatchedEdges(AttributedGraph& graph, char* outputFile);
void returnMatchedNeighborEdges(AttributedGraph& graph, uint32_t uuid,
                                MatchedEdge* matchedEdges);
void reportMatchedNeighborEdges(AttributedGraph& graph, uint32_t uuid,
                                char* outputFile);

} // extern "C"

#endif // GALOIS_PYTHON_GRAPH_H
