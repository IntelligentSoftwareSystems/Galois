/** Subgraph isomorphism -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Python Graph.
 *
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */

#ifndef GALOIS_PYTHON_GRAPH_H
#define GALOIS_PYTHON_GRAPH_H

#include "../graphsimulation/GraphSimulation.h"

extern "C" {

void initGaloisRuntime();
void setNumThreads(int numThreads);
int getNumThreads();

AttributedGraph *createGraph();
void deleteGraph(AttributedGraph *g);
void saveGraph(AttributedGraph *g, char* filename);
void loadGraph(AttributedGraph *g, char* filename);
void printGraph(AttributedGraph *g);

void allocateGraph(AttributedGraph *g, size_t numNodes, size_t numEdges, size_t numNodeLabels, size_t numEdgeLabels);
void fixEndEdge(AttributedGraph *g, uint32_t nodeIndex, uint64_t edgeIndex);
void setNode(AttributedGraph *g, uint32_t nodeIndex, uint32_t uuid, uint32_t label, char* name);
void setNodeLabel(AttributedGraph *g, uint32_t label, char* name);
void setEdgeLabel(AttributedGraph *g, uint32_t label, char* name);
void setNodeAttribute(AttributedGraph *g, uint32_t nodeIndex, char* key, char* value);
void constructEdge(AttributedGraph *g, uint64_t edgeIndex, uint32_t dstNodeIndex, uint32_t label, uint64_t timestamp);
void setEdgeAttribute(AttributedGraph *g, uint32_t edgeIndex, char* key, char* value);

size_t getNumNodes(AttributedGraph* g);
size_t getNumEdges(AttributedGraph* g);

//void setNodeAttr(AttributedGraph *g, GNode n, const KeyAltTy key, const ValAltTy val);
//const ValAltTy getNodeAttr(AttributedGraph *g, GNode n, const KeyAltTy key);
//void removeNodeAttr(AttributedGraph *g, GNode n, const KeyAltTy key);

//void setEdgeAttr(AttributedGraph *g, Edge e, const KeyAltTy key, const ValAltTy val);
//const ValAltTy getEdgeAttr(AttributedGraph *g, Edge e, const KeyAltTy key);
//void removeEdgeAttr(AttributedGraph *g, Edge e, const KeyAltTy key);

void runAttributedGraphSimulation(AttributedGraph* queryGraph, AttributedGraph* dataGraph, char* outputFile);

void listFilesWithMultipleWrites(AttributedGraph* dataGraph, char* outputFile);
void listProcessesWithReadFileWriteNetflow(AttributedGraph* dataGraph, char* outputFile);
void listProcessesOriginatingFromNetflow(AttributedGraph* dataGraph, char* outputFile);
void listProcessesOriginatingFromNetflowIndirectly(AttributedGraph* dataGraph, char* outputFile);

} // extern "C"

#endif // GALOIS_PYTHON_GRAPH_H

