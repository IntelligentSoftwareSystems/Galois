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

Graph *createGraph();
void deleteGraph(Graph *g);
void printGraph(Graph *g);

void allocateGraph(Graph *g, size_t numNodes, size_t numEdges);
void fixEndEdge(Graph *g, uint32_t nodeIndex, uint64_t edgeIndex);
void setNode(Graph *g, uint32_t nodeIndex, uint32_t label, uint64_t id);
void constructEdge(Graph *g, uint64_t edgeIndex, uint32_t dstNodeIndex, uint32_t label, uint64_t timestamp);

size_t getNumNodes(Graph* g);
size_t getNumEdges(Graph* g);

//void setNodeAttr(Graph *g, GNode n, const KeyAltTy key, const ValAltTy val);
//const ValAltTy getNodeAttr(Graph *g, GNode n, const KeyAltTy key);
//void removeNodeAttr(Graph *g, GNode n, const KeyAltTy key);

//void setEdgeAttr(Graph *g, Edge e, const KeyAltTy key, const ValAltTy val);
//const ValAltTy getEdgeAttr(Graph *g, Edge e, const KeyAltTy key);
//void removeEdgeAttr(Graph *g, Edge e, const KeyAltTy key);

} // extern "C"

#endif // GALOIS_PYTHON_GRAPH_H

