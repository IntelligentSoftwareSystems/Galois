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

#include "Galois/Graphs/Graph.h"

#include <iostream>
#include <string>
#include <map>

typedef std::string KeyTy;
typedef std::string ValTy;
typedef std::map<KeyTy, ValTy> Attr;

typedef char * KeyAltTy;
typedef char * ValAltTy;

typedef Galois::Graph::FirstGraph<Attr, Attr, true> Graph; // directed graph with nodes and edges of type Attr
typedef Graph::GraphNode GNode;
typedef Graph::edge_iterator edge_iterator;

extern "C" {

typedef struct Edge {
  typename Graph::edge_iterator::value_type *base, *end;
} Edge;

Graph *createGraph();
void deleteGraph(Graph *g);
void printGraph(Graph *g);

GNode createNode(Graph *g);
void addNode(Graph *g, const GNode n);
void addNodeAttr(Graph *g, GNode n, const KeyAltTy key, const ValAltTy val);
void removeNodeAttr(Graph *g, GNode n, const KeyAltTy key);

Edge addEdge(Graph *g, GNode src, GNode dst);
void addEdgeAttr(Graph *g, Edge e, const KeyAltTy key, const ValAltTy val);
void removeEdgeAttr(Graph *g, Edge e, const KeyAltTy key);

void analyzeBFS(Graph* g, int numThreads);

} // extern "C"

#endif // GALOIS_PYTHON_GRAPH_H

