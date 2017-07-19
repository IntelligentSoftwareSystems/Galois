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

#include <string>
#include <unordered_map>
#include <vector>

typedef std::string KeyTy;
typedef std::string ValTy;
typedef std::unordered_map<KeyTy, ValTy> Attr;

typedef char * KeyAltTy;
typedef char * ValAltTy;

struct Node;

#define DIRECTED true
#define IN_EDGES true

typedef Galois::Graph::FirstGraph<Node, Attr, DIRECTED, IN_EDGES> Graph; // Node nodes and Attr edges
typedef Graph::GraphNode GNode;

// see StatCollector for the design
struct Node {
  Attr attr;
  union {
    struct {
      size_t vInt;
      double vDouble;
    } ID;
    struct {
      size_t vInt1;
      size_t vInt2;
    } II;
    struct {
      size_t vDouble1;
      size_t vDouble2;
    } DD;
    struct {
      double vDouble;
      std::atomic<double> vAtomicDouble;
    } DAd;
    std::vector<GNode> vVec;
  };

  // analyses are responsible to construct/destruct the union properly
  Node() {}
  ~Node() {}
};

struct Edge {
  GNode src;
  GNode dst;
};

extern "C" {

Graph *createGraph();
void deleteGraph(Graph *g);
void printGraph(Graph *g);

GNode createNode(Graph *g);
void addNode(Graph *g, const GNode n);
void setNodeAttr(Graph *g, GNode n, const KeyAltTy key, const ValAltTy val);
const ValAltTy getNodeAttr(Graph *g, GNode n, const KeyAltTy key);
void removeNodeAttr(Graph *g, GNode n, const KeyAltTy key);

Edge addEdge(Graph *g, GNode src, GNode dst);
void setEdgeAttr(Graph *g, Edge e, const KeyAltTy key, const ValAltTy val);
const ValAltTy getEdgeAttr(Graph *g, Edge e, const KeyAltTy key);
void removeEdgeAttr(Graph *g, Edge e, const KeyAltTy key);

void setNumThreads(int numThreads);
size_t getNumNodes(Graph* g);
size_t getNumEdges(Graph* g);

} // extern "C"

#endif // GALOIS_PYTHON_GRAPH_H

