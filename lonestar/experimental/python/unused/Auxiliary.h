#ifndef GALOIS_PYTHON_AUXILIARY_H
#define GALOIS_PYTHON_AUXILIARY_H

#include "PythonGraph.h"

struct NodeList {
  int num;
  GNode* nodes;
};

struct NodeDouble {
  GNode n;
  double v;
};

struct NodePair {
  GNode nQ;
  GNode nD;
};

struct AttrList {
  size_t num;
  KeyAltTy* key;
  ValAltTy* value;
};

struct EdgeList {
  size_t num;
  Edge* edges;
};

extern "C" {

extern const size_t DIST_INFINITY;

AttrList getNodeAllAttr(Graph* g, GNode n);
AttrList getEdgeAllAttr(Graph* g, Edge e);

NodeList getAllNodes(Graph* g);
EdgeList getAllEdges(Graph* g);

NodeList createNodeList(int num);
void printNodeList(NodeList nl);
void deleteNodeList(NodeList nl);

EdgeList createEdgeList(int num);
void printEdgeList(EdgeList el);
void deleteEdgeList(EdgeList el);

void deleteNodeDoubles(NodeDouble* array);

void deleteGraphMatches(NodePair* pairs);

void deleteAttrList(AttrList l);

} // extern "C"

#endif // GALOIS_PYTHON_AUXILIARY_H
