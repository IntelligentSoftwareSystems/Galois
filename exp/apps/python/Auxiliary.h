#ifndef GALOIS_PYTHON_AUXILIARY_H
#define GALOIS_PYTHON_AUXILIARY_H

#include "PythonGraph.h"

struct NodeList {
  int num;
  GNode *nodes;
};

struct NodeDouble {
  GNode n;
  double v;
};

struct NodePair {
  GNode nQ;
  GNode nD;
};

extern "C" {

extern const size_t DIST_INFINITY;

NodeList createNodeList(int num);
void printNodeList(NodeList nl);
void deleteNodeList(NodeList nl);

void deleteNodeDoubles(NodeDouble *array);

void deleteGraphMatches(NodePair *pairs);

} // extern "C"

#endif // GALOIS_PYTHON_AUXILIARY_H

