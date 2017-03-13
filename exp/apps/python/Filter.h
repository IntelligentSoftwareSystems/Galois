#ifndef GALOIS_PYTHON_FILTER_H
#define GALOIS_PYTHON_FILTER_H

#include "PythonGraph.h"

struct NodeList {
  int num;
  GNode *nodes;
};

extern "C" {

NodeList filterNode(Graph* g, const KeyAltTy key, const ValAltTy value);
void deleteNodeList(NodeList nl);

} // extern "C"

#endif // GALOIS_PYTHN_FILTER_H

