#ifndef GALOIS_PYTHON_NODELIST_H
#define GALOIS_PYTHON_NODELIST_H

#include "PythonGraph.h"

struct NodeList {
  int num;
  GNode *nodes;
};

extern "C" {

NodeList createNodeList(int num);
void deleteNodeList(NodeList nl);

} // extern "C"

#endif // GALOIS_PYTHON_NODELIST_H

