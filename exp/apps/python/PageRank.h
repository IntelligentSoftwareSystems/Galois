#ifndef GALOIS_PYTHON_PAGERANK_H
#define GALOIS_PYTHON_PAGERANK_H

#include "PythonGraph.h"

struct NodeDouble {
  GNode n;
  double v;
};

extern "C" {

void deleteNodeDoubles(NodeDouble *array);
NodeDouble *analyzePagerank(Graph *g, int topK, double tolerance, const ValAltTy result);

} // extern "C"

#endif // GALOIS_PYTHON_PAGERANK_H
