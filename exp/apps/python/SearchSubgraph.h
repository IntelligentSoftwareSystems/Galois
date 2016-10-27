#ifndef GALOIS_PYTHON_SUBGRAPH_ISOMORPHISM_H
#define GALOIS_PYTHON_SUBGRAPH_ISOMORPHISM_H

#include "PythonGraph.h"

struct NodePair {
  GNode nQ;
  GNode nD;
};


extern "C" {

void deleteGraphMatches(NodePair *pairs);
NodePair *searchSubgraphUllmann(Graph *gD, Graph *gQ, size_t k);
//NodePair *searchSubgraphVF2(Graph *gD, Graph *gQ, size_t k);

} // extern "C"

#endif // GALOIS_PYTHON_SUBGRAPH_ISOMORPHISM_H

