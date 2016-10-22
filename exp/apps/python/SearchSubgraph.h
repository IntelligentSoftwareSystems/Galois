#ifndef GALOIS_PYTHON_SUBGRAPH_ISOMORPHISM_H
#define GALOIS_PYTHON_SUBGRAPH_ISOMORPHISM_H

#include "PythonGraph.h"

extern "C" {

void searchSubgraphUllmann(Graph *gD, Graph *gQ, size_t k);
void searchSubgraphVF2(Graph *gD, Graph *gQ, size_t k);

} // extern "C"

#endif // GALOIS_PYTHON_SUBGRAPH_ISOMORPHISM_H

