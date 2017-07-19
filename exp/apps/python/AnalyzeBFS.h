#ifndef GALOIS_PYTHON_BFS_H
#define GALOIS_PYTHON_BFS_H

#include "PythonGraph.h"

extern "C" {

void analyzeBFS(Graph* g, GNode start, const ValAltTy result);

} // extern "C"

#endif // GALOIS_PYTHON_BFS_H

