#ifndef GALOIS_PYTHON_COARSEN_H
#define GALOIS_PYTHON_COARSEN_H

#include "PythonGraph.h"

extern "C" {

void coarsen(Graph* fg, Graph* cg, const KeyAltTy key);
}

#endif // GALOIS_PYTHON_COARSEN_H
