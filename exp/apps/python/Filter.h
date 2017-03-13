#ifndef GALOIS_PYTHON_FILTER_H
#define GALOIS_PYTHON_FILTER_H

#include "PythonGraph.h"
#include "NodeList.h"

extern "C" {

NodeList filterNode(Graph* g, const KeyAltTy key, const ValAltTy value);

} // extern "C"

#endif // GALOIS_PYTHN_FILTER_H

