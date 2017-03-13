#ifndef GALOIS_PYTHON_REACHABILITY_H
#define GALOIS_PYTHON_REACHABILITY_H

#include "PythonGraph.h"
#include "Filter.h"

extern "C" {

NodeList findReachable(Graph *g, NodeList src, NodeList dst, int hop);

}

#endif // GALOIS_PYTHON_REACHABILITY_H

