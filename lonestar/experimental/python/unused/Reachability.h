#ifndef GALOIS_PYTHON_REACHABILITY_H
#define GALOIS_PYTHON_REACHABILITY_H

#include "PythonGraph.h"
#include "Auxiliary.h"

extern "C" {

NodeList findReachableFrom(Graph* g, NodeList dst, int hop);
NodeList findReachableTo(Graph* g, NodeList src, int hop);
NodeList findReachableBetween(Graph* g, NodeList src, NodeList dst, int hop);
}

#endif // GALOIS_PYTHON_REACHABILITY_H
