#ifndef GALOIS_GRAPH_H
#define GALOIS_GRAPH_H

#include <galois/graphs/MorphGraph.h>
#include "NodeData.h"
#include "EdgeData.h"

using Graph = galois::graphs::MorphGraph<NodeData, EdgeData, false>;
using GNode = Graph::GraphNode;
using EdgeIterator = Graph::edge_iterator;
using galois::optional;

#endif // GALOIS_GRAPH_H
