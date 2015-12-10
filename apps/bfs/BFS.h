#ifndef APPS_BFS_BFS_H
#define APPS_BFS_BFS_H

#include "llvm/Support/CommandLine.h"

typedef unsigned int Dist;
static const Dist DIST_INFINITY = std::numeric_limits<Dist>::max() - 1;

//! Standard data type on nodes
struct SNode {
  Dist dist;
};

template<typename Graph>
void readInOutGraph(Graph& graph);

extern llvm::cl::opt<unsigned int> memoryLimit;

#endif
