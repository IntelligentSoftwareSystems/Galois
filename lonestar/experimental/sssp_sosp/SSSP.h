#ifndef APPS_SSSP_SSSP_H
#define APPS_SSSP_SSSP_H

#include "llvm/Support/CommandLine.h"

#include <limits>
#include <string>
#include <sstream>
#include <stdint.h>

typedef unsigned int Dist;
static const Dist DIST_INFINITY = std::numeric_limits<Dist>::max() - 1;

template<typename GrNode>
struct UpdateRequestCommon {
  GrNode n;
  Dist w;

  UpdateRequestCommon(const GrNode& N, Dist W): n(N), w(W) {}
  
  UpdateRequestCommon(): n(), w(0) {}

  bool operator>(const UpdateRequestCommon& rhs) const {
    if (w > rhs.w) return true;
    if (w < rhs.w) return false;
    return n > rhs.n;
  }

  bool operator<(const UpdateRequestCommon& rhs) const {
    if (w < rhs.w) return true;
    if (w > rhs.w) return false;
    return n < rhs.n;
  }

  bool operator!=(const UpdateRequestCommon& other) const {
    if (w != other.w) return true;
    return n != other.n;
  }

  uintptr_t getID() const {
    return reinterpret_cast<uintptr_t>(n);
  }
};

struct SNode {
  std::atomic<Dist> dist;
};

template<typename Graph>
void readInOutGraph(Graph& graph);

extern llvm::cl::opt<unsigned int> memoryLimit;


#endif
