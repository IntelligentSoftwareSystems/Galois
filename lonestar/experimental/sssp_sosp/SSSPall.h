#ifndef SSSP_H
#define SSSP_H

#include <limits>
#include <string>
#include <sstream>
#include <stdint.h>

#define NUM 32

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

template<typename GrNode>
struct UpdateRequestCommon {
  GrNode n;
  unsigned int w;
  unsigned int c;

  UpdateRequestCommon(const GrNode& N, unsigned int W, unsigned int C)
    :n(N), w(W), c(C)
  {}

  UpdateRequestCommon()
    :n(), w(0), c(0)
  {}

  bool operator>(const UpdateRequestCommon& rhs) const {
    if (w > rhs.w) return true;
    if (w < rhs.w) return false;
    if (n > rhs.n) return true;
    if (n < rhs.n) return false;
    return c > rhs.c;
  }

  bool operator<(const UpdateRequestCommon& rhs) const {
    if (w < rhs.w) return true;
    if (w > rhs.w) return false;
    if (n < rhs.n) return true;
    if (n > rhs.n) return false;
    return c < rhs.c;
  }

  bool operator!=(const UpdateRequestCommon& other) const {
    if (w != other.w) return true;
    if (n != other.n) return true;
    return c != other.c;
  }

  uintptr_t getID() const {
    //return static_cast<uintptr_t>(n);
    return reinterpret_cast<uintptr_t>(n);
  }
};

struct SNode {
  unsigned int id;
  unsigned int dist[NUM];

  SNode(int _id = -1) : id(_id) {  }
};
#endif
