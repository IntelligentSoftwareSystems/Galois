#pragma once
#include "pangolin/types.h"

struct Edge {
  VertexId src;
  VertexId dst;
#ifdef USE_DOMAIN
  unsigned src_domain;
  unsigned dst_domain;
  Edge(VertexId _src, VertexId _dst, unsigned _src_domain, unsigned _dst_domain)
      : src(_src), dst(_dst), src_domain(_src_domain), dst_domain(_dst_domain) {
  }
#endif
  Edge(VertexId _src, VertexId _dst) : src(_src), dst(_dst) {}
  Edge() : src(0), dst(0) {}
  ~Edge() {}
  std::string toString() {
    return "(" + std::to_string(src) + ", " + std::to_string(dst) + ")";
  }
  std::string to_string() const {
    std::stringstream ss;
    ss << "e(" << src << "," << dst << ")";
    return ss.str();
  }
  void swap() {
    if (src > dst) {
      VertexId tmp = src;
      src          = dst;
      dst          = tmp;
#ifdef USE_DOMAIN
      unsigned domain = src_domain;
      src_domain      = dst_domain;
      dst_domain      = domain;
#endif
    }
  }
};

class EdgeComparator {
public:
  int operator()(const Edge& oneEdge, const Edge& otherEdge) {
    if (oneEdge.src == otherEdge.src) {
      return oneEdge.dst > otherEdge.dst;
    } else {
      return oneEdge.src > otherEdge.src;
    }
  }
};

typedef std::pair<VertexId, VertexId> OrderedEdge;
typedef std::priority_queue<Edge, std::vector<Edge>, EdgeComparator> EdgeHeap;
