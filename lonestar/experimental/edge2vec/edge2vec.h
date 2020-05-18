#ifndef EDGE2VEC_H
#define EDGE2VEC_H

constexpr galois::MethodFlag flag_no_lock    = galois::MethodFlag::UNPROTECTED;
constexpr galois::MethodFlag flag_read_lock  = galois::MethodFlag::READ;
constexpr galois::MethodFlag flag_write_lock = galois::MethodFlag::WRITE;

typedef uint32_t NodeTy;
// typedef uint32_t EdgeTy;

struct EdgeTy {

  uint32_t weight;
  uint32_t type;
};

using Graph = galois::graphs::LC_CSR_Graph<NodeTy, EdgeTy>::with_no_lockable<
    false>::type::with_numa_alloc<true>::type;

using GNode = Graph::GraphNode;

std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0.0, 1.0);

#endif // EDGE2VEC_H
