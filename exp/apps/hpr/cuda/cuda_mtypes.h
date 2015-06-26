#pragma once

// types to marshal Galois types out of Galois.

// required because of nvcc does not support clang on Linux.

struct MarshalGraph {
  int nnodes;  
};
