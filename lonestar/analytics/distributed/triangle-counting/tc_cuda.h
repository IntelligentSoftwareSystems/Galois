#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/EdgeHostDecls.h"

void sortEdgesByDestination_cuda(struct CUDA_Context* ctx);
void TC_cuda(unsigned int __begin, unsigned int __end,
             unsigned long& num_local_triangles, struct CUDA_Context* ctx);
void TC_masterNodes_cuda(unsigned long& num_local_triangles,
                         struct CUDA_Context* ctx);
