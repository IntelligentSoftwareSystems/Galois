#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cutils.h"
#include "aggregator.h"
#include "math_functions.hh"

// TODO: use warp
__device__ void scale_add(const int n, const float_t alpha, const float_t* a,
                          const float_t* b, float_t* y) {
  for (int i = 0; i < n; i++)
    y[i] = alpha * a[i] + b[i];
}

__global__ void update_all_kernel(size_t n, size_t len, CSRGraph g,
                                  const float_t* in, float_t* out,
                                  bool norm, const float_t* norm_factor) {
  CUDA_KERNEL_LOOP(src, n) {
    float_t a = 0.0, b = 1.0;
    if (norm) a = norm_factor[src];
    index_type begin = g.edge_begin(src);
    index_type end   = g.edge_end(src);
    for (index_type e = begin; e != end; e++) {
      index_type dst = g.getEdgeDst(e);
      assert(dst < n);
      if (norm) b = a * norm_factor[dst];
      scale_add(len, b, in + dst * len, out + src * len,
                out + src * len); // out[src] += in[dst]
    }
  }
}

void update_all(size_t len, CSRGraph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor) {
  unsigned n = g.nnodes;
  std::cout << "[debug]: update_all on GPU, n=" << n << ", len=" << len << "\n";
  CUDA_CHECK(cudaMemset(out, 0, n * len * sizeof(float_t)));
  update_all_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, len, g, in, out, norm, norm_factor);
  CudaTest("solving update_all kernel failed");
}
