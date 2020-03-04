#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "deepgalois/cutils.h"
#include "deepgalois/layers/aggregator.h"
#include "deepgalois/math_functions.hh"

// TODO: use warp
__device__ void scale_add(const int n, const float_t alpha, const float_t* a,
                          const float_t* b, float_t* y) {
  for (int i = 0; i < n; i++)
    y[i] = alpha * a[i] + b[i];
}

__global__ void update_all_naive(size_t n, size_t len, CSRGraph g,
                                  const float_t* in, float_t* out,
                                  bool norm, const float_t* norm_factor) {
  CUDA_KERNEL_LOOP(src, n) {
    float_t a = 0.0, b = 1.0;
    if (norm) a = norm_factor[src];
    index_type begin = g.edge_begin(src);
    index_type end   = g.edge_end(src);
    for (index_type e = begin; e != end; e++) {
      index_type dst = g.getEdgeDst(e);
      if (norm) b = a * norm_factor[dst];
      scale_add(len, b, in + dst * len, out + src * len,
                out + src * len); // out[src] += in[dst]
    }
  }
}

__global__ void update_all_warp(size_t n, size_t len, CSRGraph g,
                                  const float_t* in, float_t* out,
                                  bool norm, const float_t* norm_factor) {
  __shared__ index_type ptrs[BLOCK_SIZE/WARP_SIZE][2];
  const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for(int src = warp_id; src < n; src += num_warps) {
    float_t a = 0.0, b = 1.0;
    if (norm) a = norm_factor[src];
    if (thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(src + thread_lane);
    __syncthreads();
    const index_type row_begin = ptrs[warp_lane][0];
    const index_type row_end   = ptrs[warp_lane][1];
    index_type base_src = src * len;
    for(index_type offset = row_begin; offset < row_end; offset ++) {
      index_type dst = g.getEdgeDst(offset);
      if (norm) b = a * norm_factor[dst];
      index_type base_dst = dst * len;
      for (int i = 0; i < len; i += WARP_SIZE)
        if (thread_lane+i < len)
          out[base_src+thread_lane+i] += in[base_dst+thread_lane+i] * b;
    }
  }
}

void deepgalois::update_all(size_t len, CSRGraph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor) {
  unsigned n = g.nnodes;
  CUDA_CHECK(cudaMemset(out, 0, n * len * sizeof(float_t)));
  //update_all_naive<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, len, g, in, out, norm, norm_factor);
  update_all_warp<<<(n-1)/WARPS_PER_BLOCK+1, BLOCK_SIZE>>>(n, len, g, in, out, norm, norm_factor);
  CudaTest("solving update_all kernel failed");
}

void deepgalois::update_all_csrmm(size_t len, CSRGraph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor) {
  unsigned n = g.nnodes;
  CUDA_CHECK(cudaMemset(out, 0, n * len * sizeof(float_t)));
  //std::cout << "[debug]: update_all on GPU, n=" << n << ", len=" << len << "\n";
  //print_device_vector(10, norm_factor, "norm_factor");
  csrmm_gpu(n, len, n, g.nedges, 1.0, norm_factor, (const int*)g.row_start_ptr(), (const int*)g.edge_dst_ptr(), in, 0.0, out);
}
