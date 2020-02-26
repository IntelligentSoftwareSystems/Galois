#include "optimizer.h"
#include "cutils.h"
#include "math_functions.hh"

__global__ void update_kernel(const int n, float_t alpha, float_t b1,
                         float_t b2, float_t b1_t, float_t b2_t,
                         float_t eps, float_t* mt, float_t* vt,
                         const float_t* dW, float_t* W) {
  CUDA_KERNEL_LOOP(i, n) {
    mt[i] = b1 * mt[i] + (1.0 - b1) * dW[i];
    vt[i] = b2 * vt[i] + (1.0 - b2) * dW[i] * dW[i];
    W[i] -= alpha * (mt[i] / (1.0 - b1_t)) /
            sqrtf((vt[i] / (1.0 - b2_t)) + eps);
  }
}

void adam::update_gpu(const size_t n, const float_t* dW, float_t* W) {
  //std::cout << "updating weights on GPU, n = " << n << "\n";
  //print_device_vector(10, dW, "dW");
  float_t* cache = get_gpu<0>(n, W);
  float_t* velocity = get_gpu<1>(n, W);

  update_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, alpha, b1, b2, b1_t, b2_t, eps, cache, velocity, dW, W);
  b1_t *= b1;
  b2_t *= b2;
}
