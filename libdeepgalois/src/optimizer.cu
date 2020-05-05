#include "deepgalois/optimizer.h"
#include "deepgalois/cutils.h"
#include "deepgalois/math_functions.hh"

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

namespace deepgalois {

template <int N>
template <int Index>
float_t* stateful_optimizer<N>::get_gpu(const size_t n, const float_t *key) {
  static_assert(Index < N, "index out of range");
  if (!is_allocated_device(dE_[Index][key])) {
    float_malloc_device(n, dE_[Index][key]);
    init_const_gpu(n, 0.0, dE_[Index][key]);
  }
  return dE_[Index][key];
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

void adagrad::update_gpu(const size_t, const float_t*, float_t*) {}

void RMSprop::update_gpu(const size_t, const float_t*, float_t*) {}

void adamax::update_gpu(const size_t, const float_t*, float_t*) {}

void gradient_descent::update_gpu(const size_t, const float_t*, float_t*) {}

void momentum::update_gpu(const size_t, const float_t*, float_t*) {}

void nesterov_momentum::update_gpu(const size_t, const float_t*, float_t*) {}

}
