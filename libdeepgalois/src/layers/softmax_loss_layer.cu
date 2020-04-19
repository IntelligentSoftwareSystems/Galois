#include "deepgalois/layers/softmax_loss_layer.h"
#include "gg.h"
#include "ggcuda.h"

__global__ void masked_avg_loss_kernel(int begin, int end, mask_t* masks,
                                       float_t* loss,
                                       HGAccumulator<acc_t> total) {
  total.thread_entry();
  __shared__ cub::BlockReduce<acc_t, CUDA_NUM_THREADS>::TempStorage local_loss;
  CUDA_KERNEL_LOOP(i, end - begin) {
    if (masks[begin + i] == 1)
      // total += loss[begin+i];
      total.reduce(loss[begin + i]);
  }
  total.thread_exit<cub::BlockReduce<acc_t, CUDA_NUM_THREADS>>(local_loss);
}

//acc_t masked_avg_loss(int begin, int end, int count, mask_t* masks, float_t* loss);
acc_t masked_avg_loss(int begin, int end, int count, mask_t* masks,
                      float_t* loss) {
  assert(count > 0);
  HGAccumulator<acc_t> loss_accum;
  Shared<acc_t> total_loss   = Shared<acc_t>(1);
  *(total_loss.cpu_wr_ptr()) = 0;
  loss_accum.rv              = total_loss.gpu_wr_ptr();
  masked_avg_loss_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(
      begin, end, masks, loss, loss_accum);
  CudaTest("solving masked_avg_loss kernel failed");
  cudaDeviceSynchronize();
  return *(total_loss.cpu_rd_ptr()) / count;
}

namespace deepgalois {

softmax_loss_layer::softmax_loss_layer(unsigned level,
                                       std::vector<size_t> in_dims,
                                       std::vector<size_t> out_dims)
    : layer(level, in_dims, out_dims) {
  trainable_ = false;
  name_      = layer_type() + "_" + std::to_string(level);
  float_malloc_device(in_dims[0], loss);
}

softmax_loss_layer::~softmax_loss_layer() {
  float_free_device(loss);
}

void softmax_loss_layer::forward_propagation(const float_t* in_data,
                                             float_t* out_data) {
  init_const_gpu(input_dims[0], 0.0, loss);
  softmax_cross_entropy_gpu(input_dims[1], begin_, end_, in_data,
                            d_masks_, context->d_labels, loss, out_data);
}

void softmax_loss_layer::back_propagation(const float_t* in_data,
                                          const float_t* out_data,
                                          float_t* out_grad, float_t* in_grad) {
  d_softmax_cross_entropy_gpu(input_dims[1], begin_, end_, d_masks_,
                              context->d_labels, out_data, in_grad);
}

acc_t softmax_loss_layer::get_masked_loss() {
  return masked_avg_loss(begin_, end_, count_, d_masks_, loss);
}

} // namespace
