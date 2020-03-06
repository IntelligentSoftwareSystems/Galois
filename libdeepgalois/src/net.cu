#include "deepgalois/net.h"
#include "gg.h"
#include "ggcuda.h"

__global__ void masked_accuracy_kernel(int num_classes, int begin,
                                       int end, mask_t* masks,
                                       float_t* preds, label_t* labels,
                                       HGAccumulator<acc_t> total) {
  total.thread_entry();
  __shared__ cub::BlockReduce<acc_t, CUDA_NUM_THREADS>::TempStorage
      local_accuracy;
  CUDA_KERNEL_LOOP(i, end - begin) {
    if (masks[begin + i] == 1) {
      label_t pred = (label_t)argmax_device(num_classes,
                                            preds + (begin + i) * num_classes);
      if (pred == labels[begin + i])
        total.reduce(1.0);
    }
  }
  total.thread_exit<cub::BlockReduce<acc_t, CUDA_NUM_THREADS>>(local_accuracy);
}

acc_t masked_accuracy_gpu(int num_classes, int begin, int end,
                          int count, mask_t* masks, float_t* preds,
                          label_t* labels) {
  assert(count > 0);
  HGAccumulator<acc_t> accuracy_accum;
  Shared<acc_t> total_accuracy   = Shared<acc_t>(1);
  *(total_accuracy.cpu_wr_ptr()) = 0;
  accuracy_accum.rv              = total_accuracy.gpu_wr_ptr();
  masked_accuracy_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(
      num_classes, begin, end, masks, preds, labels, accuracy_accum);
  CudaTest("solving masked_accuracy kernel failed");
  cudaDeviceSynchronize();
  return *(total_accuracy.cpu_rd_ptr()) / count;
}

acc_t Net::masked_accuracy(size_t begin, size_t end, size_t count,
                           mask_t* masks) {
  return masked_accuracy_gpu(num_classes, begin, end, count,
                             layers[NUM_CONV_LAYERS]->get_device_masks(),
                             layers[NUM_CONV_LAYERS - 1]->next()->get_data(),
                             context->d_labels);
}

