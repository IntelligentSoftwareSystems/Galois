#include "deepgalois/net.h"
#include "deepgalois/cutils.h"
#include "gg.h"
#include "ggcuda.h"

// the arguments of the maxima
__device__ int argmax_device(const int n, const float_t* x) {
  float_t max    = x[0];
  int max_ind = 0;
  for (int i = 1; i < n; i++) {
    if (x[i] > max) {
      max_ind = i;
      max     = x[i];
    }
  }
  return max_ind;
}

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

acc_t masked_accuracy_gpu(int num_classes, int begin, int end, int count,
                          mask_t* masks, float_t* preds, label_t* labels) {
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

typedef float f1count_t;
__global__ void masked_f1_score_kernel(int num_classes, int begin,
                                       int end, mask_t* masks,
                                       float_t* preds, label_t* labels,
                                       f1count_t* true_positive,
                                       f1count_t* false_positive,
                                       f1count_t* false_negtive) {
  CUDA_KERNEL_LOOP(i, end - begin) {
    if (masks[begin + i] == 1) {
      for (size_t j = 0; j < num_classes; j++) {
        auto idx = i * num_classes + j;
        if (labels[idx] == 1 && preds[idx] > 0.5) {
          atomicAdd(&true_positive[j], 1.0);
        } else if (labels[idx] == 0 && preds[idx] > 0.5) {
          atomicAdd(&false_positive[j], 1.0);
        } else if (labels[idx] == 1 && preds[idx] <= 0.5) {
          atomicAdd(&false_negtive[j], 1.0);
        }
      }
	}
  }
}

acc_t masked_f1_score_gpu(int num_classes, int begin, int end, int count,
                          mask_t* masks, float_t* preds, label_t* labels) {
  float beta = 1.0;
  assert(count > 0);
  f1count_t* h_tp = new f1count_t[num_classes];
  f1count_t* h_fp = new f1count_t[num_classes];
  f1count_t* h_fn = new f1count_t[num_classes];
  f1count_t* d_tp, *d_fp, *d_fn;
  float_malloc_device(num_classes, d_tp);
  float_malloc_device(num_classes, d_fp);
  float_malloc_device(num_classes, d_fn);
  masked_f1_score_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(
      num_classes, begin, end, masks, preds, labels, d_tp, d_fp, d_fn);
  cudaMemcpy(h_tp, d_tp, num_classes * sizeof(f1count_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fp, d_fp, num_classes * sizeof(f1count_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fn, d_fn, num_classes * sizeof(f1count_t), cudaMemcpyDeviceToHost);

  acc_t pNumerator = 0.0;
  acc_t pDenominator = 0.0;
  acc_t rNumerator = 0.0;
  acc_t rDenominator = 0.0;
  for (size_t i = 0; i < num_classes; i++) {
    acc_t fn = (acc_t)h_fn[i]; // false negtive
    acc_t fp = (acc_t)h_fp[i]; // false positive
	acc_t tp = (acc_t)h_tp[i]; // true positive
	pNumerator = pNumerator + tp;
	pDenominator = pDenominator + (tp + fp);
    rNumerator = rNumerator + tp;
    rDenominator = rDenominator + (tp + fn);
  }
  acc_t recallMicro = rNumerator / rDenominator;
  acc_t precisionMicro = pNumerator / pDenominator;
  acc_t fscoreMicro = (((beta * beta) + 1) * precisionMicro * recallMicro) / 
                     ((beta * beta) * precisionMicro + recallMicro);
  float_free_device(d_tp);
  float_free_device(d_fp);
  float_free_device(d_fn);
  return fscoreMicro;
}

namespace deepgalois {
acc_t Net::masked_accuracy(size_t begin, size_t end, size_t count,
                           mask_t* masks, CSRGraph *g) {
  return masked_accuracy_gpu(num_classes, begin, end, count, masks,
                             layers[NUM_CONV_LAYERS - 1]->next()->get_data(),
                             context->get_labels_device_ptr());
}

acc_t Net::masked_multi_class_accuracy(size_t begin, size_t end, size_t count, 
                                       mask_t* masks, CSRGraph* g) {
	return masked_f1_score_gpu(num_classes, begin, end, count, masks,
                             layers[NUM_CONV_LAYERS - 1]->next()->get_data(),
                             context->get_labels_device_ptr());
}

} // end namespace
