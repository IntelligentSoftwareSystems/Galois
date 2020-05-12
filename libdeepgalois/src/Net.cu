#include "deepgalois/Net.h"
#include "deepgalois/cutils.h"
#include "deepgalois/math_functions.hh"
#include "gg.h"
#include "ggcuda.h"
#include <iomanip>

// the arguments of the maxima
__device__ int argmax_device(const int n, const float_t* x) {
  float_t max = x[0];
  int max_ind = 0;
  for (int i = 1; i < n; i++) {
    if (x[i] > max) {
      max_ind = i;
      max     = x[i];
    }
  }
  return max_ind;
}

__global__ void masked_accuracy_kernel(int num_classes, int begin, int end,
                                       mask_t* masks, float_t* preds,
                                       label_t* labels,
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
__global__ void
masked_f1_score_kernel(int num_classes, int begin, int end, mask_t* masks,
                       float_t* preds, label_t* labels,
                       f1count_t* true_positive, f1count_t* false_positive,
                       f1count_t* false_negtive, f1count_t* true_negtive) {
  CUDA_KERNEL_LOOP(i, end - begin) {
    int id = begin + i;
    if (masks[id] == 1) {
      for (size_t j = 0; j < num_classes; j++) {
        int idx = id * num_classes + j;
        if (labels[idx] == 1 && preds[idx] > 0.5) {
          atomicAdd(&true_positive[j], 1.0);
        } else if (labels[idx] == 0 && preds[idx] > 0.5) {
          atomicAdd(&false_positive[j], 1.0);
        } else if (labels[idx] == 1 && preds[idx] <= 0.5) {
          atomicAdd(&false_negtive[j], 1.0);
        } else {
          atomicAdd(&true_negtive[j], 1.0);
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
  f1count_t* h_tn = new f1count_t[num_classes];
  f1count_t *d_tp, *d_fp, *d_fn, *d_tn;
  float_malloc_device(num_classes, d_tp);
  float_malloc_device(num_classes, d_fp);
  float_malloc_device(num_classes, d_fn);
  float_malloc_device(num_classes, d_tn);
  init_const_gpu(num_classes, 0.0, d_tp);
  init_const_gpu(num_classes, 0.0, d_fp);
  init_const_gpu(num_classes, 0.0, d_fn);
  init_const_gpu(num_classes, 0.0, d_tn);
  masked_f1_score_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(
      num_classes, begin, end, masks, preds, labels, d_tp, d_fp, d_fn, d_tn);
  CudaTest("solving masked_f1_score_kernel kernel failed");
  CUDA_CHECK(cudaMemcpy(h_tp, d_tp, num_classes * sizeof(f1count_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_fp, d_fp, num_classes * sizeof(f1count_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_fn, d_fn, num_classes * sizeof(f1count_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_tn, d_tn, num_classes * sizeof(f1count_t),
                        cudaMemcpyDeviceToHost));

  acc_t pNumerator     = 0.0;
  acc_t pDenominator   = 0.0;
  acc_t rNumerator     = 0.0;
  acc_t rDenominator   = 0.0;
  acc_t precisionMacro = 0.0;
  acc_t recallMacro    = 0.0;
  for (size_t i = 0; i < num_classes; i++) {
    acc_t fn = (acc_t)h_fn[i]; // false negtive
    acc_t fp = (acc_t)h_fp[i]; // false positive
    acc_t tp = (acc_t)h_tp[i]; // true positive
                               // acc_t tn = (acc_t)h_tn[i]; // true positive

    precisionMacro = precisionMacro + (tp / (tp + fp));
    recallMacro    = recallMacro + (tp / (tp + fn));
    pNumerator     = pNumerator + tp;
    pDenominator   = pDenominator + (tp + fp);
    rNumerator     = rNumerator + tp;
    rDenominator   = rDenominator + (tp + fn);
  }
  precisionMacro = precisionMacro / num_classes;
  recallMacro    = recallMacro / num_classes;
  acc_t f1_macro = (((beta * beta) + 1) * precisionMacro * recallMacro) /
                   ((beta * beta) * precisionMacro + recallMacro);
  acc_t recallMicro    = rNumerator / rDenominator;
  acc_t precisionMicro = pNumerator / pDenominator;
  acc_t f1_micro       = (((beta * beta) + 1) * precisionMicro * recallMicro) /
                   ((beta * beta) * precisionMicro + recallMicro);
  std::cout << std::setprecision(3) << std::fixed << " (f1_micro: " << f1_micro
            << ", f1_macro: " << f1_macro << ") ";

  float_free_device(d_tp);
  float_free_device(d_fp);
  float_free_device(d_fn);
  float_free_device(d_tn);
  delete[] h_tp;
  delete[] h_fp;
  delete[] h_fn;
  delete[] h_tn;
  return f1_micro;
}

namespace deepgalois {

void Net::init() {
  copy_masks_device(globalSamples, globalTrainMasks, d_train_masks);
  copy_masks_device(globalSamples, globalValMasks, d_val_masks);
}

void Net::partitionInit(DGraph* graph, std::string dataset_str, bool isSingleClassLabel) {
  this->distContext = new deepgalois::DistContext();
  this->distContext->set_dataset(dataset_str);

  // read the graph into CPU memory and copy it to GPU memory
  this->distNumSamples = this->distContext->read_graph(dataset_str, is_selfloop);

  // read labels into CPU memory
  num_classes = this->distContext->read_labels(isSingleClassLabel, dataset_str);

  // read features into CPU memory
  feature_dims[0] = this->distContext->read_features(dataset_str);

  // copy labels and features from CPU memory to GPU memory
  distContext->copy_data_to_device(); // copy labels and input features to the device

  feature_dims[num_conv_layers] = num_classes; // output embedding: E
  if (this->has_l2norm) {
    // l2 normalized embedding: E
    feature_dims[num_conv_layers + 1] = num_classes;
  }
  if (this->has_dense) {
    // MLP embedding: E
    feature_dims[num_layers - 1] = num_classes;
  }
  feature_dims[num_layers] = num_classes; // normalized output embedding: E
}

void Net::read_test_masks(std::string dataset) {
  test_masks = new mask_t[distNumSamples];
  if (dataset == "reddit") {
    globalTestBegin = 177262;
    globalTestCount = 55703;
    globalTestEnd   = globalTestBegin + globalTestCount;
    for (size_t i = globalTestBegin; i < globalTestEnd; i++)
        test_masks[i] = 1;
  } else {
    globalTestCount = distContext->read_masks(dataset, std::string("test"), 
        globalSamples, globalTestBegin, globalTestEnd, test_masks, NULL);
  }
  copy_test_masks_to_device();
}

void Net::copy_test_masks_to_device() {
  copy_masks_device(globalSamples, test_masks, d_test_masks);
}

// add weight decay
void Net::regularize() {
  size_t layer_id = 0;
  auto n          = feature_dims[layer_id] * feature_dims[layer_id + 1];
  axpy_gpu(n, weight_decay, layers[layer_id]->get_weights_device_ptr(),
           layers[layer_id]->get_grads_device_ptr());
}

//void Net::normalize() {}

acc_t Net::masked_accuracy(size_t begin, size_t end, size_t count,
                           mask_t* masks, float_t* preds,
                           label_t* ground_truth) {
  return masked_accuracy_gpu(num_classes, begin, end, count, masks, preds,
                             ground_truth);
}

acc_t Net::masked_multi_class_accuracy(size_t begin, size_t end, size_t count,
                                       mask_t* masks, float_t* preds,
                                       label_t* ground_truth) {
  return masked_f1_score_gpu(num_classes, begin, end, count, masks, preds,
                             ground_truth);
}

} // namespace deepgalois
