#include "galois/GNNMath.cuh"

bool galois::cublas_is_init = false;
cublasHandle_t galois::global_cublas_handle;

void galois::InitCuBLAS() { CUBLAS_CHECK(cublasCreate(&global_cublas_handle)); }

void galois::CBlasSGEMMGPU(const cublasOperation_t trans_a,
                           const cublasOperation_t trans_b, size_t input_rows,
                           size_t input_columns, size_t output_columns,
                           const GNNFloat* a, const GNNFloat* b,
                           GNNFloat* output) {
  if (!cublas_is_init) {
    InitCuBLAS();
    cublas_is_init = true;
  }
  size_t lead_dim_a = (trans_a == CUBLAS_OP_N) ? input_columns : input_rows;
  size_t lead_dim_b = (trans_b == CUBLAS_OP_N) ? output_columns : input_columns;
  float dummy0      = 0.0;
  float dummy1      = 1.0;
  // because cusparse assumes column major even though we're passing in row
  // major, the order of multiply is reversed so that it does what we
  // want anyways
  // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
  CUBLAS_CHECK(cublasSgemm(global_cublas_handle, trans_b, trans_a,
                           output_columns, input_rows, input_columns, &dummy1,
                           b, lead_dim_b, a, lead_dim_a, &dummy0, output,
                           output_columns));
}


__global__ void SoftmaxCrossEntropyForward(char* mask, size_t num_nodes, size_t feature_length,
                                      const galois::GNNFloat* input_embeddings,
                                      galois::GNNFloat* output) {
  // XXX zero out output
  CUDA_KERNEL_LOOP(i, num_nodes) {
    if (mask[i] == 1) {
      galois::DoSoftmax(feature_length, input_embeddings + feature_length * i, output + feature_length * i);
      // ignoring crossentropy loss calculation for now because I'm not using
      // loss for anything + didn't bother allocating an array to store loss anyways
    }
  }
}

__device__ void galois::DoSoftmax(size_t vector_length, const GNNFloat* input,
                                  GNNFloat* output) {
  // find max value
  GNNFloat current_max = input[0];
  for (size_t i = 1; i < vector_length; i++) {
    if (input[i] > current_max) {
      current_max = input[i];
    }
  }
  // set output by scaling with the max
  GNNFloat denominator = 0.0;
  for (size_t i = 0; i < vector_length; i++) {
    // NOTE: expf only works for single precision float; may need to change if
    // we ever switch to double
    output[i] = expf(input[i] - current_max);
    denominator += output[i];
  }
  // denominator scale
  for (size_t i = 0; i < vector_length; i++) {
    output[i] /= denominator;
  }
}
