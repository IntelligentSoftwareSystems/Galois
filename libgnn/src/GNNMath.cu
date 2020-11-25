#include "galois/GNNMath.cuh"

bool galois::cublas_is_init = false;
cublasHandle_t galois::global_cublas_handle;
bool galois::curand_is_init = false;
curandGenerator_t galois::global_curand_generator;

void galois::InitCuBLAS() {
  CUBLAS_CHECK(cublasCreate(&global_cublas_handle));
  galois::cublas_is_init = true;
}

void galois::InitCuRAND() {
  CURAND_CHECK(curandCreateGenerator(&galois::global_curand_generator,
                                     CURAND_RNG_PSEUDO_DEFAULT));
  galois::curand_is_init = true;
}

void galois::CBlasSGEMMGPU(const cublasOperation_t trans_a,
                           const cublasOperation_t trans_b, size_t input_rows,
                           size_t input_columns, size_t output_columns,
                           const GNNFloat* a, const GNNFloat* b,
                           GNNFloat* output) {
  if (!cublas_is_init) {
    InitCuBLAS();
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
  CUDA_TEST("cublas sgemm failure");
}

__global__ void galois::SoftmaxCrossEntropyForward(
    char* mask, size_t num_nodes, size_t feature_length,
    const galois::GNNFloat* input_embeddings, galois::GNNFloat* output) {

  // NOTE: assumes that output is already 0'd out as it will not overwrite the
  // entire thing
  CUDA_KERNEL_LOOP(i, num_nodes) {
    if (mask[i] == 1) {
      galois::DoSoftmax(feature_length, input_embeddings + feature_length * i,
                        output + feature_length * i);
      // ignoring crossentropy loss calculation for now because I'm not using
      // loss for anything + didn't bother allocating an array to store loss
      // anyways
    }
  }
}

__global__ void galois::SoftmaxCrossEntropyBackward(
    char* mask, size_t num_nodes, size_t feature_length,
    const galois::GNNFloat* predictions, const galois::GNNLabel* ground_truth,
    galois::GNNFloat* output_gradient) {
  const unsigned global_thread_id =
      BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const unsigned warp_thread_lane =
      threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const unsigned warp_id = global_thread_id / WARP_SIZE; // global warp index
  const unsigned warp_lane =
      threadIdx.x / WARP_SIZE; // warp index within the CTA
  const unsigned num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  // TODO: how many classes can there be? it's a set quantity at the moment
  // copy of a particular node's prediction; put into shared memory to avoid
  // overheads of accessing it otherwise
  // TODO benchmark
  __shared__ GNNFloat
      local_node_prediction[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];
  __shared__ GNNFloat
      intermediate_gradient[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];

  // a warp works on a single node at once
  for (unsigned wid = warp_id; wid < num_nodes; wid += num_warps) {
    // operate only if masked
    if (mask[wid] == 1) {
      unsigned base_index = wid * feature_length;

      // copy over a prediction to shared memory (faster access time)
      // TODO benchmark this to see if worth
      for (unsigned feat_index = warp_thread_lane; feat_index < feature_length;
           feat_index += WARP_SIZE) {
        if (feat_index < feature_length) {
          local_node_prediction[warp_lane][feat_index] =
              predictions[base_index + feat_index];
        }
      }
      // do not proceed until entire prediction is copied to shared memory
      __syncthreads();

      // TODO can refactor below to device functions
      // cross entropy derivative
      // each thread of warp takes different feature
      for (unsigned feat_index = warp_thread_lane; feat_index < feature_length;
           feat_index += WARP_SIZE) {
        if (feat_index < feature_length) {
          if (feat_index == (unsigned)ground_truth[wid]) {
            // this thread is responsible for the truth
            intermediate_gradient[warp_lane][feat_index] =
                -1.0 / (local_node_prediction[warp_lane][feat_index] + 1e-10);
          } else {
            // all others are 0 (ground truth label = 0)
            intermediate_gradient[warp_lane][feat_index] = 0.0;
          }
        }
      }
      __syncthreads();

      // softmax derivative
      // each thread of warp takes different feature
      for (unsigned feat_index = warp_thread_lane; feat_index < feature_length;
           feat_index += WARP_SIZE) {
        if (feat_index < feature_length) {
          GNNFloat sum  = 0.0;
          GNNFloat self = local_node_prediction[warp_lane][feat_index];

          for (unsigned j = 0; j < feature_length; j++) {
            GNNFloat df = (j == feat_index)
                              ? (self * (1.0 - self))
                              : -local_node_prediction[warp_lane][j] * self;
            sum += df * intermediate_gradient[warp_lane][j];
          }

          // each thread saves final output for the feature
          output_gradient[base_index + feat_index] = sum;
        }
      }
      __syncthreads();
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
