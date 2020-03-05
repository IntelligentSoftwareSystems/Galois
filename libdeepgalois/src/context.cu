#include <ctime>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include "deepgalois/context.h"

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }
  std::cout << "System entropy source not available, "
               "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);
  pid  = getpid();
  s    = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

namespace deepgalois {

cublasHandle_t Context::cublas_handle_         = 0;
cusparseHandle_t Context::cusparse_handle_     = 0;
cusparseMatDescr_t Context::cusparse_matdescr_ = 0;
curandGenerator_t Context::curand_generator_   = 0;

Context::Context()
    : mode_(Context::GPU), solver_count_(1), solver_rank_(0),
      multiprocess_(false) {
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
  CUSPARSE_CHECK(cusparseCreateMatDescr(&cusparse_matdescr_));
  CUSPARSE_CHECK(cusparseSetMatType(cusparse_matdescr_,CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CHECK(cusparseSetMatIndexBase(cusparse_matdescr_,CUSPARSE_INDEX_BASE_ZERO));
  CURAND_CHECK(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()));
}

Context::~Context() {
  if (cublas_handle_)
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (cusparse_handle_)
    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle_));
  if (cusparse_matdescr_)
    CUSPARSE_CHECK(cusparseDestroyMatDescr(cusparse_matdescr_));
  if (curand_generator_)
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
}

void Context::SetDevice(const int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id)
    return;
  CUDA_CHECK(cudaSetDevice(device_id));
  if (cublas_handle_)
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_)
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CURAND_CHECK(
      curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()));
}

size_t Context::read_graph_gpu(std::string dataset_str, bool selfloop) {
  std::string filename = path + dataset_str + ".csgr";
  CSRGraph g;
  g.read(filename.c_str(), false);
  if (selfloop) g.add_selfloop();
  g.copy_to_gpu(graph_gpu);
  return graph_gpu.nnodes;
}

void Context::copy_data_to_device() {
  assert(labels.size() == n);
  CUDA_CHECK(cudaMalloc((void**)&d_labels, n * sizeof(label_t)));
  CUDA_CHECK(cudaMemcpy(d_labels, &labels[0], n * sizeof(label_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void**)&d_feats, n * feat_len * sizeof(float_t)));
  CUDA_CHECK(cudaMemcpy(d_feats, &h_feats[0], n * feat_len * sizeof(float_t),
                        cudaMemcpyHostToDevice));
  //print_device_vector(10, d_feats, "d_feats");
}

float_t* Context::get_in_ptr() { return d_feats; }
} // namespace context
