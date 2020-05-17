#include <ctime>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include "deepgalois/DistContext.h"
#include "deepgalois/math_functions.hh"
#include "deepgalois/configs.h"

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

// computing normalization factor for each vertex
__global__ void norm_factor_computing_node(int n, GraphGPU graph, float_t* norm_fac) {
  CUDA_KERNEL_LOOP(i, n) {
    float_t temp = sqrt(float_t(graph.getOutDegree(i)));
    if (temp == 0.0) norm_fac[i] = 0.0;
    else norm_fac[i] = 1.0 / temp;
  }
}

// TODO: make sure self-loop added for each vertex
// computing normalization factor for each edge
__global__ void norm_factor_computing_edge(int n, GraphGPU graph, float_t* norm_fac) {
  CUDA_KERNEL_LOOP(src, n) {
    assert(src < n);
    float_t d_src = float_t(graph.getOutDegree(src));
    assert(d_src != 0.0); // should never be zero since self-loop added for each vertex
    d_src       = 1.0 / sqrt(d_src);
    auto start  = graph.edge_begin(src);
    index_t end = graph.edge_end(src);
    for (index_t e = start; e != end; e++) {
      index_t dst = graph.getEdgeDst(e);
      // if (dst >= n) printf("src=%d, dst=%d, e=%d, start=%d, end=%d\n", src,
      // dst, e, start, end);
      assert(dst < n);
      float_t d_dst = float_t(graph.getOutDegree(dst));
      assert(d_dst != 0.0);
      d_dst       = 1.0 / sqrt(d_dst);
      norm_fac[e] = d_src * d_dst;
    }
  }
}

cublasHandle_t DistContext::cublas_handle_         = 0;
cusparseHandle_t DistContext::cusparse_handle_     = 0;
cusparseMatDescr_t DistContext::cusparse_matdescr_ = 0;
curandGenerator_t DistContext::curand_generator_   = 0;

DistContext::DistContext() : DistContext(true) {
  d_labels = NULL; 
  d_feats = NULL;
  d_labels_subg = NULL; 
  d_feats_subg = NULL;
  d_normFactors = NULL;
  d_normFactorsSub = NULL;
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
  CUSPARSE_CHECK(cusparseCreateMatDescr(&cusparse_matdescr_));
  CUSPARSE_CHECK(
      cusparseSetMatType(cusparse_matdescr_, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CHECK(
      cusparseSetMatIndexBase(cusparse_matdescr_, CUSPARSE_INDEX_BASE_ZERO));
  CURAND_CHECK(
      curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()));
}

DistContext::~DistContext() {
  if (cublas_handle_)
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (cusparse_handle_)
    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle_));
  if (cusparse_matdescr_)
    CUSPARSE_CHECK(cusparseDestroyMatDescr(cusparse_matdescr_));
  if (curand_generator_)
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  if (d_labels) CUDA_CHECK(cudaFree(d_labels));
  if (d_feats) CUDA_CHECK(cudaFree(d_feats));
  if (d_normFactors) CUDA_CHECK(cudaFree(d_normFactors));
  if (d_labels_subg) CUDA_CHECK(cudaFree(d_labels_subg));
  if (d_feats_subg) CUDA_CHECK(cudaFree(d_feats_subg));
  if (d_normFactorsSub) CUDA_CHECK(cudaFree(d_normFactorsSub));
}

size_t DistContext::read_labels(bool isSingleClass, std::string dataset_str) {
  num_classes = reader.read_labels(isSingleClass, h_labels);
  return num_classes;
}

size_t DistContext::read_features(std::string dataset_str) {
  feat_len = reader.read_features(h_feats);
  return feat_len;
}

size_t DistContext::read_masks(std::string dataset_str, std::string mask_type, size_t n, 
                               size_t& begin, size_t& end, mask_t* masks, DGraph* dGraph) {
  return reader.read_masks(mask_type, n, begin, end, masks);
}

//! allocate memory for subgraphs (don't actually build them)
void DistContext::allocateSubgraphs(int num_subgraphs, unsigned max_size) {
  this->partitionedSubgraphs.resize(num_subgraphs);
  for (int i = 0; i < num_subgraphs; i++) {
    this->partitionedSubgraphs[i] = new Graph();
    this->partitionedSubgraphs[i]->set_max_size(max_size);
  }
}

void DistContext::constructSubgraphLabels(size_t m, const mask_t* masks) {
  size_t labels_size = m;
  if (!usingSingleClass) labels_size = m * num_classes;
  h_labels_subg.resize(labels_size);
  size_t count = 0;
  for (size_t i = 0; i < this->partitionedGraph->size(); i++) {
    if (masks[i] == 1) {
      if (usingSingleClass) h_labels_subg[count] = h_labels[i];
      else std::copy(h_labels + i * num_classes, h_labels + (i + 1) * num_classes, 
                     &h_labels_subg[count * num_classes]);
      count++;
    }
  }
  if (d_labels_subg) uint8_free_device(d_labels_subg);
  uint8_malloc_device(labels_size, d_labels_subg);
  uint8_copy_device(labels_size, &h_labels_subg[0], d_labels_subg);
}

void DistContext::constructSubgraphFeatures(size_t m, const mask_t* masks) {
  //std::cout << "construct subgraph features (d_feats_subg: " << d_feats_subg << ") ... ";
  size_t count = 0;
  DistContext::h_feats_subg.resize(m * feat_len);
  for (size_t i = 0; i < this->partitionedGraph->size(); i++) {
    if (masks[i] == 1) {
      std::copy(h_feats + i * feat_len, h_feats + (i + 1) * feat_len, &h_feats_subg[count * feat_len]);
      count++;
    }
  }
  if (d_feats_subg) float_free_device(d_feats_subg);
  float_malloc_device(m * feat_len, d_feats_subg);
  float_copy_device(m * feat_len, &h_feats_subg[0], d_feats_subg);
  //std::cout << "Done\n";
}

void DistContext::constructNormFactorSub(int subgraphID) {
  Graph& graphToUse = *partitionedSubgraphs[subgraphID];
  auto n = graphToUse.size();
  //std::cout << "Pre-computing subgraph normalization factor (n=" << n << ") ... ";

 #ifdef USE_CUSPARSE
  auto nnz = graphToUse.sizeEdges();
  float_malloc_device(nnz, d_normFactorsSub);
  init_const_gpu(nnz, 0.0, d_normFactors);
  norm_factor_computing_edge<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, graphToUse, d_normFactorsSub);
#else
  float_malloc_device(n, d_normFactorsSub);
  norm_factor_computing_node<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, graphToUse, d_normFactorsSub);
#endif
  CudaTest("solving norm_factor_computing kernel failed");
  //std::cout << "Done\n";
}

void DistContext::constructNormFactor(deepgalois::Context* globalContext) {
  auto n = partitionedGraph->size();
  std::cout << "Pre-computing normalization factor (n=" << n << ") ... ";
  if (!is_selfloop_added) {
    std::cout << "Set -sl=1 to add selfloop\n";
    exit(0);
  }
#ifdef USE_CUSPARSE
  auto nnz = partitionedGraph->sizeEdges();
  CUDA_CHECK(cudaMalloc((void**)&d_normFactors, nnz * sizeof(float_t)));
  init_const_gpu(nnz, 0.0, d_normFactors);
  norm_factor_computing_edge<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, *partitionedGraph, d_normFactors);
#else
  CUDA_CHECK(cudaMalloc((void**)&d_normFactors, n * sizeof(float_t)));
  norm_factor_computing_node<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, *partitionedGraph, d_normFactors);
#endif
  CudaTest("solving norm_factor_computing kernel failed");
  std::cout << "Done\n";
}

/*
void DistContext::SetDevice(const int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) return;
  CUDA_CHECK(cudaSetDevice(device_id));
  if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_)
CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CURAND_CHECK(curandCreateGenerator(&curand_generator_,
CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_,
cluster_seedgen()));
}
*/
size_t DistContext::read_graph(std::string dataset, bool selfloop) {
  partitionedGraph = new DGraph();
#ifdef USE_CSRGRAPH
  std::string filename = path + dataset + ".csgr";
  GraphGPU g;
  g.read(filename.c_str(), false);
  if (selfloop) {
    g.add_selfloop();
    is_selfloop_added = selfloop;
  }
  g.copy_to_gpu(*partitionedGraph);
#else
  partitionedGraph->readGraph(dataset);
  if (selfloop) {
    partitionedGraph->add_selfloop();
    is_selfloop_added = selfloop;
  }
  partitionedGraph->copy_to_gpu();
#endif
  return partitionedGraph->size();
}

void DistContext::copy_data_to_device() {
  auto n = partitionedGraph->size();
  std::cout << "Copying labels and features to GPU memory. n = " << n << " ... ";
  if (usingSingleClass) {
    CUDA_CHECK(cudaMalloc((void**)&d_labels, n * sizeof(label_t)));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, n * sizeof(label_t), cudaMemcpyHostToDevice));
  } else {
    CUDA_CHECK(cudaMalloc((void**)&d_labels, n * num_classes * sizeof(label_t)));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, n * num_classes * sizeof(label_t), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMalloc((void**)&d_feats, n * feat_len * sizeof(float_t)));
  CUDA_CHECK(cudaMemcpy(d_feats, &h_feats[0], n * feat_len * sizeof(float_t), cudaMemcpyHostToDevice));
  // print_device_vector(10, d_feats, "d_feats");
  std::cout << "Done\n";
}

} // namespace deepgalois
