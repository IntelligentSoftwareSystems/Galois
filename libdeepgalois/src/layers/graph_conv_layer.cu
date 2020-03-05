#include "deepgalois/layers/graph_conv_layer.h"

// computing normalization factor for each vertex
__global__ void norm_factor_counting_node(int n, CSRGraph graph,
                                            float_t* norm_fac) {
  CUDA_KERNEL_LOOP(i, n) {
    float_t temp = sqrt(float_t(graph.getOutDegree(i)));
    if (temp == 0.0) norm_fac[i] = 0.0;
    else norm_fac[i] = 1.0 / temp;
  }
}

// TODO: make sure self-loop added for each vertex
// computing normalization factor for each edge
__global__ void norm_factor_counting_edge(int n, CSRGraph graph, float_t* norm_fac) {
  CUDA_KERNEL_LOOP(src, n) {
    float_t d_src = float_t(graph.getOutDegree(src));
    assert(d_src != 0.0); // should never be zero since self-loop added for each vertex
    d_src = 1.0 / sqrt(d_src);
    index_type start = graph.edge_begin(src);
    index_type end = graph.edge_end(src);
	for (index_type e = start; e != end; e++) {
      index_type dst = graph.getEdgeDst(e);
      float_t d_dst = float_t(graph.getOutDegree(dst));
      assert(d_dst != 0.0);
      d_dst = 1.0 / sqrt(d_dst);
      norm_fac[e] = d_src * d_dst;
    }
  }
}

namespace deepgalois {

void graph_conv_layer::init() {
  gconv_malloc_device(x, y, z, dropout_, dropout_mask, in_temp, out_temp, d_W, layer::d_weight_grad);
}

void graph_conv_layer::aggregate(size_t len, CSRGraph& g, const float_t* in, float_t* out) {
  #ifdef USE_CUSPARSE
  deepgalois::update_all_csrmm(y, context->graph_gpu, in_temp, in_grad, norm_, norm_factor);
  #else
  deepgalois::update_all(len, g, in, out, norm_, norm_factor);
  #endif
}

void graph_conv_layer::combine(size_t dim_x, size_t dim_y, const float_t* self, const float_t* neighbors, float_t* out) {
}

void graph_conv_layer::norm_factor_counting() {
  std::cout << "debug\n";
  int n = x;//context->graph_gpu.nnodes;
  std::cout << "Pre-computing normalization factor (n=" << n << ") ... ";
#ifdef USE_CUSPARSE
  int nnz = context->graph_gpu.nedges;
  CUDA_CHECK(cudaMalloc((void**)&norm_factor, nnz * sizeof(float_t)));
  init_const_kernel<<<CUDA_GET_BLOCKS(nnz), CUDA_NUM_THREADS>>>(nnz, 0.0, norm_factor);
  norm_factor_counting_edge<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, context->graph_gpu, norm_factor);
#else
  CUDA_CHECK(cudaMalloc((void**)&norm_factor, n * sizeof(float_t)));
  norm_factor_counting_node<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, context->graph_gpu, norm_factor);
#endif
  CudaTest("solving norm_factor_counting kernel failed");
  std::cout << "Done\n";
}

// GPU forward: compute output features
void graph_conv_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  //assert(y <= 128); // currently only support feature length <= 128
  init_const_gpu(x*z, 0.0, out_temp);
  if (dropout_ && phase_ == deepgalois::net_phase::train) {
    dropout_gpu(x * y, scale_, dropout_rate_, in_data, dropout_mask, in_temp);
    sgemm_gpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_temp, d_W, 0.0, out_temp);
  } else sgemm_gpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_data, d_W, 0.0, out_temp);
  graph_conv_layer::aggregate(z, context->graph_gpu, out_temp, out_data);
  if (act_) relu_gpu(x * z, out_data, out_data);
}

// GPU backward: compute input gradients (in_grad) and weight gradients (d_weight_grad)
void graph_conv_layer::back_propagation(const float_t* in_data,
                                        const float_t* out_data,
                                        float_t* out_grad, float_t* in_grad) {
  if (act_) d_relu_gpu(x * z, out_grad, out_data, out_temp);
  else copy_gpu(x * z, out_grad, out_temp);
  if (level_ != 0) {
    sgemm_gpu(CblasNoTrans, CblasTrans, x, y, z, 1.0, out_temp, d_W, 0.0, in_temp);
#ifdef USE_CUSPARSE
    update_all_csrmm(y, context->graph_gpu, in_temp, in_grad, true, norm_factor);
#else
    update_all(y, context->graph_gpu, in_temp, in_grad, true, norm_factor);
#endif
    if (dropout_) d_dropout_gpu(x * y, scale_, dropout_rate_, in_grad, dropout_mask, in_grad);
  }
  sgemm_gpu(CblasTrans, CblasNoTrans, y, z, x, 1.0, in_data, out_temp, 0.0, layer::d_weight_grad);
}
} // namespace
