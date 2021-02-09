#include <stdlib.h>
#include "gg.h"
#include "ggcuda.h"
#include "galois/cuda/Context.h"
#include "galois/GNNTypes.h"
#include "galois/runtime/cuda/DeviceSync.h"
#include "galois/GNNCudaContextHostDecls.h"

// The forward declaration is in the original Context.h file; as long as
// pointers to it are used it shouldn't be an issue (since space usage is
// unknown at that point)
struct CUDA_Context : public CUDA_Context_Common {
  // TODO to arrays: each context handles all layers of the graph
  // Possible to add a "layer" argument to the below functions?
  std::vector<struct CUDA_Context_Field<galois::GNNFloat>> layer_input_matrix;
  std::vector<struct CUDA_Context_Field<galois::GNNFloat>> layer_output_matrix;
  std::vector<size_t> layer_input_matrix_column_size;
  std::vector<size_t> layer_output_matrix_column_size;
};

//! Allocates a new CUDA context
//! Note: caller is responsible for freeing it
struct CUDA_Context* get_CUDA_context(int id) {
  struct CUDA_Context* ctx =
      (struct CUDA_Context*)calloc(1, sizeof(struct CUDA_Context));
  ctx->id = id;
  return ctx;
}

bool init_CUDA_context(struct CUDA_Context* ctx, int device) {
  return init_CUDA_context_common(ctx, device);
}

void resize_CUDA_layer_vector(struct CUDA_Context* ctx, size_t num_layers) {
  ctx->layer_output_matrix.resize(num_layers);
  ctx->layer_output_matrix_column_size.resize(num_layers);
  ctx->layer_input_matrix.resize(num_layers);
  ctx->layer_input_matrix_column_size.resize(num_layers);
}

void load_graph_CUDA_GNN(struct CUDA_Context* ctx, PartitionedGraphInfo& g_info,
                         unsigned num_hosts) {
  size_t mem_usage = mem_usage_CUDA_common(g_info, num_hosts);
  printf("[%d] Host memory for communication context: (%3u B) %3u MB\n",
         ctx->id, mem_usage, mem_usage / 1048756);

  // TODO This is expensive; is it required? Can we get away with less?
  // should only need one copy of mirror/masters for entire execution,
  // not per layer
  // graph does not need to be copied either since that's handled elsewhere
  // (gpu object on GNNGraph)
  load_graph_CUDA_common(ctx, g_info, num_hosts);
}

void init_CUDA_layer_vector_meta_obj(struct CUDA_Context* ctx,
                                     unsigned layer_number, unsigned num_hosts,
                                     unsigned nnodes, size_t infl_in_size,
                                     size_t infl_out_size) {
  ctx->layer_input_matrix_column_size[layer_number] = infl_in_size;
  load_graph_CUDA_field_inflating(ctx, &ctx->layer_input_matrix[layer_number],
                                  num_hosts, nnodes, infl_in_size, false);
  ctx->layer_output_matrix_column_size[layer_number] = infl_out_size;
  load_graph_CUDA_field_inflating(ctx, &ctx->layer_output_matrix[layer_number],
                                  num_hosts, nnodes, infl_out_size, false);
}

////////// layer_input_matrix (forward) synchronization function ///////////////

namespace galois {
void batch_get_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, size_t* buf_size,
    DataCommMode* mode, size_t column_size, unsigned layer_number) {
  batch_get_shared_field<GNNFloat, sharedMaster, false>(
      ctx, &ctx->layer_input_matrix[layer_number], from_id, buf, buf_size, mode,
      column_size);
}

void batch_get_node_layer_input_matrix_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* buf,
                                            size_t column_size,
                                            unsigned layer_number) {
  batch_get_shared_field<GNNFloat, sharedMaster, false>(
      ctx, &ctx->layer_input_matrix[layer_number], from_id, buf, column_size);
}

void batch_aggregate_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number) {
  batch_set_shared_field<GNNFloat, sharedMaster, addOp>(
      ctx, &ctx->layer_input_matrix[layer_number], from_id, buf, mode,
      column_size);
}

void batch_aggregate_mirror_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number) {
  batch_set_shared_field<GNNFloat, sharedMirror, addOp>(
      ctx, &ctx->layer_input_matrix[layer_number], from_id, buf, mode,
      column_size);
}

void batch_set_node_layer_input_matrix_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* buf,
                                            DataCommMode mode,
                                            size_t column_size,
                                            unsigned layer_number) {
  batch_set_shared_field<GNNFloat, sharedMaster, setOp>(
      ctx, &ctx->layer_input_matrix[layer_number], from_id, buf, mode,
      column_size);
}

void batch_set_mirror_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number) {
  batch_set_shared_field<GNNFloat, sharedMirror, setOp>(
      ctx, &ctx->layer_input_matrix[layer_number], from_id, buf, mode,
      column_size);
}

void batch_get_reset_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, size_t* buf_size,
    DataCommMode* mode, size_t column_size, unsigned layer_number) {
  batch_get_shared_field<GNNFloat, sharedMaster, true>(
      ctx, &ctx->layer_input_matrix[layer_number], from_id, buf, buf_size, mode,
      column_size);
}

void batch_get_reset_node_layer_input_matrix_cuda(struct CUDA_Context* ctx,
                                                  unsigned from_id,
                                                  uint8_t* buf,
                                                  size_t column_size,
                                                  unsigned layer_number) {
  batch_get_shared_field<GNNFloat, sharedMaster, true>(
      ctx, &ctx->layer_input_matrix[layer_number], from_id, buf, column_size);
}

////////// layer_output_matrix (backward) synchronization function /////////////

void batch_get_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, size_t* buf_size,
    DataCommMode* mode, size_t column_size, unsigned layer_number) {
  batch_get_shared_field<GNNFloat, sharedMaster, false>(
      ctx, &ctx->layer_output_matrix[layer_number], from_id, buf, buf_size,
      mode, column_size);
}

void batch_get_node_layer_output_matrix_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* buf,
                                             size_t column_size,
                                             unsigned layer_number) {
  batch_get_shared_field<GNNFloat, sharedMaster, false>(
      ctx, &ctx->layer_output_matrix[layer_number], from_id, buf, column_size);
}

void batch_aggregate_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number) {
  batch_set_shared_field<GNNFloat, sharedMaster, addOp>(
      ctx, &ctx->layer_output_matrix[layer_number], from_id, buf, mode,
      column_size);
}

void batch_aggregate_mirror_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number) {
  batch_set_shared_field<GNNFloat, sharedMirror, addOp>(
      ctx, &ctx->layer_output_matrix[layer_number], from_id, buf, mode,
      column_size);
}

void batch_set_node_layer_output_matrix_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* buf,
                                             DataCommMode mode,
                                             size_t column_size,
                                             unsigned layer_number) {
  batch_set_shared_field<GNNFloat, sharedMaster, setOp>(
      ctx, &ctx->layer_output_matrix[layer_number], from_id, buf, mode,
      column_size);
}

void batch_set_mirror_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number) {
  batch_set_shared_field<GNNFloat, sharedMirror, setOp>(
      ctx, &ctx->layer_output_matrix[layer_number], from_id, buf, mode,
      column_size);
}

void batch_get_reset_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, size_t* buf_size,
    DataCommMode* mode, size_t column_size, unsigned layer_number) {
  batch_get_shared_field<GNNFloat, sharedMaster, true>(
      ctx, &ctx->layer_output_matrix[layer_number], from_id, buf, buf_size,
      mode, column_size);
}

void batch_get_reset_node_layer_output_matrix_cuda(struct CUDA_Context* ctx,
                                                   unsigned from_id,
                                                   uint8_t* buf,
                                                   size_t column_size,
                                                   unsigned layer_number) {
  batch_get_shared_field<GNNFloat, sharedMaster, true>(
      ctx, &ctx->layer_output_matrix[layer_number], from_id, buf, column_size);
}

void cudaSetLayerInputOutput(struct CUDA_Context* ctx, GNNFloat* layer_matrix,
                             size_t column_size, size_t num_nodes,
                             unsigned layer_number) {
  if (ctx->layer_input_matrix_column_size[layer_number] == column_size) {
    ctx->layer_input_matrix[layer_number].data.set_data(
        layer_matrix, column_size * num_nodes);
  } else if (ctx->layer_output_matrix_column_size[layer_number] ==
             column_size) {
    ctx->layer_output_matrix[layer_number].data.set_data(
        layer_matrix, column_size * num_nodes);
  }
}

size_t getLayerInputMatrixColumnSize(struct CUDA_Context* ctx,
                                     unsigned layer_number) {
  return ctx->layer_input_matrix_column_size[layer_number];
}

size_t getLayerOutputMatrixColumnSize(struct CUDA_Context* ctx,
                                      unsigned layer_number) {
  return ctx->layer_output_matrix_column_size[layer_number];
}
} // namespace galois
