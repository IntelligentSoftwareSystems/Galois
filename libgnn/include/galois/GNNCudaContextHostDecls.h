#pragma once
#include "galois/cuda/HostDecls.h"

extern int gpudevice;

void load_graph_CUDA_GNN(struct CUDA_Context* ctx, PartitionedGraphInfo& g,
                         unsigned num_hosts);
void resize_CUDA_layer_vector(struct CUDA_Context* ctx, size_t num_layers);
void resize_CUDA_bitset(struct CUDA_Context* ctx, size_t bitset_size);
void init_CUDA_layer_vector_meta_obj(struct CUDA_Context* ctx,
                                     unsigned layer_number, unsigned num_hosts,
                                     unsigned nnodes, size_t infl_in_size,
                                     size_t infl_out_size);

namespace galois {
void batch_get_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, size_t* buf_size,
    DataCommMode* mode, size_t column_size, unsigned layer_number);
void batch_get_node_layer_input_matrix_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* buf,
                                            size_t column_size,
                                            unsigned layer_number);
void batch_aggregate_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number);
void batch_aggregate_mirror_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number);
void batch_set_node_layer_input_matrix_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* buf,
                                            DataCommMode mode,
                                            size_t column_size,
                                            unsigned layer_number);
void batch_set_mirror_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number);
void batch_get_reset_node_layer_input_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, size_t* buf_size,
    DataCommMode* mode, size_t column_size, unsigned layer_number);
void batch_get_reset_node_layer_input_matrix_cuda(struct CUDA_Context* ctx,
                                                  unsigned from_id,
                                                  uint8_t* buf,
                                                  size_t column_size,
                                                  unsigned layer_number);
void batch_get_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, size_t* buf_size,
    DataCommMode* mode, size_t column_size, unsigned layer_number);
void batch_get_node_layer_output_matrix_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* buf,
                                             size_t column_size,
                                             unsigned layer_number);
void batch_aggregate_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number);
void batch_aggregate_mirror_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number);
void batch_set_node_layer_output_matrix_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* buf,
                                             DataCommMode mode,
                                             size_t column_size,
                                             unsigned layer_number);
void batch_set_mirror_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, DataCommMode mode,
    size_t column_size, unsigned layer_number);
void batch_get_reset_node_layer_output_matrix_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint8_t* buf, size_t* buf_size,
    DataCommMode* mode, size_t column_size, unsigned layer_number);
void batch_get_reset_node_layer_output_matrix_cuda(struct CUDA_Context* ctx,
                                                   unsigned from_id,
                                                   uint8_t* buf,
                                                   size_t column_size,
                                                   unsigned layer_number);
void get_bitset_graph_aggregate_cuda(struct CUDA_Context* ctx,
                                     uint64_t* bitset_compute);

void bitset_graph_aggregate_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                       size_t end);

void cudaSetLayerInputOutput(struct CUDA_Context* ctx, GNNFloat* layer_matrix,
                             size_t column_size, size_t num_nodes,
                             unsigned layer_number);
size_t getLayerInputMatrixColumnSize(struct CUDA_Context* ctx,
                                     unsigned layer_number);
size_t getLayerOutputMatrixColumnSize(struct CUDA_Context* ctx,
                                      unsigned layer_number);
} // namespace galois
