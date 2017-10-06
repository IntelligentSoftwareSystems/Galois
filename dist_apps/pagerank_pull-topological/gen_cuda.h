#pragma once
#include "galois/cuda/cuda_mtypes.h"
#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/cuda_context_decl.h"

float get_node_delta_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_delta_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_delta_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
bool min_node_delta_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void batch_get_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_mirror_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i);
void batch_get_reset_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode, float i);
void batch_set_mirror_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_set_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);

void get_bitset_nout_cuda(struct CUDA_Context *ctx, unsigned long long int *bitset_compute);
void bitset_nout_reset_cuda(struct CUDA_Context *ctx);
void bitset_nout_reset_cuda(struct CUDA_Context *ctx, size_t begin, size_t end);
uint32_t get_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void add_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
bool min_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void batch_get_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_mirror_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v, uint32_t i);
void batch_get_reset_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i);
void batch_set_mirror_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);

void get_bitset_residual_cuda(struct CUDA_Context *ctx, unsigned long long int *bitset_compute);
void bitset_residual_reset_cuda(struct CUDA_Context *ctx);
void bitset_residual_reset_cuda(struct CUDA_Context *ctx, size_t begin, size_t end);
float get_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
bool min_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void batch_get_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_mirror_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i);
void batch_get_reset_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode, float i);
void batch_set_mirror_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_set_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);

void bitset_value_clear_cuda(struct CUDA_Context *ctx);
float get_node_value_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
bool min_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void batch_get_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_mirror_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i);
void batch_get_reset_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode, float i);
void batch_set_mirror_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_set_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);

void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void InitializeGraph_all_cuda(struct CUDA_Context *ctx);
void PageRank_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void PageRank_all_cuda(struct CUDA_Context *ctx);
void PageRank_delta_cuda(unsigned int __begin, unsigned int __end, int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context *ctx);
void PageRank_delta_all_cuda(int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context *ctx);
void ResetGraph_cuda(unsigned int __begin, unsigned int __end, const float & local_alpha, struct CUDA_Context *ctx);
void ResetGraph_all_cuda(const float & local_alpha, struct CUDA_Context *ctx);
