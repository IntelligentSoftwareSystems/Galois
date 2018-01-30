#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

void get_bitset_dist_current_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_dist_current_reset_cuda(struct CUDA_Context* ctx);
void bitset_dist_current_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
uint32_t get_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void add_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
bool min_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void batch_get_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v);
void batch_get_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v);
void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v, uint32_t i);
void batch_get_reset_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i);
void batch_set_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);

void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context* ctx);
void InitializeGraph_allNodes_cuda(const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context* ctx);
void InitializeGraph_masterNodes_cuda(const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context* ctx);
void InitializeGraph_nodesWithEdges_cuda(const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context* ctx);
void SSSP_cuda(unsigned int __begin, unsigned int __end, unsigned int & DGAccumulator_accum, struct CUDA_Context* ctx);
void SSSPSanityCheck_cuda(unsigned int __begin, unsigned int __end, uint64_t & DGAccumulator_sum, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context* ctx);
void SSSPSanityCheck_allNodes_cuda(uint64_t & DGAccumulator_sum, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context* ctx);
void SSSPSanityCheck_masterNodes_cuda(uint64_t & DGAccumulator_sum, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context* ctx);
void SSSPSanityCheck_nodesWithEdges_cuda(uint64_t & DGAccumulator_sum, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context* ctx);
void SSSP_allNodes_cuda(unsigned int & DGAccumulator_accum, struct CUDA_Context* ctx);
void SSSP_masterNodes_cuda(unsigned int & DGAccumulator_accum, struct CUDA_Context* ctx);
void SSSP_nodesWithEdges_cuda(unsigned int & DGAccumulator_accum, struct CUDA_Context* ctx);
