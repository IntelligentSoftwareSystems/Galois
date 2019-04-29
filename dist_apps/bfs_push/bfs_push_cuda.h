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
void batch_get_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, uint32_t i);
void batch_get_reset_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, uint32_t i);
void batch_set_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode);
void batch_set_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_reset_node_dist_current_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v);

void get_bitset_dist_old_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_dist_old_reset_cuda(struct CUDA_Context* ctx);
void bitset_dist_old_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
uint32_t get_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void add_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
bool min_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void batch_get_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, uint32_t i);
void batch_get_reset_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, uint32_t i);
void batch_set_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode);
void batch_set_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_reset_node_dist_old_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v);

void BFS_cuda(unsigned int __begin, unsigned int __end, unsigned int & active_vertices, unsigned int & work_items, uint32_t local_priority, struct CUDA_Context* ctx);
void BFSSanityCheck_cuda(unsigned int __begin, unsigned int __end, uint64_t & DGAccumulator_sum, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context* ctx);
void BFSSanityCheck_allNodes_cuda(uint64_t & DGAccumulator_sum, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context* ctx);
void BFSSanityCheck_masterNodes_cuda(uint64_t & DGAccumulator_sum, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context* ctx);
void BFSSanityCheck_nodesWithEdges_cuda(uint64_t & DGAccumulator_sum, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context* ctx);
void BFS_allNodes_cuda(unsigned int & active_vertices, unsigned int & work_items, uint32_t local_priority, struct CUDA_Context* ctx);
void BFS_masterNodes_cuda(unsigned int & active_vertices, unsigned int & work_items, uint32_t local_priority, struct CUDA_Context* ctx);
void BFS_nodesWithEdges_cuda(unsigned int & active_vertices, unsigned int & work_items, uint32_t local_priority, struct CUDA_Context* ctx);
void FirstItr_BFS_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context* ctx);
void FirstItr_BFS_allNodes_cuda(struct CUDA_Context* ctx);
void FirstItr_BFS_masterNodes_cuda(struct CUDA_Context* ctx);
void FirstItr_BFS_nodesWithEdges_cuda(struct CUDA_Context* ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context* ctx);
void InitializeGraph_allNodes_cuda(const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context* ctx);
void InitializeGraph_masterNodes_cuda(const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context* ctx);
void InitializeGraph_nodesWithEdges_cuda(const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context* ctx);
