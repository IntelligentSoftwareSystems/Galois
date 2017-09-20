#pragma once
#include "galois/Runtime/Cuda/cuda_mtypes.h"

struct CUDA_Context;

struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g, unsigned num_hosts);

void reset_CUDA_context(struct CUDA_Context *ctx);
unsigned int get_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_reset_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_set_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_add_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_min_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
unsigned int get_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_reset_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_set_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_add_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_min_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void BFS_cuda(unsigned int __begin, unsigned int __end, int & __retval, struct CUDA_Context *ctx);
void BFS_all_cuda(int & __retval, struct CUDA_Context *ctx);
void FirstItr_BFS_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void FirstItr_BFS_all_cuda(struct CUDA_Context *ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context *ctx);
void InitializeGraph_all_cuda(const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context *ctx);
