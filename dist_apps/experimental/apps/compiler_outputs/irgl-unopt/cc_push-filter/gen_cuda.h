#pragma once
#include "galois/cuda/cuda_mtypes.h"

struct CUDA_Context;

struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g, unsigned num_hosts);

void reset_CUDA_context(struct CUDA_Context *ctx);
unsigned int get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_reset_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_set_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_add_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_min_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
unsigned int get_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_reset_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_set_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_add_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_min_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void ConnectedComp_cuda(unsigned int __begin, unsigned int __end, int & __retval, struct CUDA_Context *ctx);
void ConnectedComp_all_cuda(int & __retval, struct CUDA_Context *ctx);
void FirstItr_ConnectedComp_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void FirstItr_ConnectedComp_all_cuda(struct CUDA_Context *ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void InitializeGraph_all_cuda(struct CUDA_Context *ctx);
