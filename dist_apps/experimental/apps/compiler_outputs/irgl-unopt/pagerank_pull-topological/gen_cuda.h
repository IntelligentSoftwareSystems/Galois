#pragma once
#include "galois/Runtime/Cuda/cuda_mtypes.h"

struct CUDA_Context;

struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g, unsigned num_hosts);

void reset_CUDA_context(struct CUDA_Context *ctx);
int get_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, int v);
void add_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, int v);
void min_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, int v);
void batch_get_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v);
void batch_get_mirror_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v);
void batch_get_reset_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v, int i);
void batch_set_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v);
void batch_add_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v);
void batch_min_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v);
float get_node_value_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void min_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void batch_get_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_mirror_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_reset_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i);
void batch_set_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_add_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_min_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, const float & local_alpha, struct CUDA_Context *ctx);
void InitializeGraph_all_cuda(const float & local_alpha, struct CUDA_Context *ctx);
void PageRank_cuda(unsigned int __begin, unsigned int __end, int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context *ctx);
void PageRank_all_cuda(int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context *ctx);
void ResetGraph_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void ResetGraph_all_cuda(struct CUDA_Context *ctx);
