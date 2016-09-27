#pragma once
#include "Galois/Runtime/Cuda/cuda_mtypes.h"

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
void batch_get_reset_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v, int i);
void batch_set_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v);
void batch_add_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v);
void batch_min_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, int *v);
float get_node_sum_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_sum_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_sum_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void min_node_sum_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void batch_get_node_sum_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_reset_node_sum_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i);
void batch_set_node_sum_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_add_node_sum_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_min_node_sum_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
float get_node_value_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void min_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void batch_get_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_reset_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i);
void batch_set_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_add_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_min_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void InitializeGraph_cuda(const float & local_alpha, struct CUDA_Context *ctx);
void PageRank_cuda(int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context *ctx);
void PageRank_partial_cuda(struct CUDA_Context *ctx);
void ResetGraph_cuda(struct CUDA_Context *ctx);
