#pragma once
#include "Galois/Runtime/Cuda/cuda_mtypes.h"

struct CUDA_Context;

struct CUDA_Worklist {
	int *in_items;
	int num_in_items;
	int *out_items;
	int num_out_items;
};

struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, struct CUDA_Worklist *wl, MarshalGraph &g);

void reset_CUDA_context(struct CUDA_Context *ctx);
unsigned int get_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
float get_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void min_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
float get_node_value_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void min_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void InitializeGraph_cuda(const float & local_alpha, struct CUDA_Context *ctx);
void PageRank_cuda(const float & local_alpha, float local_tolerance, struct CUDA_Context *ctx);
void ResetGraph_cuda(struct CUDA_Context *ctx);
