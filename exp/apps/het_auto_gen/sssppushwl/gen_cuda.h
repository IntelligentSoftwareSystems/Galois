#pragma once
#include "Galois/Cuda/cuda_mtypes.h"

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
unsigned int get_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void InitializeGraph_cuda(int local_src_node, unsigned int local_infinity, struct CUDA_Context *ctx);
void SSSP_cuda(struct CUDA_Context *ctx);
