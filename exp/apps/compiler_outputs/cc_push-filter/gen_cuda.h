#pragma once
#include "Galois/Cuda/cuda_mtypes.h"

struct CUDA_Context;

struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g);

void reset_CUDA_context(struct CUDA_Context *ctx);
unsigned int get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
unsigned int get_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void ConnectedComp_cuda(int & __retval, struct CUDA_Context *ctx);
void FirstItr_ConnectedComp_cuda(struct CUDA_Context *ctx);
void InitializeGraph_cuda(struct CUDA_Context *ctx);
