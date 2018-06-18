#pragma once
#include "cuda_mtypes.h"

struct CUDA_Context;

/* infrastructure */
struct CUDA_Context* get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context* ctx, int device);
void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph& g);
void test_cuda(struct CUDA_Context* ctx);

/* application */
void initialize_graph_cuda(struct CUDA_Context* ctx);
void test_graph_cuda(struct CUDA_Context* ctx);
void pagerank_cuda(struct CUDA_Context* ctx);

/* application messages */
void setNodeValue_CUDA(struct CUDA_Context* ctx, unsigned LID, unsigned v);
void setNodeAttr_CUDA(struct CUDA_Context* ctx, unsigned LID, unsigned nout);
unsigned getNodeValue_CUDA(struct CUDA_Context* ctx, unsigned LID);
unsigned getNodeAttr_CUDA(struct CUDA_Context* ctx, unsigned LID);
unsigned getNodeAttr2_CUDA(struct CUDA_Context* ctx, unsigned LID);
void setNodeAttr2_CUDA(struct CUDA_Context* ctx, unsigned LID, unsigned nout);
