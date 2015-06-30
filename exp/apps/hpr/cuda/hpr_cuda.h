#pragma once
#include "cuda_mtypes.h"

struct CUDA_Context;

struct CUDA_Context *get_CUDA_context();
void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g);

void test_cuda(struct CUDA_Context *ctx);

void initialize_graph_cuda(struct CUDA_Context *ctx);
