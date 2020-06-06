#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "tc_cuda.h"
#include "galois/runtime/cuda/DeviceEdgeSync.h"

void dump_memory_info(const char *s, int netId) {
  size_t total, free;

  if(cudaMemGetInfo(&free, &total) == cudaSuccess) {
    printf("[%d] GPU_memory_total_%s %zu\n", netId, s, total);
    printf("[%d] GPU_memory_free_%s %zu\n", netId, s, free);
  }
}

struct CUDA_Context : public CUDA_Context_Common_Edges {
};

struct CUDA_Context* get_CUDA_context(int id) {
	struct CUDA_Context* ctx;
	ctx = (struct CUDA_Context* ) calloc(1, sizeof(struct CUDA_Context));
	ctx->id = id;
	return ctx;
}

bool init_CUDA_context(struct CUDA_Context* ctx, int device) {
	return init_CUDA_context_common_edges(ctx, device);
}

void load_graph_CUDA(struct CUDA_Context* ctx, EdgeMarshalGraph &g, unsigned num_hosts) {
	dump_memory_info("start", ctx->id);
	load_graph_CUDA_common_edges(ctx, g, num_hosts, false);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
}
