#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "gen_cuda.h"

#ifdef __GALOIS_CUDA_CHECK_ERROR__
#define check_cuda_kernel check_cuda(cudaDeviceSynchronize()); check_cuda(cudaGetLastError());
#else
#define check_cuda_kernel  
#endif

#ifndef __GALOIS_CUDA_WORKLIST_DUPLICATION_FACTOR__
#define __GALOIS_CUDA_WORKLIST_DUPLICATION_FACTOR__ 2
#endif

struct CUDA_Context {
	int device;
	int id;
	size_t nowned;
	size_t g_offset;
	CSRGraphTy hg;
	CSRGraphTy gg;
	Shared<unsigned int> dist_current;
	Shared<int> p_retval;
	Worklist2 in_wl;
	Worklist2 out_wl;
	struct CUDA_Worklist *shared_wl;
	Any any_retval;
};

unsigned int get_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID) {
	unsigned int *dist_current = ctx->dist_current.cpu_rd_ptr();
	return dist_current[LID];
}

void set_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *dist_current = ctx->dist_current.cpu_wr_ptr();
	dist_current[LID] = v;
}

void add_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *dist_current = ctx->dist_current.cpu_wr_ptr();
	dist_current[LID] += v;
}

void min_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *dist_current = ctx->dist_current.cpu_wr_ptr();
	if (dist_current[LID] > v)
		dist_current[LID] = v;
}

struct CUDA_Context *get_CUDA_context(int id) {
	struct CUDA_Context *ctx;
	ctx = (struct CUDA_Context *) calloc(1, sizeof(struct CUDA_Context));
	ctx->id = id;
	return ctx;
}

bool init_CUDA_context(struct CUDA_Context *ctx, int device) {
	struct cudaDeviceProp dev;
	if(device == -1) {
		check_cuda(cudaGetDevice(&device));
	} else {
		int count;
		check_cuda(cudaGetDeviceCount(&count));
		if(device > count) {
			fprintf(stderr, "Error: Out-of-range GPU %d specified (%d total GPUs)", device, count);
			return false;
		}
		check_cuda(cudaSetDevice(device));
	}
	ctx->device = device;
	check_cuda(cudaGetDeviceProperties(&dev, device));
	fprintf(stderr, "%d: Using GPU %d: %s\n", ctx->id, device, dev.name);
	return true;
}

void load_graph_CUDA(struct CUDA_Context *ctx, struct CUDA_Worklist *wl, MarshalGraph &g) {
	CSRGraphTy &graph = ctx->hg;
	ctx->nowned = g.nowned;
	assert(ctx->id == g.id);
	graph.nnodes = g.nnodes;
	graph.nedges = g.nedges;
	if(!graph.allocOnHost()) {
		fprintf(stderr, "Unable to alloc space for graph!");
		exit(1);
	}
	memcpy(graph.row_start, g.row_start, sizeof(index_type) * (g.nnodes + 1));
	memcpy(graph.edge_dst, g.edge_dst, sizeof(index_type) * g.nedges);
	if(g.node_data) memcpy(graph.node_data, g.node_data, sizeof(node_data_type) * g.nnodes);
	if(g.edge_data) memcpy(graph.edge_data, g.edge_data, sizeof(edge_data_type) * g.nedges);
	graph.copy_to_gpu(ctx->gg);
	ctx->dist_current.alloc(graph.nnodes);
	ctx->in_wl = Worklist2(__GALOIS_CUDA_WORKLIST_DUPLICATION_FACTOR__*graph.nedges);
	ctx->out_wl = Worklist2(__GALOIS_CUDA_WORKLIST_DUPLICATION_FACTOR__*graph.nedges);
	wl->num_in_items = -1;
	wl->num_out_items = -1;
	wl->in_items = ctx->in_wl.wl;
	wl->out_items = ctx->out_wl.wl;
	ctx->shared_wl = wl;
	ctx->p_retval = Shared<int>(1);
	printf("load_graph_GPU: %d owned nodes of total %d resident, %d edges\n", ctx->nowned, graph.nnodes, graph.nedges);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context *ctx) {
	ctx->dist_current.zero_gpu();
}

void kernel_sizing(CSRGraphTy & g, dim3 &blocks, dim3 &threads) {
	threads.x = 256;
	threads.y = threads.z = 1;
	blocks.x = 14 * 8;
	blocks.y = blocks.z = 1;
}

