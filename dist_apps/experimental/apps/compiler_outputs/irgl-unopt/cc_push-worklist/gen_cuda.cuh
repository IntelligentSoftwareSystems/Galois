#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "gen_cuda.h"
#include "galois/runtime/Cuda/cuda_helpers.h"

#ifdef __GALOIS_CUDA_CHECK_ERROR__
#define check_cuda_kernel check_cuda(cudaDeviceSynchronize()); check_cuda(cudaGetLastError());
#else
#define check_cuda_kernel check_cuda(cudaGetLastError());
#endif

struct CUDA_Context {
	int device;
	int id;
	unsigned int nowned;
	CSRGraphTy hg;
	CSRGraphTy gg;
	unsigned int *num_master_nodes; // per host
	Shared<unsigned int> *master_nodes; // per host
	unsigned int *num_mirror_nodes; // per host
	Shared<unsigned int> *mirror_nodes; // per host
	Shared<unsigned int> comp_current;
	Shared<unsigned int> *master_comp_current; // per host
	Shared<unsigned int> *mirror_comp_current; // per host
	Worklist2 in_wl;
	Worklist2 out_wl;
	struct CUDA_Worklist *shared_wl;
};

unsigned int get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID) {
	unsigned int *comp_current = ctx->comp_current.cpu_rd_ptr();
	return comp_current[LID];
}

void set_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *comp_current = ctx->comp_current.cpu_wr_ptr();
	comp_current[LID] = v;
}

void add_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *comp_current = ctx->comp_current.cpu_wr_ptr();
	comp_current[LID] += v;
}

void min_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *comp_current = ctx->comp_current.cpu_wr_ptr();
	if (comp_current[LID] > v)
		comp_current[LID] = v;
}

void batch_get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_subset<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_comp_current[from_id].gpu_wr_ptr(true), ctx->comp_current.gpu_rd_ptr());
	check_cuda_kernel;
	memcpy(v, ctx->master_comp_current[from_id].cpu_rd_ptr(), sizeof(unsigned int) * ctx->num_master_nodes[from_id]);
}

void batch_get_mirror_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_subset<unsigned int> <<<blocks, threads>>>(ctx->num_mirror_nodes[from_id], ctx->mirror_nodes[from_id].gpu_rd_ptr(), ctx->mirror_comp_current[from_id].gpu_wr_ptr(true), ctx->comp_current.gpu_rd_ptr());
	check_cuda_kernel;
	memcpy(v, ctx->mirror_comp_current[from_id].cpu_rd_ptr(), sizeof(unsigned int) * ctx->num_mirror_nodes[from_id]);
}

void batch_get_reset_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_reset_subset<unsigned int> <<<blocks, threads>>>(ctx->num_mirror_nodes[from_id], ctx->mirror_nodes[from_id].gpu_rd_ptr(), ctx->mirror_comp_current[from_id].gpu_wr_ptr(true), ctx->comp_current.gpu_rd_ptr(), i);
	check_cuda_kernel;
	memcpy(v, ctx->mirror_comp_current[from_id].cpu_rd_ptr(), sizeof(unsigned int) * ctx->num_mirror_nodes[from_id]);
}

void batch_set_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->mirror_comp_current[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * ctx->num_mirror_nodes[from_id]);
	batch_set_subset<unsigned int> <<<blocks, threads>>>(ctx->num_mirror_nodes[from_id], ctx->mirror_nodes[from_id].gpu_rd_ptr(), ctx->mirror_comp_current[from_id].gpu_rd_ptr(), ctx->comp_current.gpu_wr_ptr());
	check_cuda_kernel;
}

void batch_add_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_comp_current[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * ctx->num_master_nodes[from_id]);
	batch_add_subset<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_comp_current[from_id].gpu_rd_ptr(), ctx->comp_current.gpu_wr_ptr());
	check_cuda_kernel;
}

void batch_min_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_comp_current[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * ctx->num_master_nodes[from_id]);
	batch_min_subset<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_comp_current[from_id].gpu_rd_ptr(), ctx->comp_current.gpu_wr_ptr());
	check_cuda_kernel;
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

void load_graph_CUDA(struct CUDA_Context *ctx, struct CUDA_Worklist *wl, double wl_dup_factor, MarshalGraph &g, unsigned num_hosts) {
	CSRGraphTy &graph = ctx->hg;
	ctx->nowned = g.nowned;
	assert(ctx->id == g.id);
	graph.nnodes = g.nnodes;
	graph.nedges = g.nedges;
	if(!graph.allocOnHost(!g.edge_data)) {
		fprintf(stderr, "Unable to alloc space for graph!");
		exit(1);
	}
	memcpy(graph.row_start, g.row_start, sizeof(index_type) * (g.nnodes + 1));
	memcpy(graph.edge_dst, g.edge_dst, sizeof(index_type) * g.nedges);
	if(g.node_data) memcpy(graph.node_data, g.node_data, sizeof(node_data_type) * g.nnodes);
	if(g.edge_data) memcpy(graph.edge_data, g.edge_data, sizeof(edge_data_type) * g.nedges);
	ctx->num_master_nodes = (unsigned int *) calloc(num_hosts, sizeof(unsigned int));
	memcpy(ctx->num_master_nodes, g.num_master_nodes, sizeof(unsigned int) * num_hosts);
	ctx->master_nodes = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->master_comp_current = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->num_master_nodes[h] > 0) {
			ctx->master_nodes[h].alloc(ctx->num_master_nodes[h]);
			memcpy(ctx->master_nodes[h].cpu_wr_ptr(), g.master_nodes[h], sizeof(unsigned int) * ctx->num_master_nodes[h]);
			ctx->master_comp_current[h].alloc(ctx->num_master_nodes[h]);
		}
	}
	ctx->num_mirror_nodes = (unsigned int *) calloc(num_hosts, sizeof(unsigned int));
	memcpy(ctx->num_mirror_nodes, g.num_mirror_nodes, sizeof(unsigned int) * num_hosts);
	ctx->mirror_nodes = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->mirror_comp_current = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->num_mirror_nodes[h] > 0) {
			ctx->mirror_nodes[h].alloc(ctx->num_mirror_nodes[h]);
			memcpy(ctx->mirror_nodes[h].cpu_wr_ptr(), g.mirror_nodes[h], sizeof(unsigned int) * ctx->num_mirror_nodes[h]);
			ctx->mirror_comp_current[h].alloc(ctx->num_mirror_nodes[h]);
		}
	}
	graph.copy_to_gpu(ctx->gg);
	ctx->comp_current.alloc(graph.nnodes);
	wl->max_size = wl_dup_factor*graph.nnodes;
	ctx->in_wl = Worklist2((size_t)wl->max_size);
	ctx->out_wl = Worklist2((size_t)wl->max_size);
	wl->num_in_items = -1;
	wl->num_out_items = -1;
	wl->in_items = ctx->in_wl.wl;
	wl->out_items = ctx->out_wl.wl;
	ctx->shared_wl = wl;
	printf("[%d] load_graph_GPU: %d owned nodes of total %d resident, %d edges\n", ctx->id, ctx->nowned, graph.nnodes, graph.nedges);
	printf("[%d] load_graph_GPU: worklist size %d\n", ctx->id, (size_t)wl_dup_factor*graph.nnodes);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context *ctx) {
	ctx->comp_current.zero_gpu();
}

void kernel_sizing(CSRGraphTy & g, dim3 &blocks, dim3 &threads) {
	threads.x = 256;
	threads.y = threads.z = 1;
	blocks.x = 14 * 8;
	blocks.y = blocks.z = 1;
}

