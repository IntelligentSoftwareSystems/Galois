#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "gen_cuda.h"
#include "Galois/Runtime/Cuda/cuda_helpers.h"
#include <math.h>

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
  Shared<DynamicBitset> is_updated;
	unsigned int *num_master_nodes; // per host
	Shared<unsigned int> *master_nodes; // per host
	unsigned int *num_slave_nodes; // per host
	Shared<unsigned int> *slave_nodes; // per host
	Shared<unsigned int> dist_current;
	Shared<unsigned int> *master_dist_current; // per host
	Shared<unsigned int> *slave_dist_current; // per host
	Shared<unsigned int> dist_old;
	Shared<unsigned int> *master_dist_old; // per host
	Shared<unsigned int> *slave_dist_old; // per host
  Shared<DynamicBitset> *is_master_updated;
  Shared<DynamicBitset> *is_slave_updated;
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

void batch_get_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bit_vector, unsigned int *v, size_t *v_size) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
  ctx->is_master_updated[from_id].cpu_rd_ptr()->clear();
	batch_get_subset_bitset<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->is_master_updated[from_id].gpu_rd_ptr(), ctx->is_updated.gpu_rd_ptr());
	check_cuda_kernel;
	batch_get_subset_conditional<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->is_master_updated[from_id].gpu_rd_ptr(), ctx->master_dist_current[from_id].gpu_wr_ptr(true), ctx->dist_current.gpu_rd_ptr());
	check_cuda_kernel;
  *v_size = ctx->is_master_updated[from_id].cpu_rd_ptr()->copy_to_cpu(bit_vector);
	memcpy(v, ctx->master_dist_current[from_id].cpu_rd_ptr(), sizeof(unsigned int) * (*v_size));
}

void batch_get_slave_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bit_vector, unsigned int *v, size_t *v_size) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
  ctx->is_slave_updated[from_id].cpu_rd_ptr()->clear();
	batch_get_subset_bitset<unsigned int> <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->is_slave_updated[from_id].gpu_rd_ptr(), ctx->is_updated.gpu_rd_ptr());
	check_cuda_kernel;
	batch_get_subset_conditional<unsigned int> <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->is_slave_updated[from_id].gpu_rd_ptr(), ctx->slave_dist_current[from_id].gpu_wr_ptr(true), ctx->dist_current.gpu_rd_ptr());
	check_cuda_kernel;
  *v_size = ctx->is_slave_updated[from_id].cpu_rd_ptr()->copy_to_cpu(bit_vector);
	memcpy(v, ctx->slave_dist_current[from_id].cpu_rd_ptr(), sizeof(unsigned int) * (*v_size));
}

void batch_get_reset_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bit_vector, unsigned int *v, size_t *v_size, unsigned int i) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
  ctx->is_slave_updated[from_id].cpu_rd_ptr()->clear();
	batch_get_subset_bitset<unsigned int> <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->is_slave_updated[from_id].gpu_rd_ptr(), ctx->is_updated.gpu_rd_ptr());
	check_cuda_kernel;
	batch_get_reset_subset_conditional<unsigned int> <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->is_slave_updated[from_id].gpu_rd_ptr(), ctx->slave_dist_current[from_id].gpu_wr_ptr(true), ctx->dist_current.gpu_rd_ptr(), i);
	check_cuda_kernel;
  *v_size = ctx->is_slave_updated[from_id].cpu_rd_ptr()->copy_to_cpu(bit_vector);
	memcpy(v, ctx->slave_dist_current[from_id].cpu_rd_ptr(), sizeof(unsigned int) * (*v_size));
}

void batch_set_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bit_vector, unsigned int *v, size_t v_size) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
  ctx->is_slave_updated[from_id].cpu_rd_ptr()->copy_to_gpu(bit_vector);
	memcpy(ctx->slave_dist_current[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * v_size);
	batch_set_subset_conditional<unsigned int> <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->is_slave_updated[from_id].gpu_rd_ptr(), ctx->slave_dist_current[from_id].gpu_rd_ptr(), ctx->dist_current.gpu_wr_ptr());
	check_cuda_kernel;
}

void batch_add_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bit_vector, unsigned int *v, size_t v_size) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
  ctx->is_master_updated[from_id].cpu_rd_ptr()->copy_to_gpu(bit_vector);
	memcpy(ctx->master_dist_current[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * v_size);
	batch_add_subset_conditional<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->is_master_updated[from_id].gpu_rd_ptr(), ctx->master_dist_current[from_id].gpu_rd_ptr(), ctx->dist_current.gpu_wr_ptr());
	check_cuda_kernel;
}

void batch_min_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bit_vector, unsigned int *v, size_t v_size) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
  ctx->is_master_updated[from_id].cpu_rd_ptr()->copy_to_gpu(bit_vector);
	memcpy(ctx->master_dist_current[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * v_size);
	batch_min_subset_conditional<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->is_master_updated[from_id].gpu_rd_ptr(), ctx->master_dist_current[from_id].gpu_rd_ptr(), ctx->dist_current.gpu_wr_ptr());
	check_cuda_kernel;
}

unsigned int get_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned LID) {
	unsigned int *dist_old = ctx->dist_old.cpu_rd_ptr();
	return dist_old[LID];
}

void set_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *dist_old = ctx->dist_old.cpu_wr_ptr();
	dist_old[LID] = v;
}

void add_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *dist_old = ctx->dist_old.cpu_wr_ptr();
	dist_old[LID] += v;
}

void min_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *dist_old = ctx->dist_old.cpu_wr_ptr();
	if (dist_old[LID] > v)
		dist_old[LID] = v;
}

void batch_get_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_subset<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_dist_old[from_id].gpu_wr_ptr(true), ctx->dist_old.gpu_rd_ptr());
	check_cuda_kernel;
	memcpy(v, ctx->master_dist_old[from_id].cpu_rd_ptr(), sizeof(unsigned int) * ctx->num_master_nodes[from_id]);
}

void batch_get_slave_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_subset<unsigned int> <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->slave_dist_old[from_id].gpu_wr_ptr(true), ctx->dist_old.gpu_rd_ptr());
	check_cuda_kernel;
	memcpy(v, ctx->slave_dist_old[from_id].cpu_rd_ptr(), sizeof(unsigned int) * ctx->num_slave_nodes[from_id]);
}

void batch_get_reset_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_reset_subset<unsigned int> <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->slave_dist_old[from_id].gpu_wr_ptr(true), ctx->dist_old.gpu_rd_ptr(), i);
	check_cuda_kernel;
	memcpy(v, ctx->slave_dist_old[from_id].cpu_rd_ptr(), sizeof(unsigned int) * ctx->num_slave_nodes[from_id]);
}

void batch_set_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->slave_dist_old[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * ctx->num_slave_nodes[from_id]);
	batch_set_subset<unsigned int> <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->slave_dist_old[from_id].gpu_rd_ptr(), ctx->dist_old.gpu_wr_ptr());
	check_cuda_kernel;
}

void batch_add_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_dist_old[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * ctx->num_master_nodes[from_id]);
	batch_add_subset<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_dist_old[from_id].gpu_rd_ptr(), ctx->dist_old.gpu_wr_ptr());
	check_cuda_kernel;
}

void batch_min_node_dist_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_dist_old[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * ctx->num_master_nodes[from_id]);
	batch_min_subset<unsigned int> <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_dist_old[from_id].gpu_rd_ptr(), ctx->dist_old.gpu_wr_ptr());
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

void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g, unsigned num_hosts) {
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
  ctx->is_updated.alloc(1);
  ctx->is_updated.cpu_wr_ptr()->alloc(graph.nnodes);
	ctx->num_master_nodes = (unsigned int *) calloc(num_hosts, sizeof(unsigned int));
	memcpy(ctx->num_master_nodes, g.num_master_nodes, sizeof(unsigned int) * num_hosts);
	ctx->master_nodes = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->master_dist_current = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->master_dist_old = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->is_master_updated = (Shared<DynamicBitset> *) calloc(num_hosts, sizeof(Shared<DynamicBitset>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->num_master_nodes[h] > 0) {
			ctx->master_nodes[h].alloc(ctx->num_master_nodes[h]);
			ctx->is_master_updated[h].alloc(1);
			ctx->is_master_updated[h].cpu_wr_ptr()->alloc(ctx->num_master_nodes[h]);
			memcpy(ctx->master_nodes[h].cpu_wr_ptr(), g.master_nodes[h], sizeof(unsigned int) * ctx->num_master_nodes[h]);
			ctx->master_dist_current[h].alloc(ctx->num_master_nodes[h]);
			ctx->master_dist_old[h].alloc(ctx->num_master_nodes[h]);
		}
	}
	ctx->num_slave_nodes = (unsigned int *) calloc(num_hosts, sizeof(unsigned int));
	memcpy(ctx->num_slave_nodes, g.num_slave_nodes, sizeof(unsigned int) * num_hosts);
	ctx->slave_nodes = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->slave_dist_current = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->slave_dist_old = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->is_slave_updated = (Shared<DynamicBitset> *) calloc(num_hosts, sizeof(Shared<DynamicBitset>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->num_slave_nodes[h] > 0) {
			ctx->slave_nodes[h].alloc(ctx->num_slave_nodes[h]);
			ctx->is_slave_updated[h].alloc(1);
			ctx->is_slave_updated[h].cpu_wr_ptr()->alloc(ctx->num_slave_nodes[h]);
			memcpy(ctx->slave_nodes[h].cpu_wr_ptr(), g.slave_nodes[h], sizeof(unsigned int) * ctx->num_slave_nodes[h]);
			ctx->slave_dist_current[h].alloc(ctx->num_slave_nodes[h]);
			ctx->slave_dist_old[h].alloc(ctx->num_slave_nodes[h]);
		}
	}
	graph.copy_to_gpu(ctx->gg);
	ctx->dist_current.alloc(graph.nnodes);
	ctx->dist_old.alloc(graph.nnodes);
	printf("[%d] load_graph_GPU: %d owned nodes of total %d resident, %d edges\n", ctx->id, ctx->nowned, graph.nnodes, graph.nedges);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context *ctx) {
	ctx->dist_current.zero_gpu();
	ctx->dist_old.zero_gpu();
}

void kernel_sizing(CSRGraphTy & g, dim3 &blocks, dim3 &threads) {
	threads.x = 256;
	threads.y = threads.z = 1;
	blocks.x = 14 * 8;
	blocks.y = blocks.z = 1;
}

