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

struct CUDA_Context {
	int device;
	int id;
	unsigned int nowned;
	CSRGraphTy hg;
	CSRGraphTy gg;
	unsigned int *num_master_nodes; // per host
	Shared<unsigned int> *master_nodes; // per host
	unsigned int *num_slave_nodes; // per host
	Shared<unsigned int> *slave_nodes; // per host
	Shared<unsigned int> nout;
	Shared<unsigned int> *master_nout; // per host
	Shared<unsigned int> *slave_nout; // per host
	Shared<float> residual;
	Shared<float> *master_residual; // per host
	Shared<float> *slave_residual; // per host
	Shared<float> value;
	Shared<float> *master_value; // per host
	Shared<float> *slave_value; // per host
	Shared<int> p_retval;
	Worklist2 in_wl;
	Worklist2 out_wl;
	struct CUDA_Worklist *shared_wl;
	Any any_retval;
};

unsigned int get_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID) {
	unsigned int *nout = ctx->nout.cpu_rd_ptr();
	return nout[LID];
}

void set_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *nout = ctx->nout.cpu_wr_ptr();
	nout[LID] = v;
}

void add_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *nout = ctx->nout.cpu_wr_ptr();
	nout[LID] += v;
}

void min_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *nout = ctx->nout.cpu_wr_ptr();
	if (nout[LID] > v)
		nout[LID] = v;
}

__global__ void batch_get_node_nout(index_type size, const unsigned int * __restrict__ p_master_nodes, unsigned int * __restrict__ p_master_nout, const unsigned int * __restrict__ p_nout) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_master_nodes[src];
		p_master_nout[src] = p_nout[LID];
	}
}

void batch_get_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_node_nout <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_nout[from_id].gpu_wr_ptr(true), ctx->nout.gpu_rd_ptr());
	check_cuda_kernel;
	memcpy(v, ctx->master_nout[from_id].cpu_rd_ptr(), sizeof(unsigned int) * ctx->num_master_nodes[from_id]);
}

__global__ void batch_get_reset_node_nout(index_type size, const unsigned int * __restrict__ p_slave_nodes, unsigned int * __restrict__ p_slave_nout, unsigned int * __restrict__ p_nout, unsigned int value) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_slave_nodes[src];
		p_slave_nout[src] = p_nout[LID];
		p_nout[LID] = value;
	}
}

void batch_get_reset_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_reset_node_nout <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->slave_nout[from_id].gpu_wr_ptr(true), ctx->nout.gpu_rd_ptr(), i);
	check_cuda_kernel;
	memcpy(v, ctx->slave_nout[from_id].cpu_rd_ptr(), sizeof(unsigned int) * ctx->num_slave_nodes[from_id]);
}

__global__ void batch_set_node_nout(index_type size, const unsigned int * __restrict__ p_slave_nodes, const unsigned int * __restrict__ p_slave_nout, unsigned int * __restrict__ p_nout) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_slave_nodes[src];
		p_nout[LID] = p_slave_nout[src];
	}
}

void batch_set_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->slave_nout[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * ctx->num_slave_nodes[from_id]);
	batch_set_node_nout <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->slave_nout[from_id].gpu_rd_ptr(), ctx->nout.gpu_wr_ptr());
	check_cuda_kernel;
}

__global__ void batch_add_node_nout(index_type size, const unsigned int * __restrict__ p_master_nodes, const unsigned int * __restrict__ p_master_nout, unsigned int * __restrict__ p_nout) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_master_nodes[src];
		p_nout[LID] += p_master_nout[src];
	}
}

void batch_add_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_nout[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * ctx->num_master_nodes[from_id]);
	batch_add_node_nout <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_nout[from_id].gpu_rd_ptr(), ctx->nout.gpu_wr_ptr());
	check_cuda_kernel;
}

__global__ void batch_min_node_nout(index_type size, const unsigned int * __restrict__ p_master_nodes, const unsigned int * __restrict__ p_master_nout, unsigned int * __restrict__ p_nout) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_master_nodes[src];
		p_nout[LID] = (p_nout[LID] > p_master_nout[src]) ? p_master_nout[src] : p_nout[LID];
	}
}

void batch_min_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_nout[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * ctx->num_master_nodes[from_id]);
	batch_min_node_nout <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_nout[from_id].gpu_rd_ptr(), ctx->nout.gpu_wr_ptr());
	check_cuda_kernel;
}

float get_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID) {
	float *residual = ctx->residual.cpu_rd_ptr();
	return residual[LID];
}

void set_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *residual = ctx->residual.cpu_wr_ptr();
	residual[LID] = v;
}

void add_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *residual = ctx->residual.cpu_wr_ptr();
	residual[LID] += v;
}

void min_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *residual = ctx->residual.cpu_wr_ptr();
	if (residual[LID] > v)
		residual[LID] = v;
}

__global__ void batch_get_node_residual(index_type size, const unsigned int * __restrict__ p_master_nodes, float * __restrict__ p_master_residual, const float * __restrict__ p_residual) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_master_nodes[src];
		p_master_residual[src] = p_residual[LID];
	}
}

void batch_get_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_node_residual <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_residual[from_id].gpu_wr_ptr(true), ctx->residual.gpu_rd_ptr());
	check_cuda_kernel;
	memcpy(v, ctx->master_residual[from_id].cpu_rd_ptr(), sizeof(float) * ctx->num_master_nodes[from_id]);
}

__global__ void batch_get_reset_node_residual(index_type size, const unsigned int * __restrict__ p_slave_nodes, float * __restrict__ p_slave_residual, float * __restrict__ p_residual, float value) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_slave_nodes[src];
		p_slave_residual[src] = p_residual[LID];
		p_residual[LID] = value;
	}
}

void batch_get_reset_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_reset_node_residual <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->slave_residual[from_id].gpu_wr_ptr(true), ctx->residual.gpu_rd_ptr(), i);
	check_cuda_kernel;
	memcpy(v, ctx->slave_residual[from_id].cpu_rd_ptr(), sizeof(float) * ctx->num_slave_nodes[from_id]);
}

__global__ void batch_set_node_residual(index_type size, const unsigned int * __restrict__ p_slave_nodes, const float * __restrict__ p_slave_residual, float * __restrict__ p_residual) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_slave_nodes[src];
		p_residual[LID] = p_slave_residual[src];
	}
}

void batch_set_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->slave_residual[from_id].cpu_wr_ptr(true), v, sizeof(float) * ctx->num_slave_nodes[from_id]);
	batch_set_node_residual <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->slave_residual[from_id].gpu_rd_ptr(), ctx->residual.gpu_wr_ptr());
	check_cuda_kernel;
}

__global__ void batch_add_node_residual(index_type size, const unsigned int * __restrict__ p_master_nodes, const float * __restrict__ p_master_residual, float * __restrict__ p_residual) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_master_nodes[src];
		p_residual[LID] += p_master_residual[src];
	}
}

void batch_add_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_residual[from_id].cpu_wr_ptr(true), v, sizeof(float) * ctx->num_master_nodes[from_id]);
	batch_add_node_residual <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_residual[from_id].gpu_rd_ptr(), ctx->residual.gpu_wr_ptr());
	check_cuda_kernel;
}

__global__ void batch_min_node_residual(index_type size, const unsigned int * __restrict__ p_master_nodes, const float * __restrict__ p_master_residual, float * __restrict__ p_residual) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_master_nodes[src];
		p_residual[LID] = (p_residual[LID] > p_master_residual[src]) ? p_master_residual[src] : p_residual[LID];
	}
}

void batch_min_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_residual[from_id].cpu_wr_ptr(true), v, sizeof(float) * ctx->num_master_nodes[from_id]);
	batch_min_node_residual <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_residual[from_id].gpu_rd_ptr(), ctx->residual.gpu_wr_ptr());
	check_cuda_kernel;
}

float get_node_value_cuda(struct CUDA_Context *ctx, unsigned LID) {
	float *value = ctx->value.cpu_rd_ptr();
	return value[LID];
}

void set_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *value = ctx->value.cpu_wr_ptr();
	value[LID] = v;
}

void add_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *value = ctx->value.cpu_wr_ptr();
	value[LID] += v;
}

void min_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *value = ctx->value.cpu_wr_ptr();
	if (value[LID] > v)
		value[LID] = v;
}

__global__ void batch_get_node_value(index_type size, const unsigned int * __restrict__ p_master_nodes, float * __restrict__ p_master_value, const float * __restrict__ p_value) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_master_nodes[src];
		p_master_value[src] = p_value[LID];
	}
}

void batch_get_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_node_value <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_value[from_id].gpu_wr_ptr(true), ctx->value.gpu_rd_ptr());
	check_cuda_kernel;
	memcpy(v, ctx->master_value[from_id].cpu_rd_ptr(), sizeof(float) * ctx->num_master_nodes[from_id]);
}

__global__ void batch_get_reset_node_value(index_type size, const unsigned int * __restrict__ p_slave_nodes, float * __restrict__ p_slave_value, float * __restrict__ p_value, float value) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_slave_nodes[src];
		p_slave_value[src] = p_value[LID];
		p_value[LID] = value;
	}
}

void batch_get_reset_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	batch_get_reset_node_value <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->slave_value[from_id].gpu_wr_ptr(true), ctx->value.gpu_rd_ptr(), i);
	check_cuda_kernel;
	memcpy(v, ctx->slave_value[from_id].cpu_rd_ptr(), sizeof(float) * ctx->num_slave_nodes[from_id]);
}

__global__ void batch_set_node_value(index_type size, const unsigned int * __restrict__ p_slave_nodes, const float * __restrict__ p_slave_value, float * __restrict__ p_value) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_slave_nodes[src];
		p_value[LID] = p_slave_value[src];
	}
}

void batch_set_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->slave_value[from_id].cpu_wr_ptr(true), v, sizeof(float) * ctx->num_slave_nodes[from_id]);
	batch_set_node_value <<<blocks, threads>>>(ctx->num_slave_nodes[from_id], ctx->slave_nodes[from_id].gpu_rd_ptr(), ctx->slave_value[from_id].gpu_rd_ptr(), ctx->value.gpu_wr_ptr());
	check_cuda_kernel;
}

__global__ void batch_add_node_value(index_type size, const unsigned int * __restrict__ p_master_nodes, const float * __restrict__ p_master_value, float * __restrict__ p_value) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_master_nodes[src];
		p_value[LID] += p_master_value[src];
	}
}

void batch_add_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_value[from_id].cpu_wr_ptr(true), v, sizeof(float) * ctx->num_master_nodes[from_id]);
	batch_add_node_value <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_value[from_id].gpu_rd_ptr(), ctx->value.gpu_wr_ptr());
	check_cuda_kernel;
}

__global__ void batch_min_node_value(index_type size, const unsigned int * __restrict__ p_master_nodes, const float * __restrict__ p_master_value, float * __restrict__ p_value) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
		unsigned LID = p_master_nodes[src];
		p_value[LID] = (p_value[LID] > p_master_value[src]) ? p_master_value[src] : p_value[LID];
	}
}

void batch_min_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(ctx->gg, blocks, threads);
	memcpy(ctx->master_value[from_id].cpu_wr_ptr(true), v, sizeof(float) * ctx->num_master_nodes[from_id]);
	batch_min_node_value <<<blocks, threads>>>(ctx->num_master_nodes[from_id], ctx->master_nodes[from_id].gpu_rd_ptr(), ctx->master_value[from_id].gpu_rd_ptr(), ctx->value.gpu_wr_ptr());
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
	ctx->master_nout = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->master_residual = (Shared<float> *) calloc(num_hosts, sizeof(Shared<float>));
	ctx->master_value = (Shared<float> *) calloc(num_hosts, sizeof(Shared<float>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->num_master_nodes[h] > 0) {
			ctx->master_nodes[h].alloc(ctx->num_master_nodes[h]);
			memcpy(ctx->master_nodes[h].cpu_wr_ptr(), g.master_nodes[h], sizeof(unsigned int) * ctx->num_master_nodes[h]);
			ctx->master_nout[h].alloc(ctx->num_master_nodes[h]);
			ctx->master_residual[h].alloc(ctx->num_master_nodes[h]);
			ctx->master_value[h].alloc(ctx->num_master_nodes[h]);
		}
	}
	ctx->num_slave_nodes = (unsigned int *) calloc(num_hosts, sizeof(unsigned int));
	memcpy(ctx->num_slave_nodes, g.num_slave_nodes, sizeof(unsigned int) * num_hosts);
	ctx->slave_nodes = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->slave_nout = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->slave_residual = (Shared<float> *) calloc(num_hosts, sizeof(Shared<float>));
	ctx->slave_value = (Shared<float> *) calloc(num_hosts, sizeof(Shared<float>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->num_slave_nodes[h] > 0) {
			ctx->slave_nodes[h].alloc(ctx->num_slave_nodes[h]);
			memcpy(ctx->slave_nodes[h].cpu_wr_ptr(), g.slave_nodes[h], sizeof(unsigned int) * ctx->num_slave_nodes[h]);
			ctx->slave_nout[h].alloc(ctx->num_slave_nodes[h]);
			ctx->slave_residual[h].alloc(ctx->num_slave_nodes[h]);
			ctx->slave_value[h].alloc(ctx->num_slave_nodes[h]);
		}
	}
	graph.copy_to_gpu(ctx->gg);
	ctx->nout.alloc(graph.nnodes);
	ctx->residual.alloc(graph.nnodes);
	ctx->value.alloc(graph.nnodes);
  wl->max_size = wl_dup_factor*graph.nnodes*num_hosts/2;
	ctx->in_wl = Worklist2((size_t)wl->max_size);
	ctx->out_wl = Worklist2((size_t)wl->max_size);
	wl->num_in_items = -1;
	wl->num_out_items = -1;
	wl->in_items = ctx->in_wl.wl;
	wl->out_items = ctx->out_wl.wl;
	ctx->shared_wl = wl;
	ctx->p_retval = Shared<int>(1);
	printf("[%d] load_graph_GPU: %d owned nodes of total %d resident, %d edges\n", ctx->id, ctx->nowned, graph.nnodes, graph.nedges);
	printf("[%d] load_graph_GPU: worklist size %d\n", ctx->id, (size_t)wl_dup_factor*graph.nnodes);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context *ctx) {
	ctx->nout.zero_gpu();
	ctx->residual.zero_gpu();
	ctx->value.zero_gpu();
}

void kernel_sizing(CSRGraphTy & g, dim3 &blocks, dim3 &threads) {
	threads.x = 256;
	threads.y = threads.z = 1;
	blocks.x = 14 * 8;
	blocks.y = blocks.z = 1;
}

