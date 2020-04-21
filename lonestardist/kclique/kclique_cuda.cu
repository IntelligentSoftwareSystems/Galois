// Copyright (c) 2019, Xuhao Chen
//#include "gg.h"
#include "cutil_subset.h"
#include "kclique_cuda.h"
#include "cuda_launch_config.hpp"
#include "gpu_mining/miner.cuh"
#include "gpu_mining/cutil_subset.h"
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#define USE_SHM
#define MAX_SIZE 5
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

__global__ void extend_alloc(unsigned m, unsigned level, unsigned max_size, CSRGraph graph, EmbeddingList emb_list, unsigned *num_new_emb, AccType *total) {
	unsigned tid = threadIdx.x;
	unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
#ifdef USE_SHM
	__shared__ index_type emb[BLOCK_SIZE][MAX_SIZE];
#else
	index_type emb[MAX_SIZE];
#endif
	AccType local_num = 0;
	if(pos < m) {
#ifdef USE_SHM
		emb_list.get_embedding(level, pos, emb[tid]);
#else
		emb_list.get_embedding(level, pos, emb);
#endif
		index_type vid = emb_list.get_vid(level, pos);
		//if (pos == 0) printout_embedding(level, emb[tid]);
		num_new_emb[pos] = 0;
		index_type row_begin = graph.edge_begin(vid);
		index_type row_end = graph.edge_end(vid);
		for (index_type e = row_begin; e < row_end; e++) {
			index_type dst = graph.getEdgeDst(e);
#ifdef USE_SHM
			if (is_all_connected_dag(dst, emb[tid], level, graph)) {
#else
			if (is_all_connected_dag(dst, emb, level, graph)) {
#endif
				if (level < max_size-2) num_new_emb[pos] ++;
				else local_num += 1;
			}
		}
	}
	AccType block_num = BlockReduce(temp_storage).Sum(local_num);
	if(threadIdx.x == 0) atomicAdd(total, block_num);
}

__global__ void extend_insert(unsigned m, unsigned level, CSRGraph graph, EmbeddingList emb_list, unsigned *indices) {
	unsigned tid = threadIdx.x;
	unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef USE_SHM
	__shared__ index_type emb[BLOCK_SIZE][MAX_SIZE];
#else
	index_type emb[MAX_SIZE];
#endif
	if(pos < m) {
#ifdef USE_SHM
		emb_list.get_embedding(level, pos, emb[tid]);
#else
		emb_list.get_embedding(level, pos, emb);
#endif
		index_type vid = emb_list.get_vid(level, pos);
		index_type start = indices[pos];
		index_type row_begin = graph.edge_begin(vid);
		index_type row_end = graph.edge_end(vid);
		for (index_type e = row_begin; e < row_end; e++) {
			index_type dst = graph.getEdgeDst(e);
#ifdef USE_SHM
			if (is_all_connected_dag(dst, emb[tid], level, graph)) {
#else
			if (is_all_connected_dag(dst, emb, level, graph)) {
#endif
				emb_list.set_idx(level+1, start, pos);
				emb_list.set_vid(level+1, start++, dst);
			}
		}
	}
}

CUDA_Context_Mining cuda_ctx;
void KclInitGPU(MGraph &g, unsigned k) {
	print_device_info(0);
	int m = g.num_vertices();
	int nnz = g.num_edges();
	int nthreads = BLOCK_SIZE;
	int nblocks = DIVIDE_INTO(m, nthreads);
	cuda_ctx.hg = &g;
	cuda_ctx.build_graph_gpu();
	cuda_ctx.emb_list.init(nnz, k);
	init_gpu_dag<<<nblocks, nthreads>>>(m, cuda_ctx.gg, cuda_ctx.emb_list);
	CudaTest("initializing failed");
	check_cuda(cudaDeviceSynchronize());
}

void KclSolverGPU(unsigned k, AccType &total) {
	assert(k <= MAX_SIZE);
	AccType h_total = 0, *d_total;
	AccType zero = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = DIVIDE_INTO(cuda_ctx.emb_list.size(), nthreads);
	check_cuda(cudaMalloc((void **)&d_total, sizeof(AccType)));
	printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);
	unsigned level = 1;
	while (1) {
		unsigned *num_new_emb;
		unsigned num_emb = cuda_ctx.emb_list.size();
		check_cuda(cudaMalloc((void **)&num_new_emb, sizeof(unsigned) * (num_emb+1)));
		check_cuda(cudaMemset(num_new_emb, 0, sizeof(unsigned) * (num_emb+1)));
		nblocks = (num_emb-1)/nthreads+1;
		check_cuda(cudaMemcpy(d_total, &zero, sizeof(AccType), cudaMemcpyHostToDevice));
		extend_alloc<<<nblocks, nthreads>>>(num_emb, level, k, cuda_ctx.gg, cuda_ctx.emb_list, num_new_emb, d_total);
		CudaTest("solving extend_alloc failed");
		if (level == k-2) break; 
		unsigned *indices;
		check_cuda(cudaMalloc((void **)&indices, sizeof(unsigned) * (num_emb+1)));
		thrust::exclusive_scan(thrust::device, num_new_emb, num_new_emb+num_emb+1, indices);
		unsigned new_size;
		check_cuda(cudaMemcpy(&new_size, &indices[num_emb], sizeof(unsigned), cudaMemcpyDeviceToHost));
		assert(new_size < 4294967296); // TODO: currently do not support vector size larger than 2^32
		//std::cout << "number of new embeddings: " << new_size << "\n";
		cuda_ctx.emb_list.add_level(new_size);
		extend_insert<<<nblocks, nthreads>>>(num_emb, level, cuda_ctx.gg, cuda_ctx.emb_list, indices);
		CudaTest("solving extend_insert failed");
		level ++;
	}
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
	total = h_total;
	check_cuda(cudaFree(d_total));
}

