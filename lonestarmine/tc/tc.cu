// Copyright (c) 2019, Xuhao Chen
#include "tc.h"
#include "timer.h"
#include "cutils.h"
#define USE_SIMPLE
#define USE_BASE_TYPES
#include <cub/cub.cuh>
#include "gpu_mining/miner.cuh"
typedef cub::BlockReduce<unsigned long long, BLOCK_SIZE> BlockReduce;

__global__ void warp_edge(int m, CSRGraph graph, EmbeddingList emb_list, unsigned long long *total) {
	__shared__ typename BlockReduce::TempStorage temp_storage;
	unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
	//unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	unsigned num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;      // total number of active warps

	unsigned long long local_num = 0;
	for (IndexT tid = warp_id; tid < m; tid += num_warps) {
		IndexT src = emb_list.get_idx(1, tid);
		IndexT dst = emb_list.get_vid(1, tid);
		assert(src != dst);
		IndexT src_size = graph.getOutDegree(src);
		IndexT dst_size = graph.getOutDegree(dst);
		IndexT lookup = src;
		IndexT search = dst;
		if (src_size > dst_size) {
			lookup = dst;
			search = src;
		}
		IndexT lookup_begin = graph.edge_begin(lookup);
		IndexT lookup_size = graph.getOutDegree(lookup);
		IndexT search_size = graph.getOutDegree(search);
		if (lookup_size > 0 && search_size > 0) {
			for (IndexT i = thread_lane; i < lookup_size; i += WARP_SIZE) {
				IndexT index = lookup_begin + i;
				IndexT key = graph.getEdgeDst(index);
				IndexT search_begin = graph.edge_begin(search);
				if (binary_search(graph, key, search_begin, search_begin+search_size))
					local_num += 1;
			}
		}
	}
	unsigned long long block_num = BlockReduce(temp_storage).Sum(local_num);
	if(threadIdx.x == 0) atomicAdd(total, block_num);
}

__global__ void warp(int m, IndexT *row_offsets, IndexT *column_indices, int *degrees, unsigned long long *total) {
	__shared__ typename BlockReduce::TempStorage temp_storage;
	unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
	//unsigned warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	unsigned num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;      // total number of active warps

	unsigned long long local_num = 0;
	// each warp takes one vertex
	for (IndexT src = warp_id; src < m; src += num_warps) {
		IndexT row_begin = row_offsets[src];
		IndexT row_end = row_offsets[src+1];
		IndexT src_size = degrees[src];
		// take one edge
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT dst = column_indices[offset];
			assert(src != dst);
			IndexT dst_size = degrees[dst];
			IndexT lookup = src;
			IndexT search = dst;
			if (src_size > dst_size) {
				lookup = dst;
				search = src;
			}
			IndexT lookup_begin = row_offsets[lookup];
			IndexT lookup_size = degrees[lookup];
			IndexT search_size = degrees[search];
			if (lookup_size > 0 && search_size > 0) {
				for (IndexT i = thread_lane; i < lookup_size; i += WARP_SIZE) {
					IndexT index = lookup_begin + i;
					IndexT key = column_indices[index];
					IndexT search_begin = row_offsets[search];
					if (binary_search(column_indices, key, search_begin, search_begin+search_size))
						local_num += 1;
				}
			}
		}
	}
	unsigned long long block_num = BlockReduce(temp_storage).Sum(local_num);
	if(threadIdx.x == 0) atomicAdd(total, block_num);
}


void tc_gpu_solver(std::string fname, AccType &total, size_t N_CHUNK) {
	CSRGraph graph_cpu, graph_gpu;
	graph_cpu.read(fname, false, true); // read graph into CPU memory, use DAG
	int m = graph_cpu.get_nnodes();
	int nnz = graph_cpu.get_nedges();
	graph_cpu.copy_to_gpu(graph_gpu); // copy graph to GPU memory
	EmbeddingList emb_list;
	emb_list.init(nnz);

	int nthreads = BLOCK_SIZE;
	int nblocks = DIVIDE_INTO(m, WARPS_PER_BLOCK);
	init_gpu_dag<<<nblocks, nthreads>>>(m, graph_gpu, emb_list);

	unsigned long long h_total = 0, *d_total;
	unsigned long long  zero = 0;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(unsigned long long)));
	CUDA_SAFE_CALL(cudaMemcpy(d_total, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice));
	printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	//warp<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_degrees, d_total);
	warp_edge<<<nblocks, nthreads>>>(nnz, graph_gpu, emb_list, d_total);
	CudaTest("solving failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime = %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	total = h_total;
	CUDA_SAFE_CALL(cudaFree(d_total));
}

