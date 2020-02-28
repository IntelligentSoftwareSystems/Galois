// Copyright (c) 2019, Xuhao Chen
#include <cub/cub.cuh>
#define USE_SIMPLE
#define USE_BASE_TYPES
#include "gpu_mining/miner.cuh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define USE_SHM
typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

__global__ void extend_alloc(size_t begin, size_t end, unsigned level, unsigned max_size, GraphGPU graph, EmbeddingList emb_list, size_t *num_new_emb, AccType *total) {
	unsigned tid = threadIdx.x;
	unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
#ifdef USE_SHM
	__shared__ IndexT emb[BLOCK_SIZE][MAX_SIZE];
#else
	IndexT emb[MAX_SIZE];
#endif
	AccType local_num = 0;
	if(pos < end - begin) {
#ifdef USE_SHM
		emb_list.get_embedding(level, begin + pos, emb[tid]);
#else
		emb_list.get_embedding(level, begin + pos, emb);
#endif
		IndexT vid = emb_list.get_vid(level, begin + pos);
		num_new_emb[pos] = 0;
		IndexT row_begin = graph.edge_begin(vid);
		IndexT row_end = graph.edge_end(vid);
		for (IndexT e = row_begin; e < row_end; e++) {
			IndexT dst = graph.getEdgeDst(e);
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

__global__ void extend_alloc_lb(size_t begin, size_t end, unsigned level, unsigned max_size, GraphGPU graph, EmbeddingList emb_list, unsigned long long *num_new_emb, AccType *total) {
	//expandByCta(m, row_offsets, column_indices, depths, in_queue, out_queue, depth);
	//expandByWarp(m, row_offsets, column_indices, depths, in_queue, out_queue, depth);
	unsigned tid = threadIdx.x;
	unsigned base_id = blockIdx.x * blockDim.x;
	unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage reduce_storage;

	const unsigned SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ unsigned src[SCRATCHSIZE];
	__shared__ IndexT emb[BLOCK_SIZE][MAX_SIZE];
	//IndexT emb[MAX_SIZE];

	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	IndexT row_begin = 0;
	IndexT row_end = 0;

	IndexT vid;
	AccType local_num = 0;
	if (pos < end - begin) {
		//emb_list.get_embedding(level, begin + pos, emb);
		emb_list.get_embedding(level, begin + pos, emb[tid]);
		vid = emb_list.get_vid(level, begin + pos);
		num_new_emb[pos] = 0;
		row_begin = graph.edge_begin(vid);
		row_end = graph.edge_end(vid);
		neighbor_offset = row_begin;
		neighbor_size = row_end - row_begin;
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[scratch_offset + i - done] = neighbor_offset + neighbors_done + i;
			src[scratch_offset + i - done] = tid;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tid < total_edges) {
			int e = gather_offsets[tid];
			IndexT dst = graph.getEdgeDst(e);
			unsigned idx = src[tid];
			if (is_all_connected_dag(dst, emb[idx], level, graph)) {
				if (level < max_size-2) atomicAdd(num_new_emb+base_id+idx, 1);
				else local_num += 1;
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
	AccType block_num = BlockReduce(reduce_storage).Sum(local_num);
	if (tid == 0) atomicAdd(total, block_num);
}


__global__ void extend_insert(size_t begin, size_t end, unsigned level, GraphGPU graph, EmbeddingList emb_list, size_t *indices) {
	unsigned tid = threadIdx.x;
	unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef USE_SHM
	__shared__ IndexT emb[BLOCK_SIZE][MAX_SIZE];
#else
	IndexT emb[MAX_SIZE];
#endif
	if(pos < end - begin) {
#ifdef USE_SHM
		emb_list.get_embedding(level, begin + pos, emb[tid]);
#else
		emb_list.get_embedding(level, begin + pos, emb);
#endif
		IndexT vid = emb_list.get_vid(level, begin + pos);
		IndexT start = indices[pos];
		IndexT row_begin = graph.edge_begin(vid);
		IndexT row_end = graph.edge_end(vid);
		for (IndexT e = row_begin; e < row_end; e++) {
			IndexT dst = graph.getEdgeDst(e);
#ifdef USE_SHM
			if (is_all_connected_dag(dst, emb[tid], level, graph)) {
#else
			if (is_all_connected_dag(dst, emb, level, graph)) {
#endif
				emb_list.set_idx(level+1, start, begin + pos);
				emb_list.set_vid(level+1, start++, dst);
			}
		}
	}
}

void kcl_gpu_solver(std::string filename, unsigned k, AccType &total, size_t N_CHUNK = 1) {
	GraphGPU graph_cpu, graph_gpu;
	graph_cpu.copy_to_gpu(graph_gpu); // copy graph to GPU memory
	graph_cpu.read(filename, false); // read graph into CPU memory
	int nthreads = BLOCK_SIZE;
	int nblocks = DIVIDE_INTO(m, nthreads);
	EmbeddingList emb_list;
	emb_list.init(nnz, k);
	init_gpu_dag<<<nblocks, nthreads>>>(m, gg, emb_list);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	AccType h_total = 0, *d_total;
	AccType zero = 0;
	size_t chunk_length = (nnz - 1) / N_CHUNK + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
	printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	std::cout << "number of single-edge embeddings: " << nnz << "\n";
	for (size_t cid = 0; cid < N_CHUNK; cid ++) {
		size_t chunk_begin = cid * chunk_length;
		size_t chunk_end = std::min((cid+1) * chunk_length, nnz);
		size_t cur_size = chunk_end-chunk_begin;
		std::cout << "Processing the " << cid << " chunk of " << cur_size << " edges\n";

		unsigned level = 1;
		while (1) {
			size_t *num_new_emb;
			size_t num_emb = emb_list.size();
			size_t begin = 0, end = num_emb;
			if (level == 1) { begin = chunk_begin; end = chunk_end; num_emb = end - begin; }
			std::cout << "\t number of embeddings in level " << level << ": " << num_emb << "\n";
			CUDA_SAFE_CALL(cudaMalloc((void **)&num_new_emb, sizeof(size_t) * (num_emb+1)));
			CUDA_SAFE_CALL(cudaMemset(num_new_emb, 0, sizeof(size_t) * (num_emb+1)));
			nblocks = (num_emb-1)/nthreads+1;
			CUDA_SAFE_CALL(cudaMemcpy(d_total, &zero, sizeof(AccType), cudaMemcpyHostToDevice));
			extend_alloc<<<nblocks, nthreads>>>(begin, end, level, k, gg, emb_list, num_new_emb, d_total);
			CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
			total += h_total;
			CudaTest("solving extend alloc failed");
			if (level == k-2) {
				CUDA_SAFE_CALL(cudaFree(num_new_emb));
				break; 
			}
			size_t *indices;
			CUDA_SAFE_CALL(cudaMalloc((void **)&indices, sizeof(size_t) * (num_emb+1)));
			thrust::exclusive_scan(thrust::device, num_new_emb, num_new_emb+num_emb+1, indices);
			CUDA_SAFE_CALL(cudaFree(num_new_emb));
			size_t new_size;
			CUDA_SAFE_CALL(cudaMemcpy(&new_size, &indices[num_emb], sizeof(unsigned), cudaMemcpyDeviceToHost));
			std::cout << "\t number of new embeddings: " << new_size << "\n";
			emb_list.add_level(new_size);
			extend_insert<<<nblocks, nthreads>>>(begin, end, level, gg, emb_list, indices);
			CudaTest("solving extend insert failed");
			CUDA_SAFE_CALL(cudaFree(indices));
			level ++;
		}
		emb_list.reset_level();
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime = %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaFree(d_total));
}

