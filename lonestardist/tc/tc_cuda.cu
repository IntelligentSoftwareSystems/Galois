#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
#include "moderngpu/kernel_reduce.hxx"
#include "tc_cuda.cuh"
#include "moderngpu/kernel_segsort.hxx"
#include <cuda_profiler_api.h>
using namespace mgpu;
standard_context_t context;
#define WARP_SIZE 32

inline __device__ unsigned long intersect(CSRGraph graph, index_type u, index_type v) {
	index_type u_start = graph.getFirstEdge(u);
	index_type u_end = u_start + graph.getOutDegree(u);
	index_type v_start = graph.getFirstEdge(v);
	index_type v_end = v_start + graph.getOutDegree(v);
	unsigned long count = 0;
	index_type u_it = u_start;
	index_type v_it = v_start;
	index_type a;
	index_type b;
	while (u_it < u_end && v_it < v_end) {
		a = graph.getAbsDestination(u_it);
		b = graph.getAbsDestination(v_it);
		int d = a - b;
		if (d <= 0) u_it++;
		if (d >= 0) v_it++;
		if (d == 0) count++;
	}
	return count;
}

__global__ void base(CSRGraph graph, unsigned begin, unsigned end, HGAccumulator<unsigned long> num_local_triangles) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long local_total = 0;
	__shared__ cub::BlockReduce<unsigned long, TB_SIZE>::TempStorage num_local_triangles_ts;
	num_local_triangles.thread_entry();
	for (index_type src = begin + tid; src < end; src += TOTAL_THREADS_1D) {
		index_type row_begin = graph.getFirstEdge(src);
		index_type row_end = row_begin + graph.getOutDegree(src); 
		for (index_type offset = row_begin; offset < row_end; ++ offset) {
			index_type dst = graph.getAbsDestination(offset);
			local_total = intersect(graph, dst, src);
			if (local_total) num_local_triangles.reduce(local_total);
		}
	}
	num_local_triangles.thread_exit<cub::BlockReduce<unsigned long, TB_SIZE> >(num_local_triangles_ts);
}

inline __device__ bool serial_search(CSRGraph graph, unsigned key, index_type begin, index_type end) {
	for (index_type offset = begin; offset < end; ++ offset) {
		index_type d = graph.getAbsDestination(offset);
		if (d == key) return true;
		if (d > key) return false;
	}
	return false;
}

inline __device__ bool binary_search(CSRGraph graph, index_type key, index_type begin, index_type end) {
	assert(begin < end);
	int l = begin;
	int r = end-1;
	while (r >= l) { 
		//assert(l<graph.nedges && r<graph.nedges);
		int mid = l + (r - l) / 2; 
		if (mid >= graph.nedges) printf("mid=%u, l=%u, r=%u, begin=%u, end=%u, key=%u\n", mid, l, r, begin, end, key);
		assert(mid < graph.nedges);
		index_type value = graph.getAbsDestination(mid);
		if (value == key) return true;
		if (value < key) l = mid + 1;
		else r = mid - 1;
	}
	return false;
}

__global__ void warp(CSRGraph graph, unsigned begin, unsigned end, HGAccumulator<unsigned long> num_local_triangles) {
	unsigned thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	unsigned warp_id     = thread_id   / WARP_SIZE;                // global warp index
	unsigned num_warps   = (TB_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	__shared__ cub::BlockReduce<unsigned long, TB_SIZE>::TempStorage num_local_triangles_ts;
	num_local_triangles.thread_entry();
	// each warp takes one vertex
	for (index_type src = begin + warp_id; src < end; src += num_warps) {
		index_type row_begin = graph.getFirstEdge(src);
		index_type src_size = graph.getOutDegree(src);
		index_type row_end = row_begin + src_size;
		// take one edge
		for (index_type offset = row_begin; offset < row_end; offset ++) {
			index_type dst = graph.getAbsDestination(offset);
			assert(src != dst);
			index_type dst_size = graph.getOutDegree(dst);
			index_type lookup = src;
			index_type search = dst;
			if (src_size > dst_size) {
				lookup = dst;
				search = src;
			}
			index_type lookup_begin = graph.getFirstEdge(lookup);
			index_type lookup_size = graph.getOutDegree(lookup);
			index_type search_size = graph.getOutDegree(search);
			if (lookup_size > 0 && search_size > 0) {
				for (index_type i = thread_lane; i < lookup_size; i += WARP_SIZE) {
					index_type index = lookup_begin + i;
					index_type key = graph.getAbsDestination(index);
					index_type search_begin = graph.getFirstEdge(search);
					if (binary_search(graph, key, search_begin, search_begin+search_size))
					//if (serial_search(graph, key, search_begin, search_begin+search_size))
						num_local_triangles.reduce(1);
				}
			}
		}
	}
	num_local_triangles.thread_exit<cub::BlockReduce<unsigned long, TB_SIZE> >(num_local_triangles_ts);
}

void sort_cuda(struct CUDA_Context* ctx) {
        segmented_sort(ctx->gg.edge_dst, ctx->gg.nedges, (const int *) ctx->gg.row_start + 1, ctx->gg.nnodes - 1, less_t<int>(), context);
}

void TC_cuda(unsigned __begin, unsigned __end, unsigned long & num_local_triangles, struct CUDA_Context* ctx) {
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);
	HGAccumulator<unsigned long> _num_local_triangles;
	Shared<unsigned long> num_local_trianglesval  = Shared<unsigned long>(1);
	*(num_local_trianglesval.cpu_wr_ptr()) = 0;
	_num_local_triangles.rv = num_local_trianglesval.gpu_wr_ptr();
	//mgc = mgpu::CreateCudaDevice(ctx->device);
	//mgpu::SegSortKeysFromIndices(ctx->gg.edge_dst, ctx->gg.nedges, (const int *) ctx->gg.row_start + 1, ctx->gg.nnodes - 1, *mgc);
	//base<<<blocks, TB_SIZE>>>(ctx->gg, __begin, __end, _num_local_triangles);
	warp<<<blocks, TB_SIZE>>>(ctx->gg, __begin, __end, _num_local_triangles);
	cudaDeviceSynchronize();
	check_cuda_kernel;
	num_local_triangles = *(num_local_trianglesval.cpu_rd_ptr());
	//dump_memory_info("end", ctx->id);
	cudaProfilerStop();
	//num_local_triangles = (unsigned)h_total;
}

void TC_masterNodes_cuda(unsigned long& num_local_triangles, struct CUDA_Context* ctx) {
	TC_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, num_local_triangles, ctx);
}
