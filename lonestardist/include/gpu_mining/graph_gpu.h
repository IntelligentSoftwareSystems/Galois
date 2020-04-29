#pragma once
#include <cuda.h>
#include "common.h"
#include "graph.h"

class GraphGPU {
protected:
	IndexT *d_row_offsets;
	IndexT *d_column_indices;
	ValueT *d_labels;
	int *d_degrees;
	int num_vertices;
	int num_edges;
public:
	GraphGPU() {}
	//~GraphGPU() {}
	void clean() {
		CUDA_SAFE_CALL(cudaFree(d_row_offsets));
		CUDA_SAFE_CALL(cudaFree(d_column_indices));
		CUDA_SAFE_CALL(cudaFree(d_degrees));
	}
	void init(Graph *hg) {
		int m = hg->num_vertices();
		int nnz = hg->num_edges();
		num_vertices = m;
		num_edges = nnz;
		IndexT *h_row_offsets = hg->out_rowptr();
		IndexT *h_column_indices = hg->out_colidx();
		int *h_degrees = (int *)malloc(m * sizeof(int));
		for (int i = 0; i < m; i++) h_degrees[i] = h_row_offsets[i + 1] - h_row_offsets[i];
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		ValueT *h_labels = (ValueT *)malloc(m * sizeof(ValueT));
		for (int i = 0; i < m; i++) h_labels[i] = hg->getData(i);
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_labels, m * sizeof(ValueT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_labels, h_labels, m * sizeof(ValueT), cudaMemcpyHostToDevice));
		#endif
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(d_degrees, h_degrees, m * sizeof(int), cudaMemcpyHostToDevice));
	}
	__device__ __host__ bool valid_node(IndexT node) { return (node < num_vertices); }
	__device__ __host__ bool valid_edge(IndexT edge) { return (edge < num_edges); }
	__device__ __host__ IndexT getOutDegree(unsigned src) {
		assert(src < num_vertices);
		return d_row_offsets[src+1] - d_row_offsets[src];
	};
	__device__ __host__ IndexT getDestination(unsigned src, unsigned edge) {
		assert(src < num_vertices);
		assert(edge < getOutDegree(src));
		IndexT abs_edge = d_row_offsets[src] + edge;
		assert(abs_edge < num_edges);
		return d_column_indices[abs_edge];
	};
	__device__ __host__ IndexT getAbsDestination(unsigned abs_edge) {
		assert(abs_edge < num_edges);
		return d_column_indices[abs_edge];
	};
	inline __device__ __host__ IndexT getEdgeDst(unsigned edge) {
		assert(edge < num_edges);
		return d_column_indices[edge];
	};
	inline __device__ __host__ ValueT getData(unsigned vid) {
		return d_labels[vid];
	}
	inline __device__ __host__ IndexT edge_begin(unsigned src) {
		assert(src <= num_vertices);
		return d_row_offsets[src];
	};
	inline __device__ __host__ IndexT edge_end(unsigned src) {
		assert(src <= num_vertices);
		return d_row_offsets[src+1];
	};
};
