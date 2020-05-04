/*
   csr_graph.h

   Implements a CSR Graph. Part of the GGC source code.
   Interface derived from LonestarGPU.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#ifndef CSR_GRAPH
#define CSR_GRAPH

#include <cassert>
#include <fstream>
#include "checker.h"

// Adapted from LSG CSRGraph.h

// TODO: make this template data
typedef unsigned index_type; // should be size_t, but GPU chokes on size_t
typedef int edge_data_type;
typedef int node_data_type;

// very simple implementation
struct CSRGraph {
  unsigned read(const char file[], bool read_edge_data = true);
  void copy_to_gpu(struct CSRGraph& copygraph);
  void copy_to_cpu(struct CSRGraph& copygraph);

  CSRGraph();

  unsigned init();
  unsigned allocOnHost(bool no_edge_data = false);
  unsigned allocOnDevice(bool no_edge_data = false);
  void progressPrint(unsigned maxii, unsigned ii);
  unsigned readFromGR(const char file[], bool read_edge_data = true);

  unsigned deallocOnHost();
  unsigned deallocOnDevice();
  void dealloc();

  __device__ __host__ bool valid_node(index_type node) {
    return (node < nnodes);
  }

  __device__ __host__ bool valid_edge(index_type edge) {
    return (edge < nedges);
  }

  __device__ __host__ index_type getOutDegree(unsigned src) {
    assert(src < nnodes);
    return row_start[src + 1] - row_start[src];
  };

  __device__ __host__ index_type getDestination(unsigned src, unsigned edge) {
    assert(src < nnodes);
    assert(edge < getOutDegree(src));

    index_type abs_edge = row_start[src] + edge;
    assert(abs_edge < nedges);

    return edge_dst[abs_edge];
  };

  __device__ __host__ index_type getAbsDestination(unsigned abs_edge) {
    assert(abs_edge < nedges);

    return edge_dst[abs_edge];
  };

  __device__ __host__ index_type getFirstEdge(unsigned src) {
    assert(src <= nnodes); // <= is okay
    return row_start[src];
  };

  __device__ __host__ edge_data_type getWeight(unsigned src, unsigned edge) {
    assert(src < nnodes);
    assert(edge < getOutDegree(src));

    index_type abs_edge = row_start[src] + edge;
    assert(abs_edge < nedges);

    return edge_data[abs_edge];
  };

  __device__ __host__ edge_data_type getAbsWeight(unsigned abs_edge) {
    assert(abs_edge < nedges);

    return edge_data[abs_edge];
  };

	void init_from_mgraph(int m, int nnz, index_type *h_row_offsets, index_type *h_column_indices, node_data_type *h_labels) {
		nnodes = m;
		nedges = nnz;
		check_cuda(cudaMalloc((void **)&row_start, (m + 1) * sizeof(index_type)));
		check_cuda(cudaMalloc((void **)&edge_dst, nnz * sizeof(index_type)));
		check_cuda(cudaMemcpy(row_start, h_row_offsets, (m + 1) * sizeof(index_type), cudaMemcpyHostToDevice));
		check_cuda(cudaMemcpy(edge_dst, h_column_indices, nnz * sizeof(index_type), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		check_cuda(cudaMalloc((void **)&node_data, m * sizeof(node_data_type)));
		check_cuda(cudaMemcpy(node_data, h_labels, m * sizeof(node_data_type), cudaMemcpyHostToDevice));
		#endif
	}
	void print_neighbors(index_type vid) {
		printf("Vertex %d neighbors: [ ", vid);
		index_type start = row_start[vid];
		index_type end = row_start[vid+1];
		for (index_type e = start; e != end; e++) {
			index_type dst = edge_dst[e];
			printf("%d ",  dst);
		}
		printf("]\n");
	}
	void add_selfloop() {
		//print_neighbors(nnodes-1);
		//print_neighbors(0);
		index_type *new_edge_dst = new index_type[nnodes+nedges];
		for (index_type i = 0; i < nnodes; i++) {
			index_type start = row_start[i];
			index_type end = row_start[i+1];
			bool selfloop_inserted = false;
			if (start == end) {
				new_edge_dst[start+i] = i;
				continue;
			}
			for (index_type e = start; e != end; e++) {
				index_type dst = edge_dst[e];
				if (!selfloop_inserted) {
					if (i < dst) {
						selfloop_inserted = true;
						new_edge_dst[e+i] = i;
						new_edge_dst[e+i+1] = dst;
					} else if (e+1 == end) {
						selfloop_inserted = true;
						new_edge_dst[e+i+1] = i;
						new_edge_dst[e+i] = dst;
					} else new_edge_dst[e+i] = dst;
				} else new_edge_dst[e+i+1] = dst;
			}
		}
		for (index_type i = 0; i <= nnodes; i++) row_start[i] += i;
		delete edge_dst;
		edge_dst = new_edge_dst;
		nedges += nnodes;
        printf("nnodes = %d, nedges = %d\n", nnodes, nedges);
		//print_neighbors(nnodes-1);
		//print_neighbors(0);
	}

	__device__ __host__ index_type getEdgeDst(unsigned edge) {
		assert(edge < nedges);
		return edge_dst[edge];
	};
	__device__ __host__ node_data_type getData(unsigned vid) {
		return node_data[vid];
	}
	__device__ __host__ index_type edge_begin(unsigned src) {
		assert(src <= nnodes);
		return row_start[src];
	};
	__device__ __host__ index_type edge_end(unsigned src) {
		assert(src <= nnodes);
		return row_start[src+1];
	};
	__device__ __host__ index_type *row_start_ptr() { return row_start; }
	__device__ __host__ const index_type *row_start_ptr() const { return row_start; }
	__device__ __host__ index_type *edge_dst_ptr() { return edge_dst; }
	__device__ __host__ const index_type *edge_dst_ptr() const { return edge_dst; }
	__device__ __host__ node_data_type *node_data_ptr() { return node_data; }
	__device__ __host__ const node_data_type *node_data_ptr() const { return node_data; }
	__device__ __host__ edge_data_type *edge_data_ptr() { return edge_data; }
	__device__ __host__ const edge_data_type *edge_data_ptr() const { return edge_data; }

  size_t size() { return size_t(nnodes); }
  size_t sizeEdges() { return size_t(nedges); }
  void degree_counting() {}
  index_type nnodes, nedges;
  index_type* row_start; // row_start[node] points into edge_dst, node starts at
                         // 0, row_start[nnodes] = nedges
  index_type* edge_dst;
  edge_data_type* edge_data;
  node_data_type* node_data;
  bool device_graph;
};
#endif
