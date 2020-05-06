/*
   csr_graph.h

   Implements a CSR Graph. Part of the GGC source code.
   Interface derived from LonestarGPU.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#ifndef LSG_CSR_GRAPH
#define LSG_CSR_GRAPH

// TODO: original branch has this include; revert it back eventually
//#include "graph_gpu.h"

#include <fstream>
#include "checker.h"
#include "graph_gpu.h"

/*
// Adapted from LSG CSRGraph.h

// TODO: make this template data
typedef unsigned index_type; // should be size_t, but GPU chokes on size_t
typedef int edge_data_type;
typedef int node_data_type;

// very simple implementation
struct CSRGraph {
  unsigned read(char file[], bool read_edge_data = true);
  void copy_to_gpu(struct CSRGraph& copygraph);
  void copy_to_cpu(struct CSRGraph& copygraph);

  CSRGraph();

  unsigned init();
  unsigned allocOnHost(bool no_edge_data = false);
  unsigned allocOnDevice(bool no_edge_data = false);
  void progressPrint(unsigned maxii, unsigned ii);
  unsigned readFromGR(char file[], bool read_edge_data = true);

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

  void init_from_mgraph(int m, int nnz, index_type* h_row_offsets,
                        index_type* h_column_indices,
                        node_data_type* h_labels) {
    nnodes = m;
    nedges = nnz;
    check_cuda(cudaMalloc((void**)&row_start, (m + 1) * sizeof(index_type)));
    check_cuda(cudaMalloc((void**)&edge_dst, nnz * sizeof(index_type)));
    check_cuda(cudaMemcpy(row_start, h_row_offsets,
                          (m + 1) * sizeof(index_type),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(edge_dst, h_column_indices, nnz * sizeof(index_type),
                          cudaMemcpyHostToDevice));
#ifdef ENABLE_LABEL
    check_cuda(cudaMalloc((void**)&node_data, m * sizeof(node_data_type)));
    check_cuda(cudaMemcpy(node_data, h_labels, m * sizeof(node_data_type),
                          cudaMemcpyHostToDevice));
#endif
    // int *h_degrees = (int *)malloc(m * sizeof(int));
    // for (int i = 0; i < m; i++) h_degrees[i] = h_row_offsets[i + 1] -
    // h_row_offsets[i]; check_cuda(cudaMalloc((void **)&d_degrees, m *
    // sizeof(int))); check_cuda(cudaMemcpy(d_degrees, h_degrees, m *
    // sizeof(int), cudaMemcpyHostToDevice));
  }

  inline __device__ __host__ index_type getEdgeDst(unsigned edge) {
    assert(edge < nedges);
    return edge_dst[edge];
  };
  inline __device__ __host__ node_data_type getData(unsigned vid) {
    return node_data[vid];
  }
  inline __device__ __host__ index_type edge_begin(unsigned src) {
    assert(src <= nnodes);
    return row_start[src];
  };
  inline __device__ __host__ index_type edge_end(unsigned src) {
    assert(src <= nnodes);
    return row_start[src + 1];
  };

  index_type nnodes, nedges;
  index_type* row_start; // row_start[node] points into edge_dst, node starts at
                         // 0, row_start[nnodes] = nedges
  index_type* edge_dst;
  edge_data_type* edge_data;
  node_data_type* node_data;
  bool device_graph;
};
>>>>>>> dist-dev
//*/

struct CSRGraphTex : CSRGraph {
  cudaTextureObject_t edge_dst_tx;
  cudaTextureObject_t row_start_tx;
  cudaTextureObject_t node_data_tx;

  void copy_to_gpu(struct CSRGraphTex& copygraph);
  unsigned allocOnDevice(bool no_edge_data = false);

  __device__ __host__ index_type getOutDegree(unsigned src) {
#ifdef __CUDA_ARCH__
    assert(src < nnodes);
    return tex1Dfetch<index_type>(row_start_tx, src + 1) -
           tex1Dfetch<index_type>(row_start_tx, src);
#else
    return CSRGraph::getOutDegree(src);
#endif
  };

  __device__ node_data_type node_data_ro(index_type node) {
    assert(node < nnodes);
    return tex1Dfetch<node_data_type>(node_data_tx, node);
  }

  __device__ __host__ index_type getDestination(unsigned src, unsigned edge) {
#ifdef __CUDA_ARCH__
    assert(src < nnodes);
    assert(edge < getOutDegree(src));

    index_type abs_edge = tex1Dfetch<index_type>(row_start_tx, src + edge);
    assert(abs_edge < nedges);

    return tex1Dfetch<index_type>(edge_dst_tx, abs_edge);
#else
    return CSRGraph::getDestination(src, edge);
#endif
  };

  __device__ __host__ index_type getAbsDestination(unsigned abs_edge) {
#ifdef __CUDA_ARCH__
    assert(abs_edge < nedges);

    return tex1Dfetch<index_type>(edge_dst_tx, abs_edge);
#else
    return CSRGraph::getAbsDestination(abs_edge);
#endif
  };

  __device__ __host__ index_type getFirstEdge(unsigned src) {
#ifdef __CUDA_ARCH__
    assert(src <= nnodes); // <= is okay
    return tex1Dfetch<index_type>(row_start_tx, src);
#else
    return CSRGraph::getFirstEdge(src);
#endif
  };
};

#ifdef CSRG_TEX
typedef CSRGraphTex CSRGraphTy;
#else
typedef CSRGraph CSRGraphTy;
#endif

#endif
