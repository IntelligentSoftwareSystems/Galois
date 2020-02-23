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

#include "graph_gpu.h"

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
