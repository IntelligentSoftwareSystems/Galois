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

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

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

  CUDA_HOSTDEV bool valid_node(index_type node) {
    return (node < nnodes);
  }

  CUDA_HOSTDEV bool valid_edge(index_type edge) {
    return (edge < nedges);
  }

  CUDA_HOSTDEV index_type getOutDegree(unsigned src) {
    assert(src < nnodes);
    return row_start[src + 1] - row_start[src];
  };

  CUDA_HOSTDEV index_type getDestination(unsigned src, unsigned edge) {
    assert(src < nnodes);
    assert(edge < getOutDegree(src));

    index_type abs_edge = row_start[src] + edge;
    assert(abs_edge < nedges);

    return edge_dst[abs_edge];
  };

  CUDA_HOSTDEV index_type getAbsDestination(unsigned abs_edge) {
    assert(abs_edge < nedges);

    return edge_dst[abs_edge];
  };

  CUDA_HOSTDEV index_type getFirstEdge(unsigned src) {
    assert(src <= nnodes); // <= is okay
    return row_start[src];
  };

  CUDA_HOSTDEV edge_data_type getWeight(unsigned src, unsigned edge) {
    assert(src < nnodes);
    assert(edge < getOutDegree(src));

    index_type abs_edge = row_start[src] + edge;
    assert(abs_edge < nedges);

    return edge_data[abs_edge];
  };

  CUDA_HOSTDEV edge_data_type getAbsWeight(unsigned abs_edge) {
    assert(abs_edge < nedges);

    return edge_data[abs_edge];
  };

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

	CUDA_HOSTDEV index_type getEdgeDst(unsigned edge) {
		assert(edge < nedges);
		return edge_dst[edge];
	};
	CUDA_HOSTDEV node_data_type getData(unsigned vid) {
		return node_data[vid];
	}
	CUDA_HOSTDEV index_type edge_begin(unsigned src) {
		assert(src <= nnodes);
		return row_start[src];
	};
	CUDA_HOSTDEV index_type edge_end(unsigned src) {
		assert(src <= nnodes);
		return row_start[src+1];
	};
	CUDA_HOSTDEV index_type *row_start_host_ptr() { return row_start; }
	CUDA_HOSTDEV index_type *row_start_ptr() { return row_start; }
	CUDA_HOSTDEV const index_type *row_start_ptr() const { return row_start; }
	CUDA_HOSTDEV index_type *edge_dst_ptr() { return edge_dst; }
	CUDA_HOSTDEV const index_type *edge_dst_ptr() const { return edge_dst; }
	CUDA_HOSTDEV node_data_type *node_data_ptr() { return node_data; }
	CUDA_HOSTDEV const node_data_type *node_data_ptr() const { return node_data; }
	CUDA_HOSTDEV edge_data_type *edge_data_ptr() { return edge_data; }
	CUDA_HOSTDEV const edge_data_type *edge_data_ptr() const { return edge_data; }
  CUDA_HOSTDEV void fixEndEdge(index_type vid, index_type row_end) { row_start[vid + 1] = row_end; }
  CUDA_HOSTDEV void constructEdge(index_type eid, index_type dst, edge_data_type edata = 0) {
    assert(dst < nnodes);
    assert(eid < nedges);
    edge_dst[eid] = dst;
    if (edge_data) edge_data[eid] = edata;
  }
  void malloc_index_device(index_type n, index_type*& ptr);
  void free_index_device(index_type*& ptr);
  void set_index(index_type pos, index_type value, index_type *ptr);
  void allocateFrom(index_type nv, index_type ne) {
    bool need_realloc = false;
    if (nedges < ne) need_realloc = true;
    nnodes = nv;
    nedges = ne;
    if (max_size < nnodes) max_size = nnodes;
    //printf("allocating memory on gpu nnodes %d nedges %d\n", max_size, nedges);
    if (need_realloc) {
      if (edge_dst) free_index_device(edge_dst);
      malloc_index_device(nedges, edge_dst);
    }
    if (!row_start) malloc_index_device(max_size+1, row_start);
    set_index(0, 0, row_start);
  }
  void set_max_size(index_type max) { assert(max>0); max_size = max; }
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
  index_type max_size; // this is for reallocation; avoid re-malloc
  bool is_allocated; // this is for reallocation
};
#endif
