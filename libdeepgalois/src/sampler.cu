#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "deepgalois/sampler.h"

namespace deepgalois {

// set the masks of vertices in a given vertex set
// n is the size of the vertex set
__global__ void set_masks(index_t n, index_t* vertices, mask_t* masks) {
  CUDA_KERNEL_LOOP(i, n) { masks[vertices[i]] = 1; }
}

// compute the degrees of a masked graph
// n is the size of the original graph 
__global__ void get_masked_degrees(index_t n, mask_t *masks, GraphGPU g, index_t* degrees) {
  CUDA_KERNEL_LOOP(src, n) {
    if (masks[src] == 1) {
      for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
        auto dst = g.getEdgeDst(e);
        if (masks[dst] == 1) degrees[src] ++;
      }
    }
  }
}

// Given a graph, remove any edge which has end-point masked, and generate the subgraph
// n is the size of the original graph and the subgraph
// offset was computed by using prefix-sum of the masked degrees
__global__ void generate_masked_graph_kernel(index_t n, const mask_t *masks, const index_t* offsets, GraphGPU g, GraphGPU subg) {
  CUDA_KERNEL_LOOP(src, n) {
    subg.fixEndEdge(src, offsets[src+1]);
    if (masks[src] == 1) {
      auto idx = offsets[src];
      for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
        auto dst = g.getEdgeDst(e);
        if (masks[dst] == 1) subg.constructEdge(idx++, dst);
      }
    }
  }
}

// compute the degrees of the subgraph induced by the vertex set
// n is the size of the vertex set
// new_ids array maps vertex ID in the original graph to the vertex ID in the subgraph
__global__ void get_new_degrees(index_t n, index_t* vertices, index_t* new_ids, GraphGPU g, index_t* degrees) {
  CUDA_KERNEL_LOOP(i, n) {
    auto v = vertices[i];
    degrees[new_ids[v]] = g.getOutDegree(v);
  }
}

// Given a masked graph, remove the masked vertices, reindex the rest vertices, and generate the subgraph
// offset was computed by using prefix-sum of the new degrees
// n is the size of the old_ids and the sbugraph
__global__ void generate_graph_kernel(index_t n, const index_t* offsets, const index_t* old_ids, const index_t* new_ids, GraphGPU g, GraphGPU subg) {
  CUDA_KERNEL_LOOP(i, n) {
    subg.fixEndEdge(i, offsets[i+1]);
    index_t j = 0;
    auto src  = old_ids[i];
    for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
      auto dst = new_ids[g.getEdgeDst(e)];
      assert(dst < n);
      subg.constructEdge(offsets[i] + j, dst);
      j++;
    }
  }
}

void Sampler::update_masks(size_t n, index_t* vertices, mask_t *masks) {
  set_masks<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, vertices, masks);
}

void Sampler::indexing(size_t n, index_t* vertices, index_t *new_indices) {
  index_t vid = 0;
  for (index_t i = 0; i < n; i++) {
    auto v = vertices[i];
    new_indices[v] = vid ++;
  }
}

inline VertexList Sampler::reindexing_vertices(size_t n, VertexSet vertex_set) {
  VertexList new_ids(n, 0);
  int vid = 0;
  for (auto v : vertex_set) {
    new_ids[v] = vid++; // reindex
  }
  return new_ids;
}

void Sampler::generate_masked_graph(index_t n, mask_t* masks, GraphGPU *g, GraphGPU *subg) {
  index_t *degrees, *offsets;
  CUDA_CHECK(cudaMalloc((void**)&degrees, sizeof(index_t)*n);
  get_masked_degrees<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, masks, g, degrees);
  CUDA_CHECK(cudaFree(degrees));
  CUDA_CHECK(cudaMalloc((void**)&offsets, sizeof(index_t)*(n+1));
  thrust::exclusive_scan(thrust::device, degrees, degrees+n, offsets);
  index_t ne;
  CUDA_CHECK(cudaMemcpy(&ne, offsets+n, sizeof(index_t), cudaMemcpyDeviceToHost));
  subg.allocateFrom(n, ne); // TODO: avoid reallocation
  generate_masked_graph_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, masks, offsets, g, subg);
  CUDA_CHECK(cudaFree(pffsets));
}

// use a random walk to select vertex subset
void Sampler::select_vertices(size_t n, int m, VertexSet &st) {
}

// n: size of the original graph
// nv: size of the subgraph; i.e. size of vertex_set
// masks, graph g and subgraph sub are on the device (GPU)
void Sampler::generate_subgraph(index_t nv, VertexSet vertex_set, mask_t* masks, GraphGPU *g, GraphGPU *sub) {
  // convert the vertex_set to a vertex_list and copy it to the device
  VertexList vertex_list(vertex_set.begin(), vertex_set.end());
  index_t *d_vertex_list;
  cudaMalloc((void **) &d_vertex_list, nv*sizeof(index_t));
  CUDA_CHECK(cudaMemcpy(d_vertex_list, &vertex_list[0], nv*sizeof(index_t), cudaMemcpyHostToDevice));

  index_t n = graph->size();
  update_masks(n, d_vertex_list, masks); // set masks for vertices in the vertex_set
  GraphGPU masked_sg; // size is the same as original graph, but masked dst removed
  generate_masked_graph(n, masks, g, &masked_sg); // remove edges whose destination is not masked

  // re-index the subgraph
  index_t *d_new_ids; // Given an old vertex ID ∈ [0, n), returns a new vertex ID ∈ [0, nv)
  cudaMalloc((void **) &d_new_ids, n*sizeof(index_t));
  auto new_ids = reindexing_vertices(nv, vertex_set);
  CUDA_CHECK(cudaMemcpy(d_new_ids, &new_ids[0], n*sizeof(index_t), cudaMemcpyHostToDevice));

  // generate the offsets for the re-indexed subgraph
  index_t *degrees, *offsets;
  CUDA_CHECK(cudaMalloc((void**)&degrees, sizeof(index_t)*nv);
  get_new_degrees<<<CUDA_GET_BLOCKS(nv), CUDA_NUM_THREADS>>>(nv, d_vertex_list, d_new_ids, masked_sg, degrees);
  CUDA_CHECK(cudaFree(degrees));
  CUDA_CHECK(cudaMalloc((void**)&offsets, sizeof(index_t)*(nv+1));
  thrust::exclusive_scan(thrust::device, degrees, degrees+nv, offsets);
  index_t ne;
  CUDA_CHECK(cudaMemcpy(&ne, offsets+nv, sizeof(index_t), cudaMemcpyDeviceToHost));

  // allocate memory for the subgraph
  sub.allocateFrom(nv, ne); // avoid reallocation
  // generate the subgraph
  generate_graph_kernel<<<CUDA_GET_BLOCKS(nv), CUDA_NUM_THREADS>>>(nv, offsets, d_vertex_list, d_new_ids, masked_sg, sub);
}

}
