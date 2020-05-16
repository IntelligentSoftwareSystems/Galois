#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "deepgalois/cutils.h"
#include "deepgalois/Sampler.h"

namespace deepgalois {

__global__ void clear_masks(index_t n, mask_t* masks) {
  CUDA_KERNEL_LOOP(i, n) { masks[i] = 0; }
}

// set the masks of vertices in a given vertex set
// n is the size of the vertex set
__global__ void set_masks(index_t n, index_t* vertices, mask_t* masks) {
  CUDA_KERNEL_LOOP(i, n) { masks[vertices[i]] = 1; }
}

// compute the degrees of a masked graph
// n is the size of the original graph
__global__ void get_masked_degrees(index_t n, mask_t* masks, GraphGPU g,
                                   index_t* degrees) {
  CUDA_KERNEL_LOOP(src, n) {
    if (src < 10) printf("masks[%d] = %d\n", src, masks[src]);
    degrees[src] = 0;
    if (masks[src] == 1) {
      for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
        auto dst = g.getEdgeDst(e);
        if (masks[dst] == 1)
          degrees[src]++;
      }
    }
    if (src < 10) printf("degrees[%d] = %d\n", src, degrees[src]);
  }
}

// Given a graph, remove any edge which has end-point masked, and generate the
// subgraph n is the size of the original graph and the subgraph offset was
// computed by using prefix-sum of the masked degrees
__global__ void generate_masked_graph_kernel(index_t n, const mask_t* masks,
                                             const index_t* offsets, GraphGPU g,
                                             GraphGPU subg) {
  CUDA_KERNEL_LOOP(src, n) {
    subg.fixEndEdge(src, offsets[src + 1]);
    if (masks[src] == 1) {
      auto idx = offsets[src];
      for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
        auto dst = g.getEdgeDst(e);
        if (masks[dst] == 1)
          subg.constructEdge(idx++, dst);
      }
    }
  }
}

// compute the degrees of the subgraph induced by the vertex set
// n is the size of the vertex set
// new_ids array maps vertex ID in the original graph to the vertex ID in the
// subgraph
__global__ void get_new_degrees(index_t n, index_t* vertices, index_t* new_ids,
                                GraphGPU g, index_t* degrees) {
  CUDA_KERNEL_LOOP(i, n) {
    auto v              = vertices[i];
    degrees[new_ids[v]] = g.getOutDegree(v);
  }
}

// Given a masked graph, remove the masked vertices, reindex the rest vertices,
// and generate the subgraph offset was computed by using prefix-sum of the new
// degrees n is the size of the old_ids and the sbugraph
__global__ void generate_graph_kernel(index_t n, const index_t* offsets,
                                      const index_t* old_ids,
                                      const index_t* new_ids, GraphGPU g,
                                      GraphGPU subg) {
  CUDA_KERNEL_LOOP(i, n) {
    subg.fixEndEdge(i, offsets[i + 1]);
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

/*
void Sampler::indexing(size_t n, index_t* vertices, index_t* new_indices) {
  index_t vid = 0;
  for (index_t i = 0; i < n; i++) {
    auto v         = vertices[i];
    new_indices[v] = vid++;
  }
}
*/

template <typename GraphTy, typename SubgraphTy>
void Sampler::getMaskedGraph(index_t n, mask_t* masks, GraphTy* g, SubgraphTy* subg) {
  std::cout << "Original graph size: " << g->size() << " edges: " << g->sizeEdges() << "\n";
  index_t *degrees, *offsets;
  CUDA_CHECK(cudaMalloc((void**)&degrees, sizeof(index_t)*n));
  get_masked_degrees<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, masks, *g, degrees);
  CUDA_CHECK(cudaMalloc((void**)&offsets, sizeof(index_t)*(n+1)));
  thrust::exclusive_scan(thrust::device, degrees, degrees+n+1, offsets);
  CUDA_CHECK(cudaFree(degrees));
  index_t ne;
  CUDA_CHECK(cudaMemcpy(&ne, &offsets[n], sizeof(index_t), cudaMemcpyDeviceToHost));
  std::cout << "maskedSG num_edges " << ne << "\n";
  subg->allocateFrom(n, ne); // TODO: avoid reallocation
  generate_masked_graph_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, masks, offsets, *g, *subg);
  CUDA_CHECK(cudaFree(offsets));
}

// n: size of the original graph
// nv: size of the subgraph; i.e. size of vertex_set
// masks, graph g and subgraph sub are on the device (GPU)
void Sampler::generateSubgraph(VertexSet &vertex_set, mask_t* masks, GraphGPU* sub) {
  index_t n = partGraph->size();
  auto nv = vertex_set.size();
  std::cout << "g size: " << n << " sg sizes: " << nv << "\n";
  // convert the vertex_set to a vertex_list and copy it to the device
  VertexList vertex_list(vertex_set.begin(), vertex_set.end());
  index_t* d_vertex_list;
  cudaMalloc((void**)&d_vertex_list, nv * sizeof(index_t));
  CUDA_CHECK(cudaMemcpy(d_vertex_list, &vertex_list[0], nv * sizeof(index_t), cudaMemcpyHostToDevice));

  createMasks(n, vertex_set, masks);
  mask_t* d_masks;
  cudaMalloc((void**)&d_masks, n * sizeof(mask_t));
  CUDA_CHECK(cudaMemcpy(d_masks, masks, n * sizeof(mask_t), cudaMemcpyHostToDevice));
  //clear_masks<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, d_masks); // set all 0
  //CudaTest("solving clear_masks kernel failed");
  // createMasks: set masks for vertices in the vertex_set
  //set_masks<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, d_vertex_list, d_masks);
  //CudaTest("solving set_masks kernel failed");
  GraphGPU masked_sg; // size is the same as original graph, but masked dst removed
  getMaskedGraph(n, d_masks, partGraph, &masked_sg); // remove edges whose destination is not masked
  std::cout << "maskedGraph generated\n";

  // re-index the subgraph
  index_t* d_new_ids;
  cudaMalloc((void**)&d_new_ids, n * sizeof(index_t));
  // Given an old vertex ID ∈ [0, n), returns a new vertex ID ∈ [0, nv)
  auto new_ids = reindexVertices(n, vertex_set);
  CUDA_CHECK(cudaMemcpy(d_new_ids, &new_ids[0], n * sizeof(index_t), cudaMemcpyHostToDevice));

  // generate the offsets for the re-indexed subgraph
  index_t *degrees, *offsets;
  CUDA_CHECK(cudaMalloc((void**)&degrees, sizeof(index_t)*nv));
  get_new_degrees<<<CUDA_GET_BLOCKS(nv), CUDA_NUM_THREADS>>>(nv, d_vertex_list, d_new_ids, masked_sg, degrees);
  CudaTest("solving get_new_degrees kernel failed");
  CUDA_CHECK(cudaMalloc((void**)&offsets, sizeof(index_t)*(nv+1)));
  thrust::exclusive_scan(thrust::device, degrees, degrees+nv+1, offsets);
  CUDA_CHECK(cudaFree(degrees));
  index_t ne;
  CUDA_CHECK(cudaMemcpy(&ne, offsets+nv, sizeof(index_t), cudaMemcpyDeviceToHost));
  std::cout << "subgraph num_edges " << ne << "\n";

  // allocate memory for the subgraph
  sub->allocateFrom(nv, ne); // avoid reallocation
  // generate the subgraph
  generate_graph_kernel<<<CUDA_GET_BLOCKS(nv), CUDA_NUM_THREADS>>>(nv, offsets, d_vertex_list, d_new_ids, masked_sg, *sub);
  CudaTest("solving generate_graph kernel failed");
  CUDA_CHECK(cudaFree(offsets));
  std::cout << "Subgraph generated\n";
}

} // namespace deepgalois
