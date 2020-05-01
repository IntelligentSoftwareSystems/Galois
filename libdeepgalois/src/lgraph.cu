#include "deepgalois/lgraph.h"
#include "deepgalois/cutils.h"
#include <cassert>

namespace deepgalois {

void LearningGraph::dealloc() {
  assert(is_device);
  CUDA_CHECK(cudaFree(colidx_));
  CUDA_CHECK(cudaFree(rowptr_));
  CUDA_CHECK(cudaFree(degrees_));
  if (edge_data_ != NULL) CUDA_CHECK(cudaFree(edge_data_));
  if (vertex_data_ != NULL) CUDA_CHECK(cudaFree(vertex_data_));
}

void LearningGraph::allocOnDevice(bool no_edge_data__) {
  if (colidx_ != NULL) return;  
  CUDA_CHECK(cudaMalloc((void **) &colidx_, num_edges_ * sizeof(index_t)));
  CUDA_CHECK(cudaMalloc((void **) &rowptr_, (num_vertices_+1) * sizeof(index_t)));
  CUDA_CHECK(cudaMalloc((void **) &degrees_, num_vertices_ * sizeof(index_t)));
  //if (!no_edge_data__) CUDA_CHECK(cudaMalloc((void **) &edge_data__, num_edges_ * sizeof(edge_data___t)));
  //CUDA_CHECK(cudaMalloc((void **) &vertex_data__, num_vertices_ * sizeof(vdata_t)));
  is_device = true;
}

void LearningGraph::copy_to_gpu(LearningGraph &copygraph) {
  copygraph.init(num_vertices_, num_edges_);
  copygraph.allocOnDevice(edge_data_ == NULL);
  CUDA_CHECK(cudaMemcpy(copygraph.colidx_, colidx_, num_edges_ * sizeof(index_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(copygraph.rowptr_, rowptr_, (num_vertices_+1) * sizeof(index_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(copygraph.degrees_, degrees_, num_vertices_ * sizeof(index_t), cudaMemcpyHostToDevice));
  //if (edge_data__ != NULL) CUDA_CHECK(cudaMemcpy(copygraph.edge_data__, edge_data__, num_edges_ * sizeof(edata_t), cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaMemcpy(copygraph.vertex_data__, vertex_data__, num_vertices_ * sizeof(vdata_t), cudaMemcpyHostToDevice));
}

void LearningGraph::copy_to_cpu(LearningGraph &copygraph) {
  assert(is_device);
  assert(copygraph.size() == num_vertices_);
  assert(copygraph.sizeEdges() == num_edges_);
  CUDA_CHECK(cudaMemcpy(copygraph.edge_dst_ptr(), colidx_, num_edges_ * sizeof(index_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(copygraph.row_start_ptr(), rowptr_, (num_vertices_+1) * sizeof(index_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(copygraph.degrees_ptr(), degrees_, num_vertices_ * sizeof(index_t), cudaMemcpyDeviceToHost));
  //if (edge_data__ != NULL) CUDA_CHECK(cudaMemcpy(copygraph.edge_data__ptr(), edge_data__, num_edges_ * sizeof(edata_t), cudaMemcpyDeviceToHost));
  //CUDA_CHECK(cudaMemcpy(copygraph.vertex_data__ptr(), vertex_data__, num_vertices_ * sizeof(vdata_t), cudaMemcpyDeviceToHost));
}

}
