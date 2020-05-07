#include "deepgalois/lgraph.h"
#include "deepgalois/cutils.h"
#include "deepgalois/reader.h"
#include <cassert>

namespace deepgalois {

void LearningGraph::readGraph(std::string dataset) {
  deepgalois::Reader reader(dataset);
  reader.readGraphFromGRFile(this);
}

void LearningGraph::dealloc() {
  assert(is_device);
  CUDA_CHECK(cudaFree(d_colidx_));
  CUDA_CHECK(cudaFree(d_rowptr_));
  CUDA_CHECK(cudaFree(d_degrees_));
  if (edge_data_ != NULL) CUDA_CHECK(cudaFree(d_edge_data_));
  if (vertex_data_ != NULL) CUDA_CHECK(cudaFree(d_vertex_data_));
}

void LearningGraph::allocOnDevice(bool no_edge_data__) {
  if (d_colidx_ != NULL) return;  
  CUDA_CHECK(cudaMalloc((void **) &d_colidx_, num_edges_ * sizeof(index_t)));
  CUDA_CHECK(cudaMalloc((void **) &d_rowptr_, (num_vertices_+1) * sizeof(index_t)));
  //CUDA_CHECK(cudaMalloc((void **) &d_degrees_, num_vertices_ * sizeof(index_t)));
  //if (!no_edge_data__) CUDA_CHECK(cudaMalloc((void **) &edge_data__, num_edges_ * sizeof(edge_data___t)));
  //CUDA_CHECK(cudaMalloc((void **) &vertex_data__, num_vertices_ * sizeof(vdata_t)));
  is_device = true;
}

void LearningGraph::print_test() {
  printf("d_rowptr_: 0x%x\n", d_rowptr_);
  printf("d_colidx_: 0x%x\n", d_colidx_);
  print_device_int_vector(10, (const int*)d_rowptr_, "row_start");
  print_device_int_vector(10, (const int*)d_colidx_, "edge_dst");
}

void LearningGraph::copy_to_gpu() {
  allocOnDevice(edge_data_ == NULL);
  CUDA_CHECK(cudaMemcpy(d_colidx_, edge_dst_host_ptr(), num_edges_ * sizeof(index_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rowptr_, row_start_host_ptr(), (num_vertices_+1) * sizeof(index_t), cudaMemcpyHostToDevice));
  print_test();
  //CUDA_CHECK(cudaMemcpy(degrees_ptr(), d_degrees_, num_vertices_ * sizeof(index_t), cudaMemcpyHostToDevice));
  //if (edge_data__ != NULL) CUDA_CHECK(cudaMemcpy(copygraph.edge_data__, edge_data__, num_edges_ * sizeof(edata_t), cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaMemcpy(copygraph.vertex_data__, vertex_data__, num_vertices_ * sizeof(vdata_t), cudaMemcpyHostToDevice));
}

void LearningGraph::copy_to_cpu() {
  CUDA_CHECK(cudaMemcpy(edge_dst_host_ptr(), d_colidx_, num_edges_ * sizeof(index_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(row_start_host_ptr(), d_rowptr_, (num_vertices_+1) * sizeof(index_t), cudaMemcpyDeviceToHost));
  //CUDA_CHECK(cudaMemcpy(degrees_ptr(), d_degrees_, num_vertices_ * sizeof(index_t), cudaMemcpyDeviceToHost));
  //if (edge_data__ != NULL) CUDA_CHECK(cudaMemcpy(copygraph.edge_data__ptr(), edge_data__, num_edges_ * sizeof(edata_t), cudaMemcpyDeviceToHost));
  //CUDA_CHECK(cudaMemcpy(copygraph.vertex_data__ptr(), vertex_data__, num_vertices_ * sizeof(vdata_t), cudaMemcpyDeviceToHost));
}

void LearningGraph::degree_counting() {}

}
