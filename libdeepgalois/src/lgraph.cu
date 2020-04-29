
void LearningGraph::dealloc() {
  assert(is_device);
  CUDA_CHECK(cudaFree(colidx_));
  CUDA_CHECK(cudaFree(rowptr_));
  CUDA_CHECK(cudaFree(degrees_));
  if (edge_data != NULL) CUDA_CHECK(cudaFree(edge_data));
  if (vertex_data != NULL) CUDA_CHECK(cudaFree(vertex_data));
}

void LearningGraph::allocOnDevice(bool no_edge_data_) {
  if (colidx_ != NULL) return true;  
  CUDA_CHECK(cudaMalloc((void **) &colidx_, num_edges_ * sizeof(index_type)));
  CUDA_CHECK(cudaMalloc((void **) &rowptr_, (num_vertices_+1) * sizeof(index_type)));
  CUDA_CHECK(cudaMalloc((void **) &degrees_, num_vertices_ * sizeof(index_type)));
  //if (!no_edge_data_) CUDA_CHECK(cudaMalloc((void **) &edge_data_, num_edges_ * sizeof(edge_data__t)));
  //CUDA_CHECK(cudaMalloc((void **) &vertex_data_, num_vertices_ * sizeof(vdata_t)));
  is_device = true;
}

void LearningGraph::copy_to_gpu(LearningGraph &copygraph) {
  copygraph.init(num_vertices_, num_edges_);
  copygraph.allocOnDevice(edge_data_ == NULL);
  CUDA_CHECK(cudaMemcpy(copygraph.colidx_, colidx_, num_edges_ * sizeof(index_type), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(copygraph.rowptr_, rowptr_, (num_vertices_+1) * sizeof(index_type), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(copygraph.degrees_, degrees_, num_vertices_ * sizeof(index_type), cudaMemcpyHostToDevice));
  //if (edge_data_ != NULL) CUDA_CHECK(cudaMemcpy(copygraph.edge_data_, edge_data_, num_edges_ * sizeof(edata_t), cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaMemcpy(copygraph.vertex_data_, vertex_data_, num_vertices_ * sizeof(vdata_t), cudaMemcpyHostToDevice));
}

void LearningGraph::copy_to_cpu(LearningGraph &copygraph) {
  assert(is_device);
  assert(copygraph.size() = num_vertices_);
  assert(copygraph.sizeEdges() = num_edges_);
  CUDA_CHECK(cudaMemcpy(copygraph.edge_dst_ptr(), colidx_, num_edges_ * sizeof(index_type), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(copygraph.row_start_ptr(), rowptr_, (num_vertices_+1) * sizeof(index_type), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(copygraph.degrees_ptr(), degrees_, num_vertices_ * sizeof(index_type), cudaMemcpyDeviceToHost));
  //if (edge_data_ != NULL) CUDA_CHECK(cudaMemcpy(copygraph.edge_data_ptr(), edge_data_, num_edges_ * sizeof(edata_t), cudaMemcpyDeviceToHost));
  //CUDA_CHECK(cudaMemcpy(copygraph.vertex_data_ptr(), vertex_data_, num_vertices_ * sizeof(vdata_t), cudaMemcpyDeviceToHost));
}

