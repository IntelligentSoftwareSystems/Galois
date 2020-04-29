#pragma once
#include "deepgalois/types.h"
#include <string>
#include <boost/iterator/counting_iterator.hpp>

namespace deepgalois {

class LearningGraph {
protected:
  bool is_device;
  index_t num_vertices_;
  index_t num_edges_;
  index_t *rowptr_;
  index_t *colidx_;
  index_t *degrees_;
  vdata_t *vertex_data_;
  edata_t *edge_data_;

public:
  //typedef index_t* iterator;
  using iterator = boost::counting_iterator<index_t>;
  LearningGraph();
  ~LearningGraph();
  void init(index_t nv, index_t ne) { num_vertices_ = nv; num_edges_ = ne; }
  void readGraph(std::string path, std::string dataset);
  index_t size() { return num_vertices_; }
  index_t sizeEdges() { return num_edges_; }
  index_t getDegree(index_t vid) { return degrees_[vid]; }
  index_t getEdgeDst(index_t eid) { return colidx_[eid]; }
  index_t edge_begin(index_t vid) { return rowptr_[vid]; }
  index_t edge_end(index_t vid) { return rowptr_[vid+1]; }
  index_t* row_start_ptr() { return rowptr_; }
  index_t* edge_dst_ptr() { return colidx_; }
  index_t* degrees_ptr() { return degrees_; }
  edata_t* edge_data_ptr() { return edge_data_; }
  vdata_t* vertex_data_ptr() { return vertex_data_; }
  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(num_vertices_); }
  void progressPrint(unsigned maxii, unsigned ii);
  void allocOnDevice(bool no_edge_data_);
  void copy_to_cpu(LearningGraph &copygraph);
  void copy_to_gpu(LearningGraph &copygraph);
};

}
