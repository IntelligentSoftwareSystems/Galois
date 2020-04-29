#pragma once
#include "deepgalois/types.h"
#include <string>
#include <boost/iterator/counting_iterator.hpp>

namespace deepgalois {

typedef uint32_t index_t;

class LearningGraph {
protected:
  index_t num_vertices_;
  index_t num_edges_;
  index_t *rowptr_;
  index_t *colidx_;
  index_t *degrees_;
public:
  //typedef index_t* iterator;
  using iterator = boost::counting_iterator<index_t>;
  LearningGraph();
  ~LearningGraph();
  void readGraph(std::string path, std::string dataset);
  index_t getDegree(index_t vid) { return degrees_[vid]; }
  index_t getEdgeDst(index_t eid) { return colidx_[eid]; }
  index_t edge_begin(index_t vid) { return rowptr_[vid]; }
  index_t edge_end(index_t vid) { return rowptr_[vid+1]; }
  index_t* row_start_ptr() { return rowptr_; }
  index_t* edge_dst_ptr() { return colidx_; }
  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(num_vertices_); }
};

}
