#include "deepgalois/lgraph.h"
#include "deepgalois/utils.h"
#include "deepgalois/reader.h"
#include "galois/Galois.h"
#include <iostream>
#include <cassert>

namespace deepgalois {

bool LearningGraph::isLocal(index_t) { return true; }

index_t LearningGraph::getLID(index_t) { return 0; }

bool LearningGraph::is_vertex_cut() {return true; }

std::vector<std::vector<size_t>>& LearningGraph::getMirrorNodes() {
  return mirrorNodes;
}

uint64_t LearningGraph::numMasters() { return 0; }

uint64_t LearningGraph::globalSize() { return 0; }

void LearningGraph::constructNodes() {
}

void LearningGraph::readGraph(std::string dataset) {
  deepgalois::Reader reader(dataset);
  reader.readGraphFromGRFile(this);
}

void LearningGraph::fixEndEdge(index_t vid, index_t row_end) {
  rowptr_[vid+1] = row_end;
}

void LearningGraph::constructEdge(index_t eid, index_t dst, edata_t edata) {
  assert(dst < num_vertices_);
  assert(eid < num_edges_);
  colidx_[eid] = dst;
  if (edge_data_) edge_data_[eid] = edata;
}

void LearningGraph::degree_counting() {
  //if (degrees_ != NULL) return;
  //degrees_ = new index_t[num_vertices_];
  galois::do_all(galois::iterate(size_t(0), size_t(num_vertices_)), [&] (auto v) {
    degrees_[v] = rowptr_[v+1] - rowptr_[v];
  }, galois::loopname("DegreeCounting"));
}

void LearningGraph::add_selfloop() {
  //print_neighbors(nnodes-1);
  //print_neighbors(0);
  auto old_colidx_ = colidx_;
  colidx_.resize(num_vertices_ + num_edges_);
  for (index_t i = 0; i < num_vertices_; i++) {
    auto start = rowptr_[i];
    auto end = rowptr_[i+1];
    bool selfloop_inserted = false;
    if (start == end) {
      colidx_[start+i] = i;
      continue;
    }
    for (auto e = start; e != end; e++) {
      auto dst = old_colidx_[e];
      if (!selfloop_inserted) {
        if (i < dst) {
          selfloop_inserted = true;
          colidx_[e+i] = i;
          colidx_[e+i+1] = dst;
        } else if (e+1 == end) {
          selfloop_inserted = true;
          colidx_[e+i+1] = i;
          colidx_[e+i] = dst;
        } else colidx_[e+i] = dst;
      } else colidx_[e+i+1] = dst;
    }
  }
  for (index_t i = 0; i <= num_vertices_; i++) rowptr_[i] += i;
  num_edges_ += num_vertices_;
  //print_neighbors(nnodes-1);
  //print_neighbors(0);
}

#ifdef CPU_ONLY
void LearningGraph::dealloc() {
/*
  assert (!is_device);
  if (rowptr_ != NULL) delete [] rowptr_;
  if (colidx_ != NULL) delete [] colidx_;
  if (degrees_ != NULL) delete [] degrees_;
  if (vertex_data_ != NULL) delete [] vertex_data_;
  if (edge_data_ != NULL) delete [] edge_data_;
//*/
}
#endif

} // end namespace
