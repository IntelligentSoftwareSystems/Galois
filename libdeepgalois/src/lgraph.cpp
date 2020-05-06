#include "deepgalois/lgraph.h"
#include "deepgalois/utils.h"
#include "deepgalois/reader.h"
#include "galois/Galois.h"
#include <iostream>

namespace deepgalois {

bool LearningGraph::isLocal(index_t) { return true; }

index_t LearningGraph::getLID(index_t) { return 0; }

bool LearningGraph::is_vertex_cut() {return true; }

std::vector<std::vector<size_t>>& LearningGraph::getMirrorNodes() {
  return mirrorNodes;
}

uint64_t LearningGraph::numMasters() { return 0; }

uint64_t LearningGraph::globalSize() { return 0; }

void LearningGraph::readGraph(std::string dataset) {
  deepgalois::Reader reader(dataset);
  reader.readGraphFromGRFile(this);
}

void LearningGraph::degree_counting() {
  //if (degrees_ != NULL) return;
  //degrees_ = new index_t[num_vertices_];
  galois::do_all(galois::iterate(size_t(0), size_t(num_vertices_)), [&] (auto v) {
    degrees_[v] = rowptr_[v+1] - rowptr_[v];
  }, galois::loopname("DegreeCounting"));
}

void LearningGraph::dealloc() {}

} // end namespace
