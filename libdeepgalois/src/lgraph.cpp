#include "deepgalois/lgraph.h"
#include <fstream>

namespace deepgalois {

LearningGraph::LearningGraph() : num_vertices_(0), num_edges_(0),
                                 rowptr_(NULL), colidx_(NULL), degrees_(NULL) {}

void LearningGraph::readGraph(std::string path, std::string dataset) {
  std::string file_dims = path + dataset + "-dims.bin";
  std::string file_rowptr = path + dataset + "-rowptr.bin";
  std::string file_colidx = path + dataset + "-colidx.bin";
  index_t dims[2];
  std::ifstream ifs;
  ifs.open(file_dims, std::ios::binary|std::ios::in);
  ifs.read((char*)dims, sizeof(index_t) * 2);
  ifs.close();
  num_vertices_ = dims[0];
  num_edges_ = dims[1];
  degrees_ = new index_t[num_vertices_];
  rowptr_ = new index_t[num_vertices_+1];
  colidx_ = new index_t[num_edges_];
  ifs.open(file_rowptr, std::ios::binary|std::ios::in);
  ifs.read((char*)rowptr_, sizeof(index_t) * (num_vertices_+1));
  ifs.close();
  ifs.open(file_colidx, std::ios::binary|std::ios::in);
  ifs.read((char*)colidx_, sizeof(index_t) * num_edges_);
  ifs.close();
}

}
