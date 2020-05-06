#pragma once
#include "deepgalois/types.h"
#include <string>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace deepgalois {

class LearningGraph {
  typedef std::vector<index_t> IndexList;
  //typedef index_t* IndexList;
protected:
  bool is_device;
  index_t num_vertices_;
  index_t num_edges_;
  IndexList rowptr_;
  IndexList colidx_;
  IndexList degrees_;
  vdata_t *vertex_data_;
  edata_t *edge_data_;

  index_t *d_rowptr_;
  index_t *d_colidx_;
  index_t *d_degrees_;
  vdata_t *d_vertex_data_;
  edata_t *d_edge_data_;
  std::vector<std::vector<size_t>> mirrorNodes;

public:
  typedef size_t iterator;
  LearningGraph(bool use_gpu) : is_device(use_gpu), num_vertices_(0), num_edges_(0),
                                //rowptr_(NULL), colidx_(NULL), degrees_(NULL),
                                vertex_data_(NULL), edge_data_(NULL) {}
  LearningGraph() : LearningGraph(false) {}
  ~LearningGraph() { dealloc(); }
  void init(index_t nv, index_t ne) { num_vertices_ = nv; num_edges_ = ne; }
  size_t size() { return (size_t)num_vertices_; }
  size_t sizeEdges() { return (size_t)num_edges_; }
  index_t get_degree(index_t vid) { return degrees_[vid]; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(num_vertices_); }
  void progressPrint(unsigned maxii, unsigned ii);
  void allocOnDevice(bool no_edge_data_);
  void copy_to_cpu();
  void copy_to_gpu();
  void dealloc();
  void degree_counting();
  void constructNodes();
  void fixEndEdge(index_t vid, index_t row_end);
  void constructEdge(index_t eid, index_t dst, edata_t edata);
  void add_selfloop();
  void readGraph(std::string dataset);
  void allocateFrom(index_t nv, index_t ne) {
    //printf("Allocating num_vertices_=%d, num_edges_=%d.\n", num_vertices_, num_edges_);
    num_vertices_ = nv;
    num_edges_ = ne;
    rowptr_.resize(num_vertices_+1);
    colidx_.resize(num_edges_);
    degrees_.resize(num_vertices_);
    rowptr_[0] = 0;
  }
  bool isLocal(index_t vid);
  index_t getLID(index_t vid);
  bool is_vertex_cut();
  std::vector<std::vector<size_t>>& getMirrorNodes();
  uint64_t numMasters();
  uint64_t globalSize();

#ifdef CPU_ONLY
  index_t getEdgeDst(index_t eid) { return colidx_[eid]; }
  index_t edge_begin(index_t vid) { return rowptr_[vid]; }
  index_t edge_end(index_t vid) { return rowptr_[vid+1]; }
	vdata_t getData(index_t vid) { return vertex_data_[vid]; }
  index_t getDegree(index_t vid) { return degrees_[vid]; }
  index_t* row_start_ptr() { return &rowptr_[0]; }
  const index_t* row_start_ptr() const { return &rowptr_[0]; }
  index_t* edge_dst_ptr() { return &colidx_[0]; }
  const index_t* edge_dst_ptr() const { return &colidx_[0]; }
  index_t* degrees_ptr() { return &degrees_[0]; }
  edata_t* edge_data_ptr() { return edge_data_; }
  vdata_t* vertex_data_ptr() { return vertex_data_; }
#else
	CUDA_HOSTDEV index_t getEdgeDst(index_t edge) { return d_colidx_[edge]; }
	CUDA_HOSTDEV index_t edge_begin(index_t src) { return d_rowptr_[src]; }
	CUDA_HOSTDEV index_t edge_end(index_t src) { return d_rowptr_[src+1]; }
	CUDA_HOSTDEV vdata_t getData(index_t vid) { return d_vertex_data_[vid]; }
	CUDA_HOSTDEV index_t getDegree(index_t vid) { return d_degrees_[vid]; }
	CUDA_HOSTDEV index_t getOutDegree(index_t vid) { return d_degrees_[vid]; }
	index_t *row_start_ptr() { return d_rowptr_; }
	const index_t *row_start_ptr() const { return d_rowptr_; }
	index_t *edge_dst_ptr() { return d_colidx_; }
	const index_t *edge_dst_ptr() const { return d_colidx_; }
  index_t* degrees_ptr() { return d_degrees_; }
	edata_t *edge_data_ptr() { return d_edge_data_; }
	vdata_t *vertex_data_ptr() { return d_vertex_data_; }
	//const vdata_t *vertex_data_ptr() const { return vertex_data_; }
	//const edata_t *edge_data_ptr() const { return edge_data; }
#endif

};

}
