#pragma once
/**
 * Based on common.hpp file of the Caffe deep learning library.
 */

#include <string>
#include <cassert>
#include "deepgalois/types.h"
#include "deepgalois/reader.h"
#include "deepgalois/GraphTypes.h"

#ifdef __GALOIS_HET_CUDA__
#include "deepgalois/cutils.h"
#endif

namespace deepgalois {

class Context {
  std::string dataset;
  bool is_device;         // is this on device or host
  bool is_selfloop_added; // whether selfloop is added to the input graph

  label_t* d_labels;      // labels on device
  label_t* d_labels_subg; // labels for subgraph on device
  float_t* d_feats;       // input features on device
  float_t* d_feats_subg;  // input features for subgraph on device

  Reader reader;

public:
// TODO separate below to public and private
#ifndef __GALOIS_HET_CUDA__
  Graph* graph_cpu; // the input graph, |V| = N
  std::vector<Graph*> subgraphs_cpu;
  void add_selfloop(Graph& og, Graph& g);
  //! returns pointer to the graph
  Graph* getGraphPointer() { return graph_cpu; }
#else
  static cublasHandle_t cublas_handle_;         // used to call cuBLAS
  static cusparseHandle_t cusparse_handle_;     // used to call cuSPARSE
  static cusparseMatDescr_t cusparse_matdescr_; // used to call cuSPARSE
  static curandGenerator_t
      curand_generator_; // used to generate random numbers on GPU

  GraphGPU graph_gpu; // the input graph, |V| = N
  std::vector<GraphGPU*> subgraphs_gpu;
  GraphGPU* getGraphPointer() { return &graph_gpu; }
  GraphGPU* getSubgraphPointer(int id) { return subgraphs_gpu[id]; };
  float_t* get_feats_ptr() { return d_feats; }
  float_t* get_feats_subg_ptr() { return d_feats_subg; }
  label_t* get_labels_ptr() { return d_labels; }
  label_t* get_labels_subg_ptr() { return d_labels_subg; }
  inline static cublasHandle_t cublas_handle() { return cublas_handle_; }
  inline static cusparseHandle_t cusparse_handle() { return cusparse_handle_; }
  inline static cusparseMatDescr_t cusparse_matdescr() {
    return cusparse_matdescr_;
  }
  inline static curandGenerator_t curand_generator() {
    return curand_generator_;
  }
#endif

  Context();
  //! initializer for gpu; goes ahead and sets a few things
  Context(bool use_gpu)
      : is_device(use_gpu),
        is_selfloop_added(false), d_labels(NULL), d_labels_subg(NULL),
        d_feats(NULL), d_feats_subg(NULL) {}
  ~Context();

  size_t read_graph(bool selfloop);

  size_t read_masks(std::string mask_type, size_t n, size_t& begin, size_t& end,
                    mask_t* masks) {
    return reader.read_masks(mask_type, n, begin, end, masks);
  }

  void set_dataset(std::string dataset_str) {
    dataset = dataset_str;
    reader.init(dataset);
  }

  //! Checks if subgraph being used, sets currenet graph, then calls degreex
  //! counting
  Graph* getFullGraph();

  void copy_data_to_device(); // copy labels and input features
};

} // namespace deepgalois
