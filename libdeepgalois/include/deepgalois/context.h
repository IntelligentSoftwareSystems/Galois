#pragma once
/**
 * Based on common.hpp file of the Caffe deep learning library.
 */

#include <string>
#include <cassert>
#include "deepgalois/types.h"
#include "deepgalois/utils.h"
#ifdef CPU_ONLY
#include "deepgalois/lgraph.h"
#include "deepgalois/gtypes.h"
#else
#include "graph_gpu.h"
#include "deepgalois/cutils.h"
#endif

namespace deepgalois {
class Context {
public:
  Context();
  ~Context();

  size_t read_graph(std::string dataset_str, bool selfloop);
  size_t read_labels(std::string dataset_str);
  size_t read_features(std::string dataset_str, std::string filetype = "bin");
  label_t get_label(size_t i) { return labels[i]; } // single-class (one-hot) label
  label_t get_label(size_t i, size_t j) { return labels[i*num_classes+j]; } // multi-class label
  label_t* get_labels_ptr() { return labels; }
  float_t* get_in_ptr();

  size_t read_graph_cpu(std::string dataset_str, std::string filetype, bool selfloop);
  size_t read_graph_gpu(std::string dataset_str, bool selfloop);
  void copy_data_to_device(); // copy labels and input features
  void norm_factor_counting();
  void set_label_class(bool is_single = true) { is_single_class = is_single; }

  float_t* d_feats;            // input features on device
  label_t* d_labels;           // labels on device
  float_t* norm_factor;        // normalization constant based on graph structure

#ifdef CPU_ONLY
  Graph* graph_cpu; // the input graph, |V| = N
  void genGraph(LGraph& lg, Graph& g);
  void add_selfloop(Graph &og, Graph &g);
  //! returns pointer to the graph
  Graph* getGraphPointer();
#else
  CSRGraph graph_gpu; // the input graph, |V| = N
  inline static cublasHandle_t cublas_handle() { return cublas_handle_; }
  inline static cusparseHandle_t cusparse_handle() { return cusparse_handle_; }
  inline static cusparseMatDescr_t cusparse_matdescr() { return cusparse_matdescr_; }
  inline static curandGenerator_t curand_generator() { return curand_generator_; }
#endif

protected:
  size_t n;                    // number of samples: N
  size_t num_classes;          // number of classes: E
  size_t feat_len;             // input feature length: D
  bool is_single_class;        // single-class (one-hot) or multi-class label
  label_t *labels;             // labels for classification: N x 1
  float_t* h_feats;            // input features: N x D
#ifndef CPU_ONLY
  static cublasHandle_t cublas_handle_; // used to call cuBLAS
  static cusparseHandle_t cusparse_handle_; // used to call cuSPARSE
  static cusparseMatDescr_t cusparse_matdescr_; // used to call cuSPARSE
  static curandGenerator_t curand_generator_; // used to generate random numbers on GPU
#endif
};
} // end deepgalois namespace
