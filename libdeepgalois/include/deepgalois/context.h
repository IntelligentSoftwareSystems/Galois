#pragma once
/**
 * Code modified from below
 *
 * https://github.com/BVLC/caffe/blob/master/include/caffe/common.hpp
 *
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 * Reused/revised under BSD 2-Clause license
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
  enum Brew { CPU, GPU };
  Brew mode() { return mode_; }
  void set_mode(Brew mode) { mode_ = mode; }
  int solver_count() { return solver_count_; }
  void set_solver_count(int val) { solver_count_ = val; }
  int solver_rank() { return solver_rank_; }
  void set_solver_rank(int val) { solver_rank_ = val; }
  bool multiprocess() { return multiprocess_; }
  void set_multiprocess(bool val) { multiprocess_ = val; }
  bool root_solver() { return solver_rank_ == 0; }
  size_t read_graph(std::string dataset_str, bool selfloop);
  size_t read_labels(std::string dataset_str);
  size_t read_features(std::string dataset_str);
  label_t get_label(size_t i) { return labels[i]; }
  label_t* get_labels_ptr(size_t i) { return &(labels[0]); }
  float_t* get_in_ptr();

  size_t read_graph_cpu(std::string dataset_str, std::string filetype, bool selfloop);
  size_t read_graph_gpu(std::string dataset_str, bool selfloop);
  void copy_data_to_device(); // copy labels and input features
  void SetDevice(const int device_id);
  void DeviceQuery() {}
  bool CheckDevice(const int device_id) { return true; }
  int FindDevice(const int start_id = 0) { return 0; }
  void norm_factor_counting();
  void norm_factor_counting_gpu();

  size_t n;                    // number of samples: N
  size_t num_classes;          // number of classes: E
  size_t feat_len;             // input feature length: D
  std::vector<label_t> labels; // labels for classification: N x 1
  label_t* d_labels;           // labels on device
  vec_t h_feats;               // input features: N x D
  float_t* d_feats;            // input features on device
  float_t* norm_factor;   // normalization constant based on graph structure
  float_t* d_norm_factor; // norm_factor on device

#ifdef CPU_ONLY
  Graph graph_cpu; // the input graph, |V| = N
  void genGraph(LGraph& lg, Graph& g);
  void add_selfloop(Graph &og, Graph &g);
#else
  CSRGraph graph_gpu; // the input graph, |V| = N
  inline static cublasHandle_t cublas_handle() { return cublas_handle_; }
  inline static cusparseHandle_t cusparse_handle() { return cusparse_handle_; }
  inline static cusparseMatDescr_t cusparse_matdescr() { return cusparse_matdescr_; }
  inline static curandGenerator_t curand_generator() {
    return curand_generator_;
  }
#endif

protected:
#ifndef CPU_ONLY
  static cublasHandle_t cublas_handle_; // used to call cuBLAS
  static cusparseHandle_t cusparse_handle_; // used to call cuSPARSE
  static cusparseMatDescr_t cusparse_matdescr_; // used to call cuSPARSE
  static curandGenerator_t curand_generator_; // used to generate random numbers on GPU
#endif
  Brew mode_;
  int solver_count_;
  int solver_rank_;
  bool multiprocess_;
};
} // end deepgalois namespace
