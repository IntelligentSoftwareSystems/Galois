#pragma once
/**
 * Based on common.hpp file of the Caffe deep learning library.
 */

#include <string>
#include <cassert>
#include "deepgalois/types.h"
//#include <boost/shared_ptr.hpp>
#ifdef CPU_ONLY
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
  size_t read_graph_cpu(std::string dataset_str, std::string filetype, bool selfloop);
  size_t read_graph_gpu(std::string dataset_str, bool selfloop);
  size_t read_labels(std::string dataset_str);
  size_t read_features(std::string dataset_str, std::string filetype = "bin");
  size_t read_masks(std::string dataset_str, std::string mask_type,
                    size_t n, size_t& begin, size_t& end, mask_t* masks);

  label_t get_label(size_t i) { return h_labels[i]; } // single-class (one-hot) label
  //label_t get_label(size_t i, size_t j) { return labels[i*num_classes+j]; } // multi-class label
  float_t* get_norm_factors_ptr() { return norm_factors; }
  float_t* get_norm_factors_subg_ptr() { return norm_factors_subg; }

  void set_label_class(bool is_single = true) { is_single_class = is_single; }
  void set_use_subgraph(bool use_subg) { use_subgraph = use_subg; }
  void copy_data_to_device(); // copy labels and input features
  void norm_factor_computing(bool is_subgraph, int subg_id = 0);
  void gen_subgraph_labels(size_t m, const mask_t *masks);
  void gen_subgraph_feats(size_t m, const mask_t *masks);
  void createSubgraphs(int num_subgraphs);

#ifdef CPU_ONLY
  Graph* graph_cpu; // the input graph, |V| = N
  std::vector<Graph*> subgraphs_cpu;
  void add_selfloop(Graph &og, Graph &g);
  //! returns pointer to the graph
  Graph* getGraphPointer() { return graph_cpu; }
  Graph* getSubgraphPointer(int id) { return subgraphs_cpu[id]; };
  float_t* get_feats_ptr() { return h_feats; }
  float_t* get_feats_subg_ptr() { return h_feats_subg; }
  label_t* get_labels_ptr() { return h_labels; }
  label_t* get_labels_subg_ptr() { return h_labels_subg; }
#else
  CSRGraph graph_gpu; // the input graph, |V| = N
  CSRGraph subgraph_gpu;
  CSRGraph* getGraphPointer() { return &graph_gpu; }
  CSRGraph* getSubgraphPointer() { return &subgraph_gpu; };
  float_t* get_feats_ptr() { return d_feats; }
  float_t* get_feats_subg_ptr() { return d_feats_subg; }
  label_t* get_labels_ptr() { return d_labels; }
  label_t* get_labels_subg_ptr() { return d_labels_subg; }
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
  bool is_selfloop_added;      // whether selfloop is added to the input graph
  bool use_subgraph;           // whether to use subgraph
  label_t *h_labels;           // labels for classification. Single-class label: Nx1, multi-class label: NxE 
  label_t *h_labels_subg;      // labels for subgraph
  float_t* h_feats;            // input features: N x D
  float_t* h_feats_subg;       // input features for subgraph
  label_t* d_labels;           // labels on device
  label_t *d_labels_subg;      // labels for subgraph on device
  float_t* d_feats;            // input features on device
  float_t* d_feats_subg;       // input features for subgraph on device
  float_t* norm_factors;       // normalization constant based on graph structure
  float_t* norm_factors_subg;  // normalization constant for subgraph
  void alloc_norm_factor();
  void alloc_subgraph_norm_factor(int subg_id);

#ifdef CPU_ONLY
  void read_edgelist(const char* filename, bool symmetrize = false, bool add_self_loop = false);
#else
  static cublasHandle_t cublas_handle_; // used to call cuBLAS
  static cusparseHandle_t cusparse_handle_; // used to call cuSPARSE
  static cusparseMatDescr_t cusparse_matdescr_; // used to call cuSPARSE
  static curandGenerator_t curand_generator_; // used to generate random numbers on GPU
#endif
};

} // end deepgalois namespace
