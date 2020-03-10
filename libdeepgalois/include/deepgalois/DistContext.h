#ifndef __DG_DIST_CONTEXT__
#define __DG_DIST_CONTEXT__
/**
 * Based on common.hpp file of the Caffe deep learning library.
 */
#include "deepgalois/types.h"
#include "deepgalois/utils.h"
#include "deepgalois/gtypes.h"

namespace deepgalois {

class DistContext {
  size_t n;                    // number of samples: N
  size_t num_classes;          // number of classes: E
  size_t feat_len;             // input feature length: D
  std::vector<label_t> labels; // labels for classification: N x 1
  vec_t h_feats;               // input features: N x D

public:
  DistContext();
  ~DistContext();

  void saveGraph(Graph* dGraph);
  size_t read_labels(std::string dataset_str);
  size_t read_features(std::string dataset_str);
  void norm_factor_counting();

  // TODO why are these public
  float_t* norm_factor;        // normalization constant based on graph structure
  Graph* graph_cpu; // the input graph, |V| = N

  label_t get_label(size_t i) {
    // TODO global id only or lid only or both?
    return labels[i];
  }
  size_t read_graph_cpu(std::string dataset_str);
  float_t* get_in_ptr();
};

} // end deepgalois namespace

#endif
