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
  size_t localVertices;        // number of samples: N
  size_t num_classes;          // number of classes: E
  size_t feat_len;             // input feature length: D
  std::vector<label_t> labels; // labels for classification: N x 1
  vec_t h_feats;               // input features: N x D

public:
  // TODO why are these public
  float_t* norm_factor; // normalization constant based on graph structure
  Graph* graph_cpu; // the input graph, |V| = N

  DistContext();
  ~DistContext();

  //! save graph pointer to context object
  void saveGraph(Graph* dGraph);
  //! read labels of local nodes only
  size_t read_labels(std::string dataset_str);
  //! read features of local nodes only
  size_t read_features(std::string dataset_str);
  //! find norm factor by looking at degree
  // TODO this is a distributed operation
  void norm_factor_counting();

  //! return label for some node
  label_t get_label(size_t i) {
    // TODO global id only or lid only or both?
    return labels[i];
  }

  //! returns pointer to the features of each local node
  float_t* get_in_ptr();
};

} // end deepgalois namespace

#endif
