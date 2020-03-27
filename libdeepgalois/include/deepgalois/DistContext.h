#ifndef __DG_DIST_CONTEXT__
#define __DG_DIST_CONTEXT__
/**
 * Based on common.hpp file of the Caffe deep learning library.
 */
#include "galois/graphs/GluonSubstrate.h"
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
  galois::graphs::GluonSubstrate<Graph>* syncSubstrate;

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

  void initializeSyncSubstrate();
  galois::graphs::GluonSubstrate<Graph>* getSyncSubstrate();

  Graph* getGraphPointer() {
    return graph_cpu;
  }

  //! return label for some node
  //! NOTE: this is LID, not GID
  label_t get_label(size_t i) {
    return labels[i];
  }

  //! returns pointer to the features of each local node
  float_t* get_in_ptr();
};

} // end deepgalois namespace

#endif
