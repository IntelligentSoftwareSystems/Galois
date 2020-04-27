#ifndef __DG_DIST_CONTEXT__
#define __DG_DIST_CONTEXT__
/**
 * Based on common.hpp file of the Caffe deep learning library.
 */
#include "galois/graphs/GluonSubstrate.h"
#include "deepgalois/types.h"
#include "deepgalois/gtypes.h"

namespace deepgalois {

class DistContext {
protected:
  size_t localVertices;        // number of samples: N
  size_t num_classes;          // number of classes: E
  size_t feat_len;             // input feature length: D
  galois::graphs::GluonSubstrate<Graph>* syncSubstrate;

  Graph* graph_cpu;            // the input graph, |V| = N
  Graph* subgraph_cpu;
  label_t *h_labels;           // labels for classification. Single-class label: Nx1, multi-class label: NxE 
  label_t *h_labels_subg;      // labels for subgraph
  float_t* h_feats;            // input features: N x D
  float_t* h_feats_subg;       // input features for subgraph
  label_t* d_labels;           // labels on device
  label_t *d_labels_subg;      // labels for subgraph on device
  float_t* d_feats;            // input features on device
  float_t* d_feats_subg;       // input features for subgraph on device
  float_t* norm_factor;        // normalization constant based on graph structure

public:
  DistContext();
  ~DistContext();

  //! save graph pointer to context object
  void saveGraph(Graph* dGraph);

  //! read labels of local nodes only
  size_t read_labels(std::string dataset_str);

  //! read features of local nodes only
  size_t read_features(std::string dataset_str);

  //! read masks of local nodes only
  size_t read_masks(std::string dataset_str, std::string mask_type,
                    size_t n, size_t& begin, size_t& end, mask_t* masks, Graph* dGraph);

  //! find norm factor by looking at degree
  // TODO this is a distributed operation
  void norm_factor_counting();
  void createSubgraph() {}

  float_t* get_norm_factor_ptr() { return norm_factor; }
  Graph* getGraphPointer() { return graph_cpu; }
  Graph* getSubgraphPointer() { return subgraph_cpu; };
  float_t* get_feats_ptr() { return h_feats; }
  float_t* get_feats_subg_ptr() { return h_feats_subg; }
  label_t* get_labels_ptr() { return h_labels; }
  label_t* get_labels_subg_ptr() { return h_labels_subg; }

  void initializeSyncSubstrate();
  galois::graphs::GluonSubstrate<Graph>* getSyncSubstrate();


  //! return label for some node
  //! NOTE: this is LID, not GID
  label_t get_label(size_t i) { return h_labels[i]; }

  //! returns pointer to the features of each local node
  float_t* get_in_ptr();
};

} // end deepgalois namespace

#endif
