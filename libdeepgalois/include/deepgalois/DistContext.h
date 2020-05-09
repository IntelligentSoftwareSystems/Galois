#ifndef __DG_DIST_CONTEXT__
#define __DG_DIST_CONTEXT__
/**
 * Based on common.hpp file of the Caffe deep learning library.
 */
#include "galois/graphs/GluonSubstrate.h"
#include "deepgalois/types.h"
#include "deepgalois/Context.h"
#include "deepgalois/GraphTypes.h"

namespace deepgalois {

class DistContext {
  size_t num_classes;   // number of classes: E
  size_t feat_len;      // input feature length: D
  galois::graphs::GluonSubstrate<DGraph>* syncSubstrate;

  DGraph* partitionedGraph; // the input graph, |V| = N
  std::vector<Graph*> subgraphs_cpu;
  label_t* h_labels;      // labels for classification. Single-class label: Nx1,
                          // multi-class label: NxE
  label_t* h_labels_subg; // labels for subgraph
  float_t* h_feats;       // input features: N x D
  float_t* h_feats_subg;  // input features for subgraph

  // TODO needs to come from whole graph
  float_t* normFactors;  // normalization constant based on graph structure
  std::vector<float_t> norm_factors_subg; // normalization constant for subgraph

public:
  DistContext();
  ~DistContext();

  void saveDistGraph(DGraph* a) { partitionedGraph = a; }

  //! read labels of local nodes only
  size_t read_labels(std::string dataset_str);
  //! read features of local nodes only
  size_t read_features(std::string dataset_str);
  //! read masks of local nodes only
  size_t read_masks(std::string dataset_str, std::string mask_type, size_t n,
                    size_t& begin, size_t& end, mask_t* masks, DGraph* dGraph);

  // TODO define these
  void createSubgraphs(int) {}
  void gen_subgraph_labels(size_t, const mask_t*) {}
  void gen_subgraph_feats(size_t, const mask_t*) {}


  float_t* get_norm_factors_ptr() { return normFactors; }
  // TODO shouldn't return 0 always
  float_t* get_norm_factors_subg_ptr() { return &norm_factors_subg[0]; }
  DGraph* getGraphPointer() { return partitionedGraph; }
  Graph* getSubgraphPointer(int id) { return subgraphs_cpu[id]; };
  float_t* get_feats_ptr() { return h_feats; }
  float_t* get_feats_subg_ptr() { return h_feats_subg; }
  label_t* get_labels_ptr() { return h_labels; }
  label_t* get_labels_subg_ptr() { return h_labels_subg; }

  void initializeSyncSubstrate();
  galois::graphs::GluonSubstrate<DGraph>* getSyncSubstrate();

  //! allocate the norm factor vector
  void allocNormFactor();
  //! construct norm factor vector by using data from global graph
  void constructNormFactor(deepgalois::Context* globalContext);

  //! return label for some node
  //! NOTE: this is LID, not GID
  label_t get_label(size_t i) { return h_labels[i]; }

  //! returns pointer to the features of each local node
  float_t* get_in_ptr();
};

} // namespace deepgalois

#endif
