#ifndef __DG_DIST_CONTEXT__
#define __DG_DIST_CONTEXT__
/**
 * Based on common.hpp file of the Caffe deep learning library.
 */
#ifndef __GALOIS_HET_CUDA__
#include "galois/graphs/GluonSubstrate.h"
#endif
#include "deepgalois/types.h"
#include "deepgalois/Context.h"
#include "deepgalois/GraphTypes.h"

namespace deepgalois {

class DistContext {
  size_t num_classes; // number of classes: E
  size_t feat_len;    // input feature length: D
  Graph* lGraph;            // laerning graph version
#ifndef __GALOIS_HET_CUDA__
  galois::graphs::GluonSubstrate<DGraph>* syncSubstrate;
#endif
  DGraph* partitionedGraph; // the input graph, |V| = N
  std::vector<Graph*> partitionedSubgraphs;
  label_t* h_labels; // labels for classification. Single-class label: Nx1,
                     // multi-class label: NxE
  float_t* h_feats;                    // input features: N x D
  std::vector<label_t> h_labels_subg;  // labels for subgraph
  std::vector<float_t> h_feats_subg;   // input features for subgraph
  std::vector<float_t> normFactors;    // normalization constant based on graph structure
  std::vector<float_t> normFactorsSub; // normalization constant for subgraph
  bool usingSingleClass;

public:
  // TODO better constructor
  DistContext() : usingSingleClass(true){};
  ~DistContext();

  //! read labels of local nodes only
  size_t read_labels(bool isSingleClassLabel, std::string dataset_str);
  //! read features of local nodes only
  size_t read_features(std::string dataset_str);
  //! read masks of local nodes only
  size_t read_masks(std::string dataset_str, std::string mask_type, size_t n,
                    size_t& begin, size_t& end, mask_t* masks, DGraph* dGraph);

  DGraph* getGraphPointer() { return partitionedGraph; }
  Graph* getLGraphPointer() { return lGraph; }
  Graph* getSubgraphPointer(int id) { return partitionedSubgraphs[id]; };
  float_t* get_feats_ptr() { return h_feats; }
  float_t* get_feats_subg_ptr() { return h_feats_subg.data(); }
  label_t* get_labels_ptr() { return h_labels; }
  label_t* get_labels_subg_ptr() { return h_labels_subg.data(); }

  void initializeSyncSubstrate();
#ifndef __GALOIS_HET_CUDA__
  void saveDistGraph(DGraph* a);
  galois::graphs::GluonSubstrate<DGraph>* getSyncSubstrate();
#endif

  //! allocate the norm factor vector
  void allocNormFactor();
  void allocNormFactorSub(int subID);
  //! construct norm factor vector by using data from global graph
  void constructNormFactor(deepgalois::Context* globalContext);
  void constructNormFactorSub(int subgraphID);

  void constructSubgraphLabels(size_t m, const mask_t* masks);
  void constructSubgraphFeatures(size_t m, const mask_t* masks);

  float_t* get_norm_factors_ptr() { return normFactors.data(); }
  float_t* get_norm_factors_subg_ptr() { return &normFactorsSub[0]; }

  //! return label for some node
  //! NOTE: this is LID, not GID
  label_t get_label(size_t i) { return h_labels[i]; }

  //! returns pointer to the features of each local node
  float_t* get_in_ptr();

  //! allocate memory for subgraphs (don't actually build them)
  void allocateSubgraphs(int num_subgraphs);
};

} // namespace deepgalois

#endif
