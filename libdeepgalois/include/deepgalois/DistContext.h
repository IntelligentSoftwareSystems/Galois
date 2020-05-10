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

  Graph* lGraph; // laerning graph version
  DGraph* partitionedGraph; // the input graph, |V| = N
  std::vector<Graph*> partitionedSubgraphs;
  label_t* h_labels;      // labels for classification. Single-class label: Nx1,
                          // multi-class label: NxE
  std::vector<label_t> h_labels_subg; // labels for subgraph
  float_t* h_feats;       // input features: N x D
  std::vector<float_t> h_feats_subg;  // input features for subgraph

  //  change regular one to a vector as well
  float_t* normFactors;  // normalization constant based on graph structure
  std::vector<float_t> normFactorsSub; // normalization constant for subgraph

public:
  DistContext();
  ~DistContext();

  void saveDistGraph(DGraph* a) {
    partitionedGraph = a;

    // construct lgraph from underlying lc csr graph
    this->lGraph = new Graph();
    this->lGraph->allocateFrom(a->size(), a->sizeEdges());
    this->lGraph->constructNodes();

    galois::do_all(
        galois::iterate((size_t)0, a->size()),
        [&](const auto src) {
          this->lGraph->fixEndEdge(src, *a->edge_end(src));
          index_t idx = *(a->edge_begin(src));

          for (auto e = a->edge_begin(src); e != a->edge_end(src); e++) {
            const auto dst = a->getEdgeDst(e);
            this->lGraph->constructEdge(idx++, dst, 0);
          }
        },
        galois::loopname("lgraphcopy")
    );
  }

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

  DGraph* getGraphPointer() { return partitionedGraph; }
  Graph* getLGraphPointer() { return lGraph; }

  Graph* getSubgraphPointer(int id) { return partitionedSubgraphs[id]; };
  float_t* get_feats_ptr() { return h_feats; }
  float_t* get_feats_subg_ptr() { return h_feats_subg.data(); }
  label_t* get_labels_ptr() { return h_labels; }
  label_t* get_labels_subg_ptr() { return h_labels_subg.data(); }

  void initializeSyncSubstrate();
  galois::graphs::GluonSubstrate<DGraph>* getSyncSubstrate();

  //! allocate the norm factor vector
  void allocNormFactor();
  void allocNormFactorSub(int subID);
  //! construct norm factor vector by using data from global graph
  void constructNormFactor(deepgalois::Context* globalContext);
  void constructNormFactorSub(int subgraphID);

  void constructSubgraphLabels(size_t m, const mask_t* masks);
  void constructSubgraphFeatures(size_t m, const mask_t* masks);

  float_t* get_norm_factors_ptr() { return normFactors; }
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
