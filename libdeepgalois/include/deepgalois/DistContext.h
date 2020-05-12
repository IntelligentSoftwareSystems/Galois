#ifndef __DG_DIST_CONTEXT__
#define __DG_DIST_CONTEXT__
#ifdef __GALOIS_HET_CUDA__
#include "deepgalois/cutils.h"
#else
#include "galois/graphs/GluonSubstrate.h"
#endif

#include "deepgalois/types.h"
#include "deepgalois/Context.h"
#include "deepgalois/GraphTypes.h"
#include "deepgalois/reader.h"

namespace deepgalois {

class DistContext {
  bool is_device;         // is this on device or host
  bool is_selfloop_added; // whether selfloop is added to the input graph
  bool usingSingleClass;
  std::string dataset;
  size_t num_classes;     // number of classes: E
  size_t feat_len;        // input feature length: D
  Graph* lGraph;          // laerning graph version
#ifdef __GALOIS_HET_CUDA__
  label_t* d_labels;      // labels on device
  label_t* d_labels_subg; // labels for subgraph on device
  float_t* d_feats;       // input features on device
  float_t* d_feats_subg;  // input features for subgraph on device
#else
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

  Reader reader;

public:
  // TODO better constructor
  DistContext();
  DistContext(bool isDevice) : is_device(isDevice) {}
  ~DistContext();

  size_t read_graph(std::string dataset_str, bool selfloop = false);

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

  void initializeSyncSubstrate();
#ifdef __GALOIS_HET_CUDA__
  float_t* get_feats_ptr() { return d_feats; }
  float_t* get_feats_subg_ptr() { return d_feats_subg; }
  label_t* get_labels_ptr() { return d_labels; }
  label_t* get_labels_subg_ptr() { return d_labels_subg; }
  void copy_data_to_device(); // copy labels and input features
  static cublasHandle_t cublas_handle_;         // used to call cuBLAS
  static cusparseHandle_t cusparse_handle_;     // used to call cuSPARSE
  static cusparseMatDescr_t cusparse_matdescr_; // used to call cuSPARSE
  static curandGenerator_t curand_generator_; // used to generate random numbers on GPU
  inline static cublasHandle_t cublas_handle() { return cublas_handle_; }
  inline static cusparseHandle_t cusparse_handle() { return cusparse_handle_; }
  inline static cusparseMatDescr_t cusparse_matdescr() { return cusparse_matdescr_; }
  inline static curandGenerator_t curand_generator() { return curand_generator_; }
#else
  void saveDistGraph(DGraph* a);
  galois::graphs::GluonSubstrate<DGraph>* getSyncSubstrate();
  float_t* get_feats_ptr() { return h_feats; }
  float_t* get_feats_subg_ptr() { return h_feats_subg.data(); }
  label_t* get_labels_ptr() { return h_labels; }
  label_t* get_labels_subg_ptr() { return h_labels_subg.data(); }
#endif

  void set_dataset(std::string dataset_str) {
    dataset = dataset_str;
    reader.init(dataset);
  }

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
