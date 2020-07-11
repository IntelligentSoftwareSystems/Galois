#ifndef __DG_DIST_CONTEXT__
#define __DG_DIST_CONTEXT__
#ifdef GALOIS_ENABLE_GPU
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
  size_t num_classes;       // number of classes: E
  size_t feat_len;          // input feature length: D
  Graph* lGraph;            // learning graph version
  DGraph* partitionedGraph; // the input graph, |V| = N
  std::vector<Graph*> partitionedSubgraphs;
  label_t* h_labels; // labels for classification. Single-class: Nx1,
                     // multi-class: NxE
  float_t* h_feats;  // input features: N x D
#ifdef GALOIS_ENABLE_GPU
  label_t* d_labels;      // labels on device
  label_t* d_labels_subg; // labels for subgraph on device
  float_t* d_feats;       // input features on device
  float_t* d_feats_subg;  // input features for subgraph on device
  float_t* d_normFactors;
  float_t* d_normFactorsSub;
#else
  galois::graphs::GluonSubstrate<DGraph>* syncSubstrate;
#endif
  std::vector<label_t> h_labels_subg; // labels for subgraph
  std::vector<float_t> h_feats_subg;  // input features for subgraph
  std::vector<float_t>
      normFactors; // normalization constant based on graph structure
  std::vector<float_t> normFactorsSub; // normalization constant for subgraph

  Reader reader;

public:
  // TODO better constructor
  DistContext();
  DistContext(bool isDevice)
      : is_device(isDevice), is_selfloop_added(false), usingSingleClass(true),
        dataset(""), num_classes(0), feat_len(0), lGraph(NULL),
        partitionedGraph(NULL), h_labels(0), h_feats(0) {}
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
#ifdef GALOIS_ENABLE_GPU
  float_t* get_feats_ptr() { return d_feats; }
  float_t* get_feats_subg_ptr() { return d_feats_subg; }
  label_t* get_labels_ptr() { return d_labels; }
  label_t* get_labels_subg_ptr() { return d_labels_subg; }
  float_t* get_norm_factors_ptr() { return d_normFactors; }
  float_t* get_norm_factors_subg_ptr() { return d_normFactorsSub; }
  void copy_data_to_device();               // copy labels and input features
  static cublasHandle_t cublas_handle_;     // used to call cuBLAS
  static cusparseHandle_t cusparse_handle_; // used to call cuSPARSE
  static cusparseMatDescr_t cusparse_matdescr_; // used to call cuSPARSE
  static curandGenerator_t
      curand_generator_; // used to generate random numbers on GPU
  inline static cublasHandle_t cublas_handle() { return cublas_handle_; }
  inline static cusparseHandle_t cusparse_handle() { return cusparse_handle_; }
  inline static cusparseMatDescr_t cusparse_matdescr() {
    return cusparse_matdescr_;
  }
  inline static curandGenerator_t curand_generator() {
    return curand_generator_;
  }
#else
  void saveDistGraph(DGraph* a);
  galois::graphs::GluonSubstrate<DGraph>* getSyncSubstrate();
  float_t* get_feats_ptr() { return h_feats; }
  float_t* get_feats_subg_ptr() { return h_feats_subg.data(); }
  label_t* get_labels_ptr() { return h_labels; }
  label_t* get_labels_subg_ptr() { return h_labels_subg.data(); }
  float_t* get_norm_factors_ptr() { return normFactors.data(); }
  float_t* get_norm_factors_subg_ptr() { return &normFactorsSub[0]; }
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

  //! return label for some node
  //! NOTE: this is LID, not GID
  label_t get_label(size_t lid) { return h_labels[lid]; }

  //! returns pointer to the features of each local node
  float_t* get_in_ptr();

  //! allocate memory for subgraphs (don't actually build them)
  void allocateSubgraphs(int num_subgraphs, unsigned max_size);

  //! return if a vertex is owned by the partitioned graph this context contains
  bool isOwned(unsigned gid);
  //! return if part graph has provided vertex for given gid locally
  bool isLocal(unsigned gid);
  //! get GID of an lid for a vertex
  unsigned getGID(unsigned lid);
  //! get local id of a vertex given a global id for that vertex
  unsigned getLID(unsigned gid);
};

} // namespace deepgalois

#endif
