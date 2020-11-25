#pragma once
#include "galois/GNNTypes.h"

namespace galois {
namespace graphs {

//! Class to hold everything allocated on the GPU that has to do with GNNGraph.
//! Similar in nature to the CUDAContext class in existing D-IrGL
class GNNGraphGPUAllocations {
public:
  //! CUDA frees all allocated memory (i.e. non-nullptr)
  ~GNNGraphGPUAllocations();
  //! Copies graph topology over to GPU; using ints because cuSparse lib
  //! expects ints for the CSR arrays
  void SetGraphTopology(const std::vector<int>& edge_index,
                        const std::vector<int>& edge_dests);
  //! Host side function that allocates memory for the features on the vertices
  //! and copies them over to the GPU.
  void SetFeatures(const std::vector<GNNFeature>& features,
                   unsigned num_features);
  //! Copy over ground truth for the graph to GPU
  void SetLabels(const std::vector<GNNLabel>& ground_truth);
  //! Copy over masks for the 3 sets to GPU
  void SetMasks(const std::vector<char>& train, const std::vector<char>& val,
                const std::vector<char>& test);
  //! Copy over norm factors
  void SetNormFactors(const std::vector<GNNFloat> norm_factors);

  GNNFeature* feature_vector() const { return feature_vector_; };
  int* edge_index() const { return edge_index_; }
  int* edge_destinations() const { return edge_destinations_; }
  GNNLabel* ground_truth() const { return ground_truth_; }
  char* local_training_mask() const { return local_training_mask_; }
  char* local_validation_mask() const { return local_validation_mask_; }
  char* local_testing_mask() const { return local_testing_mask_; }
  GNNFloat* norm_factors() const { return norm_factors_; }

private:
  // ALL THESE VARIABLES ARE DEVICE SIDE (GPU) POINTERS

  //! Number of features (which is equivalent to number of nodes)
  unsigned* num_features_{nullptr};
  //! Length of a feature vector
  unsigned* feature_length_{nullptr};
  //! Number of edges in graph
  unsigned* num_edges_{nullptr};

  // Note: no graph object, similar to Xuhao's LGraph in older code
  //! edge_index[n] gets the first edge index for node n (i.e. edge_index_[0]
  //! = 0)
  int* edge_index_{nullptr};
  //! edge_destinations_[i] = destination for edge i
  int* edge_destinations_{nullptr};
  //! (Local) feature vector
  GNNFeature* feature_vector_{nullptr};
  //! (Local) ground truth vector
  GNNLabel* ground_truth_{nullptr};
  // masks for phases
  char* local_training_mask_{nullptr};
  char* local_validation_mask_{nullptr};
  char* local_testing_mask_{nullptr};
  //! Norm factors used during aggregation
  GNNFloat* norm_factors_;
};

} // namespace graphs
} // namespace galois
