#pragma once
#include "galois/GNNTypes.h"

namespace galois {
namespace graphs {

//! Class to hold everything allocated on the GPU that has to do with GNNGraph.
//! Similar in nature to the CUDAContext class in existing D-IrGL
class GNNGraphGPUAllocations {
public:
  // XXX getters for everything, the rest of the setters, etc.

  // XXX destructor for allocated memory

  //! Host side function that allocates memory for the features on the vertices
  //! and copies them over to the GPU.
  void SetFeatures(const std::vector<GNNFeature>& features);

private:
  // Note: no graph object, similar to Xuhao's LGraph in older code
  //! edge_index[n] gets the first edge index for node n (i.e. edge_index_[0]
  //! = 0)
  GPUEdgeIndex* edge_index_{nullptr};
  //! edge_destinations_[i] = destination for edge i
  GPUNodeIndex* edge_destinations_{nullptr};
  //! (Local) feature vector
  GNNFeature* feature_vector_{nullptr};
  //! (Local) ground truth vector
  GNNFloat* ground_truth_{nullptr};
  //! (Local) norm factors
  GNNFloat* norm_factors_{nullptr};

  // XXX masks? other things I haven't considered yet?
};

} // namespace graphs
} // namespace galois
