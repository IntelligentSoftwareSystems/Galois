#ifndef GALOIS_GPU_GNN_OPT
#define GALOIS_GPU_GNN_OPT

#include <vector>
#include "galois/GNNTypes.h"

namespace galois {

//! Holds GPU memory for the adam optimizer as well as function definitions
//! for weight adjustment
class AdamOptimizerGPU {
public:
  //! Initializes the moment vectors on the GPU based on provided sizes
  AdamOptimizerGPU(const std::vector<size_t>& trainable_layer_sizes,
                   size_t num_trainable);
  //! Frees moment vectors and vector of pointers to moments
  ~AdamOptimizerGPU();

  GNNFloat* first_moment(size_t i) { return first_moments_[i]; };
  GNNFloat* second_moment(size_t i) { return second_moments_[i]; };

private:
  size_t num_layers_;
  std::vector<GNNFloat*> first_moments_;
  std::vector<GNNFloat*> second_moments_;
};

} // namespace galois

#endif
