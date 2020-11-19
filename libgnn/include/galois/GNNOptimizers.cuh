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

  //! Calls into a GPU kernel; needs to be done this way as this cuh is included
  //! in a GCC build, so the kernel cannot be defined in this header.
  void AdamUpdate(const GNNFloat* derivatives, GNNFloat* matrix_to_update,
                  size_t matrix_size, GNNFloat* first_moment,
                  GNNFloat* second_moment, GNNFloat alpha, GNNFloat beta1,
                  GNNFloat beta2, GNNFloat epsilon, GNNFloat beta1t,
                  GNNFloat beta2t);

  //! Helper to copy gpu pointer to cpu vector
  void CopyToVector(std::vector<GNNFloat>& to, PointerWithSize<GNNFloat> from);

private:
  size_t num_layers_;
  std::vector<GNNFloat*> first_moments_;
  std::vector<GNNFloat*> second_moments_;
};

} // namespace galois

#endif
