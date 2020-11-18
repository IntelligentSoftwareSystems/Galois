#pragma once
// Code inspired from this; actual code style is not the same + changed some
// things such as adding params for every layer which TinyDNN does not seem to
// do
// https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/optimizers/optimizer.h
// Copyright (c) 2013, Taiga Nomi and the respective contributors
// All rights reserved.
// Changed by Galois under 3-BSD
#include "galois/GNNTypes.h"
#include <vector>
#include <cassert>

#ifdef GALOIS_ENABLE_GPU
#include "galois/GNNOptimizers.cuh"
#endif

namespace galois {

//! Virtual class; optimizers all need the descent function
class BaseOptimizer {
public:
  virtual void GradientDescent(PointerWithSize<GNNFloat> derivatives,
                               PointerWithSize<GNNFloat> matrix,
                               size_t layer_number) = 0;
};

//! Maintains a first and second moment for each weight in the weight matrix and
//! does gradient descent invidiually on each weight
class AdamOptimizer : public BaseOptimizer {
public:
  //! Struct for specifying adam config. Defaults based on the Adam paper.
  struct AdamConfiguration {
    GNNFloat alpha{0.01};
    GNNFloat beta1{0.9};
    GNNFloat beta2{0.999};
    GNNFloat epsilon{1e-8};
  };

  AdamOptimizer(const std::vector<size_t>& trainable_layer_sizes,
                size_t num_trainable_layers)
      : AdamOptimizer(AdamConfiguration(), trainable_layer_sizes,
                      num_trainable_layers) {}

  //! Constructor allocates memory, initializes training vars for each layer
  AdamOptimizer(const AdamConfiguration& config,
                const std::vector<size_t>& trainable_layer_sizes,
                size_t num_trainable_layers)
      :
#ifdef GALOIS_ENABLE_GPU
        gpu_object_(trainable_layer_sizes, num_trainable_layers),
#endif
        config_(config), num_trainable_layers_(num_trainable_layers),
        beta1_power_t_(num_trainable_layers_, config.beta1),
        beta2_power_t_(num_trainable_layers_, config.beta2) {
    // >= because only prefix will be considered otherwise
    assert(trainable_layer_sizes.size() >= num_trainable_layers_);
#ifndef GALOIS_ENABLE_GPU
    // allocate vectors based on # of trainable layers
    for (size_t i = 0; i < num_trainable_layers_; i++) {
      first_moments_.emplace_back(trainable_layer_sizes[i], 0.0);
      second_moments_.emplace_back(trainable_layer_sizes[i], 0.0);
      // Pointer with size construction
      p_first_moments_.emplace_back(first_moments_.back());
      p_second_moments_.emplace_back(second_moments_.back());
    }
    assert(first_moments_.size() == num_trainable_layers_);
    assert(second_moments_.size() == num_trainable_layers_);
#else
    // pointer with size initialization with GPU pointers
    for (size_t i = 0; i < num_trainable_layers_; i++) {
      p_first_moments_.emplace_back(gpu_object_.first_moment(i),
                                    trainable_layer_sizes[i]);
      p_second_moments_.emplace_back(gpu_object_.second_moment(i),
                                     trainable_layer_sizes[i]);
    }
#endif
  }
  //! Adam based gradient descent
  void GradientDescent(PointerWithSize<GNNFloat> derivatives,
                       PointerWithSize<GNNFloat> matrix,
                       size_t layer_number) final;

private:
#ifdef GALOIS_ENABLE_GPU
  AdamOptimizerGPU gpu_object_;
#endif

  //! Configuration options for this layer
  AdamConfiguration config_;
  //! First moment vectors; one for each trainable layer
  std::vector<std::vector<GNNFloat>> first_moments_;
  //! Second moment vectors; one for each trainable layer
  std::vector<std::vector<GNNFloat>> second_moments_;
  // PointerWithSize versions of first/second moments (for use in function
  // to support GPU pointers as well
  std::vector<PointerWithSize<GNNFloat>> p_first_moments_;
  std::vector<PointerWithSize<GNNFloat>> p_second_moments_;
  //! Number of layers that can be trained (need moment vectors for each)
  size_t num_trainable_layers_;
  // power terms used in adam: updated by raising power every time update is
  // called
  // vector because one is necessary for each layer
  std::vector<GNNFloat> beta1_power_t_;
  std::vector<GNNFloat> beta2_power_t_;
};

} // namespace galois
