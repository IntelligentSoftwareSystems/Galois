#include "galois/Galois.h"
#include "galois/GNNOptimizers.h"
#include "galois/Logging.h"
#include <cassert>

void galois::AdamOptimizer::GradientDescent(
    PointerWithSize<GNNFloat> derivatives, PointerWithSize<GNNFloat> matrix,
    size_t layer_number) {
  assert(derivatives.size() == matrix.size());

  // grab based on layer being used
  PointerWithSize<GNNFloat>& first_moment  = p_first_moments_[layer_number];
  PointerWithSize<GNNFloat>& second_moment = p_second_moments_[layer_number];
  assert(derivatives.size() == first_moment.size());
  assert(derivatives.size() == second_moment.size());

#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    gpu_object_.AdamUpdate(derivatives.data(), matrix.data(), matrix.size(),
                           first_moment.data(), second_moment.data(),
                           config_.alpha, config_.beta1, config_.beta2,
                           config_.epsilon, beta1_power_t_[layer_number],
                           beta2_power_t_[layer_number]);
  } else {
#endif
    // individual weight updates via gradients
    galois::do_all(
        galois::iterate(static_cast<size_t>(0), matrix.size()),
        [&](size_t i) {
          // moment estimate updates
          first_moment[i] = config_.beta1 * first_moment[i] +
                            (1.0 - config_.beta1) * derivatives[i];
          second_moment[i] =
              config_.beta2 * second_moment[i] +
              (1.0 - config_.beta2) * (derivatives[i] * derivatives[i]);
          // bias corrected moments using beta power
          GNNFloat bias_correct_first =
              first_moment[i] / (1.0 - beta1_power_t_[layer_number]);
          GNNFloat bias_correct_second =
              second_moment[i] / (1.0 - beta2_power_t_[layer_number]);
          // weight update using bias corrected moments
          (matrix.data())[i] -=
              config_.alpha * bias_correct_first /
              std::sqrt(bias_correct_second + config_.epsilon);
        },
        galois::loopname("AdamOptimizerGradientDescent"));
#ifdef GALOIS_ENABLE_GPU
  }
#endif

  // update the power terms for next update call
  beta1_power_t_[layer_number] *= config_.beta1;
  beta2_power_t_[layer_number] *= config_.beta2;
}
