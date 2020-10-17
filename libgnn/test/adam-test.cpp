//! @file adam-test.cpp
//! Tests the adam optimizer
#include "galois/DistGalois.h"
#include "galois/GNNOptimizers.h"
#include "galois/Logging.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);
  GALOIS_LOG_VERBOSE("[{}] Using {} threads",
                     galois::runtime::getSystemNetworkInterface().ID,
                     num_threads);

  // create sample config that is easy to trace
  galois::AdamOptimizer::AdamConfiguration config;
  config.alpha   = 1;
  config.beta1   = 0.5;
  config.beta2   = 0.5;
  config.epsilon = 0;

  std::vector<size_t> layer_sizes = {2, 1};
  galois::AdamOptimizer adam(config, layer_sizes, 2);

  std::vector<galois::GNNFloat> weights1 = {1, 1};
  std::vector<galois::GNNFloat> weights2 = {10};
  std::vector<galois::GNNFloat> grad1    = {1, 1};
  std::vector<galois::GNNFloat> grad2    = {10};

  adam.GradientDescent(grad1, &weights1, 0);
  // check weights
  GALOIS_LOG_ASSERT(weights1[0] == 0.0);
  GALOIS_LOG_ASSERT(weights1[1] == 0.0);

  adam.GradientDescent(grad2, &weights2, 1);
  GALOIS_LOG_ASSERT(weights2[0] == 9.0);

  // run again to check if adam keeps moments from before
  adam.GradientDescent(grad1, &weights1, 0);
  // check weights again (turns out derivative one ends up doing same thing)
  GALOIS_LOG_ASSERT(weights1[0] == -1.0);
  GALOIS_LOG_ASSERT(weights1[1] == -1.0);

  // grad 2 again
  adam.GradientDescent(grad2, &weights2, 1);
  GALOIS_LOG_ASSERT(weights2[0] == 8.0);
}
