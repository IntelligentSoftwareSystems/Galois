//! @file adam-test.cpp
//! Tests the adam optimizer
#include "galois/DistGalois.h"
#include "galois/GNNOptimizers.h"
#include "galois/Logging.h"
#include "galois/layers/SoftmaxLayer.h"

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

  // make this layer to get access to a gpu helper function; TODO
  // need a helper alloc function
  galois::graphs::GNNGraph test_graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true);
  galois::GNNLayerDimensions dimension_0;
  dimension_0.input_rows     = 7;
  dimension_0.input_columns  = test_graph.GetNumLabelClasses();
  dimension_0.output_columns = test_graph.GetNumLabelClasses();
  auto alloc_layer =
      std::make_unique<galois::SoftmaxLayer>(3, test_graph, dimension_0);

  std::vector<galois::GNNFloat> weights1 = {1, 1};
  std::vector<galois::GNNFloat> weights2 = {10};
  std::vector<galois::GNNFloat> grad1    = {1, 1};
  std::vector<galois::GNNFloat> grad2    = {10};

  galois::PointerWithSize<galois::GNNFloat> p_grad1 =
      alloc_layer->AllocateGPU(grad1);
  galois::PointerWithSize<galois::GNNFloat> p_weights1 =
      alloc_layer->AllocateGPU(weights1);
  galois::PointerWithSize<galois::GNNFloat> p_grad2 =
      alloc_layer->AllocateGPU(grad2);
  galois::PointerWithSize<galois::GNNFloat> p_weights2 =
      alloc_layer->AllocateGPU(weights2);

  adam.GradientDescent(p_grad1, p_weights1, 0);
  adam.CopyToVector(weights1, p_weights1);

  // check weights
  GALOIS_LOG_ASSERT(weights1[0] == 0.0);
  GALOIS_LOG_ASSERT(weights1[1] == 0.0);

  adam.GradientDescent(p_grad2, p_weights2, 1);
  adam.CopyToVector(weights2, p_weights2);
  GALOIS_LOG_ASSERT(weights2[0] == 9.0);

  // run again to check if adam keeps moments from before
  adam.GradientDescent(p_grad1, p_weights1, 0);
  adam.CopyToVector(weights1, p_weights1);
  // check weights again (turns out derivative one ends up doing same thing)
  GALOIS_LOG_ASSERT(weights1[0] == -1.0);
  GALOIS_LOG_ASSERT(weights1[1] == -1.0);

  // grad 2 again
  adam.GradientDescent(p_grad2, p_weights2, 1);
  adam.CopyToVector(weights2, p_weights2);
  GALOIS_LOG_ASSERT(weights2[0] == 8.0);
}
