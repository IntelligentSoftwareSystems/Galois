//! @file softmaxlayer-test.cpp
//! Softmax layer test with a test graph
//! No automated ground truth checking; when this was written it was compared
//! manually with pytorch
//! TODO add in automated checking eventually; for now this just makes sure it
//! runs

#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/SoftmaxLayer.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);
  GALOIS_LOG_VERBOSE("Num threads is {}", num_threads);

  // load test graph
  galois::graphs::GNNGraph<char, void> test_graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true, false);

  // input/output columns must be same in softmax
  galois::GNNLayerDimensions dimension_0;
  dimension_0.input_rows     = 7;
  dimension_0.input_columns  = test_graph.GetNumLabelClasses();
  dimension_0.output_columns = test_graph.GetNumLabelClasses();

  GALOIS_LOG_VERBOSE("Num output classes is {}", dimension_0.input_columns);

  // input to softmax
  std::vector<galois::GNNFloat> softmax_input(49, 0.0);
  // create input with perfect accuracy
  softmax_input[0]  = 1;
  softmax_input[8]  = 1;
  softmax_input[16] = 1;
  softmax_input[24] = 1;
  softmax_input[32] = 1;
  softmax_input[40] = 1;
  softmax_input[48] = 1;

  std::vector<galois::GNNFloat> back_matrix(49);
  galois::PointerWithSize<galois::GNNFloat> p_back(back_matrix);

  // train mode
  auto output_layer = std::make_unique<galois::SoftmaxLayer<char, void>>(
      3, test_graph, &p_back, dimension_0);
  galois::PointerWithSize<galois::GNNFloat> prediction_distribution =
      output_layer->ForwardPhase(softmax_input);

  // assert that predictions are as expected
  for (size_t i = 0; i < 5; i++) {
    GALOIS_LOG_ASSERT(galois::MaxIndex(7, &(prediction_distribution[i * 7])) ==
                      i);
  }
  // train mode means last 2 vertices should be empty
  for (size_t i = 5; i < 7; i++) {
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 0] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 1] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 2] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 3] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 4] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 5] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 6] == 0.0);
  }

  // NOTE: checked before backward because backward overwrites this matrix

  galois::PointerWithSize<galois::GNNFloat> asdf =
      output_layer->BackwardPhase(softmax_input, nullptr);
  printf("Output 1\n========\n");
  for (unsigned i = 0; i < asdf.size(); i++) {
    printf("%f\n", asdf[i]);
  }

  // validation mode
  output_layer->SetLayerPhase(galois::GNNPhase::kValidate);
  galois::PointerWithSize<galois::GNNFloat> pd2 =
      output_layer->ForwardPhase(softmax_input);

  // validate vertex is index 5
  GALOIS_LOG_ASSERT(galois::MaxIndex(7, &(pd2[5 * 7])) == 5);
  for (size_t i = 0; i < 5; i++) {
    GALOIS_LOG_ASSERT(pd2[i * 7 + 0] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 1] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 2] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 3] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 4] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 5] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 6] == 0.0);
  }
  for (size_t i = 6; i < 7; i++) {
    GALOIS_LOG_ASSERT(pd2[i * 7 + 0] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 1] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 2] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 3] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 4] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 5] == 0.0);
    GALOIS_LOG_ASSERT(pd2[i * 7 + 6] == 0.0);
  }

  asdf = output_layer->BackwardPhase(softmax_input, nullptr);
  printf("Output 2\n========\n");
  for (unsigned i = 0; i < asdf.size(); i++) {
    printf("%f\n", asdf[i]);
  }

  // test mode
  output_layer->SetLayerPhase(galois::GNNPhase::kTest);
  galois::PointerWithSize<galois::GNNFloat> pd3 =
      output_layer->ForwardPhase(softmax_input);
  // validate vertex is index 6
  GALOIS_LOG_ASSERT(galois::MaxIndex(7, &(pd3[6 * 7])) == 6);
  // all but last are empty distributions
  for (size_t i = 0; i < 6; i++) {
    GALOIS_LOG_ASSERT(pd3[i * 7 + 0] == 0.0);
    GALOIS_LOG_ASSERT(pd3[i * 7 + 1] == 0.0);
    GALOIS_LOG_ASSERT(pd3[i * 7 + 2] == 0.0);
    GALOIS_LOG_ASSERT(pd3[i * 7 + 3] == 0.0);
    GALOIS_LOG_ASSERT(pd3[i * 7 + 4] == 0.0);
    GALOIS_LOG_ASSERT(pd3[i * 7 + 5] == 0.0);
    GALOIS_LOG_ASSERT(pd3[i * 7 + 6] == 0.0);
  }

  asdf = output_layer->BackwardPhase(softmax_input, nullptr);
  printf("Output 3\n========\n");
  for (unsigned i = 0; i < asdf.size(); i++) {
    printf("%f\n", asdf[i]);
  }
}
