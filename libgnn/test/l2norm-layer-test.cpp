#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/L2NormLayer.h"

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
  dimension_0.input_columns  = 2;
  dimension_0.output_columns = 2;

  std::vector<galois::GNNFloat> l2_input(14, 0.0);
  l2_input[0]  = 4;
  l2_input[1]  = 3;
  l2_input[2]  = 4;
  l2_input[3]  = 3;
  l2_input[4]  = 4;
  l2_input[5]  = 3;
  l2_input[6]  = 4;
  l2_input[7]  = 3;
  l2_input[8]  = 4;
  l2_input[9]  = 3;
  l2_input[10] = 4;
  l2_input[11] = 3;
  l2_input[12] = 4;
  l2_input[13] = 3;

  std::vector<galois::GNNFloat> back_matrix(14);
  galois::PointerWithSize<galois::GNNFloat> p_back(back_matrix);

  auto l2_layer = std::make_unique<galois::L2NormLayer<char, void>>(
      2, test_graph, &p_back, dimension_0);
  galois::PointerWithSize<galois::GNNFloat> normed =
      l2_layer->ForwardPhase(l2_input);

  // only go up to 5 because training set
  for (size_t row = 0; row < 5; row++) {
    GALOIS_LOG_VASSERT(std::abs(normed[row * 2] - 0.8) < 0.0001,
                       "input 4 should become 0.8 not {}, index {}",
                       normed[row * 2], row * 2);
    GALOIS_LOG_VASSERT(std::abs(normed[row * 2 + 1] - 0.6) < 0.0001,
                       "input 3 should become 0.6 not {}, index {}",
                       normed[row * 2 + 1], row * 2 + 1);
  }
  // only go up to 5 because training set
  for (size_t row = 5; row < 7; row++) {
    GALOIS_LOG_VASSERT(std::abs(normed[row * 2] - 0.0) < 0.0001,
                       "index {} should be 0, not part of train", row * 2);
    GALOIS_LOG_VASSERT(std::abs(normed[row * 2 + 1] - 0.0) < 0.0001,
                       "index {} should be 0, not part of train", row * 2 + 1);
  }

  // backward
  std::vector<galois::GNNFloat> dummy_ones_v(14, 1);
  galois::PointerWithSize dummy_ones(dummy_ones_v);

  galois::PointerWithSize<galois::GNNFloat> grads =
      l2_layer->BackwardPhase(l2_input, &dummy_ones);
  float out_4 = (-3.0 / 125.0);
  float out_3 = (4.0 / 125.0);
  for (size_t row = 0; row < 5; row++) {
    GALOIS_LOG_VASSERT(std::abs(grads[row * 2] - out_4) < 0.0001,
                       "index {} grad 4 gradient should be {} not {}", row * 2,
                       out_4, grads[row * 2]);
    GALOIS_LOG_VASSERT(std::abs(grads[row * 2 + 1] - out_3) < 0.0001,
                       "index {} grad 3 gradient should be {} not {}",
                       row * 2 + 1, out_3, grads[row * 2 + 1]);
  }

  for (size_t row = 5; row < 7; row++) {
    GALOIS_LOG_VASSERT(std::abs(grads[row * 2] - 0.0) < 0.0001,
                       "index {} should be 0, not part of train", row * 2);
    GALOIS_LOG_VASSERT(std::abs(grads[row * 2 + 1] - 0.0) < 0.0001,
                       "index {} should be 0, not part of train", row * 2 + 1);
  }

  return 0;
}
