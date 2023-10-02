#include "galois/Logging.h"
#include "galois/layers/GraphConvolutionalLayer.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);

  GALOIS_LOG_VERBOSE("[{}] Using {} threads",
                     galois::runtime::getSystemNetworkInterface().ID,
                     num_threads);
  // load test graph
  galois::graphs::GNNGraph<char, void> test_graph(
      "tester", galois::graphs::GNNPartitionScheme::kCVC, true, false);
  galois::PointerWithSize<galois::GNNFloat> feats =
      test_graph.GetLocalFeatures();
  for (size_t row = 0; row < test_graph.size(); row++) {
    // row -> GID
    size_t global_row             = test_graph.GetGID(row);
    galois::GNNFloat ground_truth = 0.0;

    switch (global_row) {
    case 0:
      ground_truth = 0;
      break;
    case 1:
      ground_truth = 1;
      break;
    case 2:
      ground_truth = 2;
      break;
    case 3:
      ground_truth = 3;
      break;
    case 4:
      ground_truth = 4;
      break;
    case 5:
      ground_truth = 5;
      break;
    case 6:
      ground_truth = 6;
      break;
    default:
      GALOIS_LOG_FATAL("bad global row for test graph");
      break;
    }
    // size 2 columns
    for (size_t c = 0; c < 3; c++) {
      GALOIS_LOG_VASSERT(feats[row * 3 + c] == ground_truth, "{} not {}",
                         ground_truth, feats[row * 2 + c]);
    }
  }

  galois::GNNLayerDimensions dimension_0;
  dimension_0.input_rows     = test_graph.size();
  dimension_0.input_columns  = 3;
  dimension_0.output_columns = 2;

  galois::GNNLayerConfig dcon;
  dcon.DebugConfig();
  dcon.disable_aggregate_after_update = true;

  // dummy 1 matrix
  std::vector<galois::GNNFloat> dummy_ones_v(test_graph.size() * 2, 1);
  galois::PointerWithSize dummy_ones(dummy_ones_v);

  std::vector<galois::GNNFloat> back_matrix(test_graph.size() * 3);
  galois::PointerWithSize<galois::GNNFloat> p_back(back_matrix);

  // create layer 1 for testing backward prop actually giving weights back
  std::unique_ptr<galois::GraphConvolutionalLayer<char, void>> layer_1 =
      std::make_unique<galois::GraphConvolutionalLayer<char, void>>(
          1, test_graph, &p_back, dimension_0, dcon);
  layer_1->InitAllWeightsTo1();
  galois::PointerWithSize<galois::GNNFloat> layer_1_forward_output =
      layer_1->ForwardPhase(test_graph.GetLocalFeatures());

  for (size_t row = 0; row < test_graph.size(); row++) {
    // row -> GID
    size_t global_row             = test_graph.GetGID(row);
    galois::GNNFloat ground_truth = 0.0;

    switch (global_row) {
    case 0:
      ground_truth = 3;
      break;
    case 1:
      ground_truth = 6;
      break;
    case 2:
      ground_truth = 12;
      break;
    case 3:
      ground_truth = 18;
      break;
    case 4:
      ground_truth = 24;
      break;
    case 5:
      ground_truth = 30;
      break;
    case 6:
      ground_truth = 15;
      break;
    default:
      GALOIS_LOG_FATAL("bad global row for test graph");
      break;
    }
    // size 2 columns
    for (size_t c = 0; c < 2; c++) {
      GALOIS_LOG_VASSERT(layer_1_forward_output[row * 2 + c] == ground_truth,
                         "{} not {}", ground_truth,
                         layer_1_forward_output[row * 2 + c]);
    }
  }

  galois::PointerWithSize<galois::GNNFloat> layer_1_backward_output =
      layer_1->BackwardPhase(test_graph.GetLocalFeatures(), &dummy_ones);

  for (size_t row = 0; row < test_graph.size(); row++) {
    // row -> GID
    size_t global_row             = test_graph.GetGID(row);
    galois::GNNFloat ground_truth = 0.0;

    switch (global_row) {
    case 0:
      ground_truth = 2;
      break;
    case 1:
      ground_truth = 4;
      break;
    case 2:
      ground_truth = 4;
      break;
    case 3:
      ground_truth = 4;
      break;
    case 4:
      ground_truth = 4;
      break;
    case 5:
      ground_truth = 4;
      break;
    case 6:
      ground_truth = 2;
      break;
    default:
      GALOIS_LOG_FATAL("bad global row for test graph");
      break;
    }
    // size 2 columns
    for (size_t c = 0; c < 3; c++) {
      GALOIS_LOG_ASSERT(layer_1_backward_output[row * 3 + c] == ground_truth);
    }
  }

  galois::PointerWithSize<galois::GNNFloat> layer_1_weight_gradients =
      layer_1->GetLayerWeightGradients();

  // make sure they are sane
  GALOIS_LOG_ASSERT(layer_1_weight_gradients.size() == 6);
  GALOIS_LOG_VASSERT(layer_1_weight_gradients[0] == 36, "36 not {}",
                     layer_1_weight_gradients[0]);
  GALOIS_LOG_VASSERT(layer_1_weight_gradients[1] == 36, "36 not {}",
                     layer_1_weight_gradients[1]);
  GALOIS_LOG_VASSERT(layer_1_weight_gradients[2] == 36, "36 not {}",
                     layer_1_weight_gradients[2]);
  GALOIS_LOG_VASSERT(layer_1_weight_gradients[3] == 36, "36 not {}",
                     layer_1_weight_gradients[3]);
  GALOIS_LOG_VASSERT(layer_1_weight_gradients[4] == 36, "36 not {}",
                     layer_1_weight_gradients[4]);
  GALOIS_LOG_VASSERT(layer_1_weight_gradients[5] == 36, "36 not {}",
                     layer_1_weight_gradients[5]);

  layer_1.reset();

  return 0;
}
