//! @file sample-test.cpp
//! Sampling tester

/// TODO(hc): This test is deprecated as GCN layer now supports
/// edge sampling, as well as node sampling.
/// The previous GCN only checks if node is sampled, but
/// now it checks edge sampling and for that, it utilizes
/// a bitset to mark sampled edges.
/// If that bitset is not set, the corresponding edge is ignored.
/// However, this test currently does not consider this case,
/// and doesn't work.
/// To satisfy the previous assumption and make this test work,
/// we should mark the entire adjacent edges of the sampled nodes.
/// In this case, we should not mark the edges' destination nodes as
/// sampled nodes, and so, let src node iterator skip those nodes
/// but only allow to iterate them as outgoing destinations.
/// We can reuse this code later, and so, I will not remove this
/// from the current source tree.

#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/GraphConvolutionalLayer.h"
#include "galois/layers/SoftmaxLayer.h"
#include "galois/layers/SigmoidLayer.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);

  GALOIS_LOG_VERBOSE("[{}] Using {} threads",
                     galois::runtime::getSystemNetworkInterface().ID,
                     num_threads);
  // load test graph
  galois::graphs::GNNGraph<char, void> test_graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true, false);

  galois::GNNLayerDimensions dimension_0;
  dimension_0.input_rows     = 7;
  dimension_0.input_columns  = 3;
  dimension_0.output_columns = 2;

  galois::GNNLayerConfig dcon;
  dcon.disable_aggregate_after_update = false;
  dcon.DebugConfig();

  // choose a few sample nodes
  test_graph.SetSampledNode(0);
  test_graph.SetSampledNode(2);
  test_graph.SetSampledNode(4);
  test_graph.SetSampledNode(5);
  test_graph.UnsetSampledNode(1);
  test_graph.UnsetSampledNode(3);
  test_graph.UnsetSampledNode(6);

  //////////////////////////////////////////////////////////////////////////////

  std::vector<galois::GNNFloat> back_matrix(21);
  galois::PointerWithSize<galois::GNNFloat> p_back(back_matrix);

  std::unique_ptr<galois::GraphConvolutionalLayer<char, void>> layer_1 =
      std::make_unique<galois::GraphConvolutionalLayer<char, void>>(
          1, test_graph, &p_back, dimension_0, dcon);
  layer_1->InitAllWeightsTo1();
  layer_1->EnableSampling();

  galois::PointerWithSize<galois::GNNFloat> layer_1_forward_output =
      layer_1->ForwardPhase(test_graph.GetLocalFeatures());
  // same check as before for sanity purposes
  GALOIS_LOG_ASSERT(layer_1_forward_output.size() == 14);
  GALOIS_LOG_ASSERT(layer_1_forward_output[0] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[1] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[2] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[3] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[4] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[5] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[6] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[7] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[8] == 15);
  GALOIS_LOG_ASSERT(layer_1_forward_output[9] == 15);
  GALOIS_LOG_ASSERT(layer_1_forward_output[10] == 12);
  GALOIS_LOG_ASSERT(layer_1_forward_output[11] == 12);
  GALOIS_LOG_ASSERT(layer_1_forward_output[12] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[13] == 0);

  // dummy 1 matrix
  std::vector<galois::GNNFloat> dummy_ones_v(14, 1);
  galois::PointerWithSize dummy_ones(dummy_ones_v);

  // since layer isn't 0 anymore, backward phase will actually return something
  dummy_ones_v.assign(14, 1);
  // 0 out unsampled nodes
  dummy_ones_v[2]  = 0;
  dummy_ones_v[3]  = 0;
  dummy_ones_v[6]  = 0;
  dummy_ones_v[7]  = 0;
  dummy_ones_v[12] = 0;
  dummy_ones_v[13] = 0;

  galois::PointerWithSize<galois::GNNFloat> layer_1_backward_output =
      layer_1->BackwardPhase(test_graph.GetLocalFeatures(), &dummy_ones);

  //////////////////////////////////////////////////////////////////////////////
  // check that multiplies go as expected
  //////////////////////////////////////////////////////////////////////////////

  GALOIS_LOG_ASSERT(layer_1_backward_output.size() == 21);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[0] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[1] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[2] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[3] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[4] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[5] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[6] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[7] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[8] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[9] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[10] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[11] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[12] == 2);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[13] == 2);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[14] == 2);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[15] == 2);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[16] == 2);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[17] == 2);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[18] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[19] == 0);
  GALOIS_LOG_ASSERT((layer_1_backward_output)[20] == 0);

  galois::PointerWithSize<galois::GNNFloat> layer_1_weight_gradients =
      layer_1->GetLayerWeightGradients();
  // make sure they are sane
  GALOIS_LOG_ASSERT(layer_1_weight_gradients.size() == 6);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[0] == 9);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[1] == 9);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[2] == 9);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[3] == 9);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[4] == 9);

  layer_1.reset();

  //////////////////////////////////////////////////////////////////////////////
  // softmax
  //////////////////////////////////////////////////////////////////////////////

  galois::GNNLayerDimensions dimension_out;
  dimension_out.input_rows     = 7;
  dimension_out.input_columns  = test_graph.GetNumLabelClasses();
  dimension_out.output_columns = test_graph.GetNumLabelClasses();
  std::vector<galois::GNNFloat> softmax_input(49, 0.0);
  // create input with perfect accuracy
  softmax_input[0]  = 1;
  softmax_input[8]  = 1;
  softmax_input[16] = 1;
  softmax_input[24] = 1;
  softmax_input[32] = 1;
  softmax_input[40] = 1;
  softmax_input[48] = 1;

  std::vector<galois::GNNFloat> back_matrix_2(49);
  galois::PointerWithSize<galois::GNNFloat> p_back_2(back_matrix_2);

  auto output_layer = std::make_unique<galois::SoftmaxLayer<char, void>>(
      3, test_graph, &p_back_2, dimension_out);
  output_layer->EnableSampling();
  galois::PointerWithSize<galois::GNNFloat> prediction_distribution =
      output_layer->ForwardPhase(softmax_input);

  GALOIS_LOG_ASSERT(galois::MaxIndex(7, &(prediction_distribution[0])) == 0);
  GALOIS_LOG_ASSERT(galois::MaxIndex(7, &(prediction_distribution[2 * 7])) ==
                    2);
  GALOIS_LOG_ASSERT(galois::MaxIndex(7, &(prediction_distribution[4 * 7])) ==
                    4);

  std::vector<size_t> sampled_out = {1, 3, 6};
  // assert sampled out are all 0s
  for (size_t i : sampled_out) {
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 0] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 1] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 2] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 3] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 4] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 5] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 6] == 0.0);
  }
  // softmax back: check sampled out is all 0s (others are floats, too painful)
  galois::PointerWithSize<galois::GNNFloat> asdf =
      output_layer->BackwardPhase(softmax_input, nullptr);
  for (size_t i : sampled_out) {
    GALOIS_LOG_ASSERT(asdf[i * 7 + 0] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 1] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 2] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 3] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 4] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 5] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 6] == 0.0);
  }

  output_layer.reset();

  //////////////////////////////////////////////////////////////////////////////
  // sigmoid
  //////////////////////////////////////////////////////////////////////////////
  galois::graphs::GNNGraph<char, void> multi_graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, false, false);

  auto sigmoid_layer = std::make_unique<galois::SigmoidLayer<char, void>>(
      3, multi_graph, &p_back_2, dimension_out);
  sigmoid_layer->EnableSampling();
  // reuse softmax input; only thing interested in is checking for 0s
  prediction_distribution = sigmoid_layer->ForwardPhase(softmax_input);
  for (size_t i : sampled_out) {
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 0] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 1] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 2] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 3] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 4] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 5] == 0.0);
    GALOIS_LOG_ASSERT(prediction_distribution[i * 7 + 6] == 0.0);
  }
  asdf = sigmoid_layer->BackwardPhase(softmax_input, nullptr);
  for (size_t i : sampled_out) {
    GALOIS_LOG_ASSERT(asdf[i * 7 + 0] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 1] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 2] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 3] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 4] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 5] == 0.0);
    GALOIS_LOG_ASSERT(asdf[i * 7 + 6] == 0.0);
  }

  return 0;
}
