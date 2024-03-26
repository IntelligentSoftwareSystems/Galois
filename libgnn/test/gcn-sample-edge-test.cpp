/**
 * This test checks correctness by comparing hand calculation
 * of the forward and backward phases.
 * This is implemented to check correctness of GCN layer.
 * Below is the process:
 * 1. Mark and check nodes and edges to be initially sampled.
 * 2. Nodes adjacent to the sampled edges are sampled.
 * 3. Perform forward/backward phases and compare the results
 *    with hand calculation results.
 */

// TODO(hc): Designing and implementing multi-host execution is
// a time consuming task and so, I will work on that later.
// But, without test, I confirmed correctness of the multi-host
// execution based on the below changes.
//
// 1. Set all layer weights to 1, instead of random values.
// 2. Used nodes within global node ID range for training.
// (So, the nodes are deterministic)
// (The original code uses random selection to match SHAD's one)
// 3. Compared 1-host and multi hosts, like 2 and 4 hosts,
// accuracy results on the graph sampling mode.
// 4. They should be same if the GCN graph sampling is correct.
// (It was on the test done on 09/15/2023)

#include "galois/layers/GraphConvolutionalLayer.h"
#include "galois/layers/SAGELayer.h"

int main() {
  galois::DistMemSys G;

  // tester graph: 0 - 1 - 2 - 3 - 4 - 5 - 6
  galois::graphs::GNNGraph<char, void> test_graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true, false);
  test_graph.InitializeSamplingData();

  galois::GNNLayerConfig dcon;
  dcon.disable_aggregate_after_update = false;
  dcon.disable_normalization          = false;
  dcon.DebugConfig();
  // Choose a few sample nodes
  test_graph.SetSampledNode(0);
  test_graph.SetSampledNode(4);
  test_graph.UnsetSampledNode(1);
  test_graph.UnsetSampledNode(2);
  test_graph.UnsetSampledNode(3);
  test_graph.UnsetSampledNode(5);
  test_graph.UnsetSampledNode(6);

  test_graph.ResizeSamplingBitsets();
  test_graph.SampleAllEdges(0, false, 1);

  // After the above lines, nodes 0, 1, 3, 4, 5 and
  // edges 0, 7, 8 should be sampled.
  // So,
  // 0 -> 1, 2 <- 3 -> 4
  GALOIS_LOG_ASSERT(test_graph.IsInSampledGraph(0));
  GALOIS_LOG_ASSERT(test_graph.IsInSampledGraph(1));
  GALOIS_LOG_ASSERT(test_graph.IsInSampledGraph(3));
  GALOIS_LOG_ASSERT(test_graph.IsInSampledGraph(4));
  GALOIS_LOG_ASSERT(test_graph.IsInSampledGraph(5));

  GALOIS_LOG_ASSERT(test_graph.IsEdgeSampledAny(7));
  GALOIS_LOG_ASSERT(test_graph.IsEdgeSampledAny(8));

  galois::DynamicBitSet& bset = test_graph.GetDefinitelySampledNodesBset();
  bset.ParallelReset();
  bset.set(0);
  bset.set(1);
  bset.set(3);
  bset.set(4);
  bset.set(5);
  test_graph.ConstructSampledSubgraph(1);
  test_graph.EnableSubgraph();

  galois::GNNLayerDimensions dimension_0;
  dimension_0.input_rows     = 5;
  dimension_0.input_columns  = 3;
  dimension_0.output_columns = 2;

  // Layer declaration
  std::vector<galois::GNNFloat> back_matrix(15);
  galois::PointerWithSize<galois::GNNFloat> p_back(back_matrix);
  std::unique_ptr<galois::GraphConvolutionalLayer<char, void>> layer_1 =
      std::make_unique<galois::GraphConvolutionalLayer<char, void>>(
          1, test_graph, &p_back, dimension_0, dcon);

  layer_1->InitAllWeightsTo1();
  layer_1->EnableSampling();
  galois::PointerWithSize<galois::GNNFloat> features =
      test_graph.GetLocalFeatures();

  galois::PointerWithSize<galois::GNNFloat> layer_1_forward_output =
      layer_1->ForwardPhase(features);

  GALOIS_LOG_ASSERT(layer_1_forward_output[0] == 3);
  GALOIS_LOG_ASSERT(layer_1_forward_output[1] == 3);
  GALOIS_LOG_ASSERT(layer_1_forward_output[2] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[3] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[4] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[5] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[6] == 24);
  GALOIS_LOG_ASSERT(layer_1_forward_output[7] == 24);
  GALOIS_LOG_ASSERT(layer_1_forward_output[8] == 0);
  GALOIS_LOG_ASSERT(layer_1_forward_output[9] == 0);

  // Dummy gradients
  std::vector<galois::GNNFloat> dummy_ones_v(10, 1);
  galois::PointerWithSize dummy_ones(dummy_ones_v);
  dummy_ones_v.assign(10, 1);
  dummy_ones_v[4] = 0;
  dummy_ones_v[5] = 0;

  galois::PointerWithSize<galois::GNNFloat> layer_1_backward_output =
      layer_1->BackwardPhase(test_graph.GetLocalFeatures(), &dummy_ones);

  GALOIS_LOG_ASSERT(layer_1_backward_output[0] == 0);
  GALOIS_LOG_ASSERT(layer_1_backward_output[1] == 0);
  GALOIS_LOG_ASSERT(layer_1_backward_output[2] == 0);
  GALOIS_LOG_ASSERT(layer_1_backward_output[3] == 2);
  GALOIS_LOG_ASSERT(layer_1_backward_output[4] == 2);
  GALOIS_LOG_ASSERT(layer_1_backward_output[5] == 2);
  GALOIS_LOG_ASSERT(layer_1_backward_output[6] == 2);
  GALOIS_LOG_ASSERT(layer_1_backward_output[7] == 2);
  GALOIS_LOG_ASSERT(layer_1_backward_output[8] == 2);
  GALOIS_LOG_ASSERT(layer_1_backward_output[9] == 0);
  GALOIS_LOG_ASSERT(layer_1_backward_output[10] == 0);
  GALOIS_LOG_ASSERT(layer_1_backward_output[11] == 0);
  GALOIS_LOG_ASSERT(layer_1_backward_output[12] == 2);
  GALOIS_LOG_ASSERT(layer_1_backward_output[13] == 2);
  GALOIS_LOG_ASSERT(layer_1_backward_output[14] == 2);

  galois::PointerWithSize<galois::GNNFloat> layer_1_weight_gradients =
      layer_1->GetLayerWeightGradients();

  GALOIS_LOG_ASSERT(layer_1_weight_gradients[0] == 9);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[1] == 9);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[2] == 9);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[3] == 9);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[4] == 9);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[5] == 9);

  return 0;
}
