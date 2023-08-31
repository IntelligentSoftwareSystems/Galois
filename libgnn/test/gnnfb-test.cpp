//! @file gnnfb-test.cpp
//! Runs a forward and backward phase on a GCN and an example graph.

#include "galois/Logging.h"
#include "galois/GraphNeuralNetwork.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);

  GALOIS_LOG_VERBOSE("[{}] Using {} threads",
                     galois::runtime::getSystemNetworkInterface().ID,
                     num_threads);
  // load test graph
  auto test_graph = std::make_unique<galois::graphs::GNNGraph<char, void>>(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true, false);

  // 2 layer test with softmax
  std::vector<galois::GNNLayerType> layer_types = {
      galois::GNNLayerType::kGraphConvolutional,
      galois::GNNLayerType::kGraphConvolutional};
  // note this includes the output; last 2 must be same because softmax
  std::vector<size_t> layer_output_sizes = {4, 7, 7};
  galois::GNNLayerConfig dcon;
  dcon.disable_aggregate_after_update = false;
  dcon.DebugConfig();
  // note GNNLayerConfig is passed in; use a config that does not do anything
  // extra like dropout or activation and the like so that input is easier to
  // verify
  galois::GraphNeuralNetworkConfig gnn_config(
      2, layer_types, layer_output_sizes, galois::GNNOutputLayerType::kSoftmax,
      dcon);
  // input is 7 x 3, layers are then 3 x 4 and 4 x 7 and 7 x 7
  // middle 2 are trainable so 12 and 28
  std::vector<size_t> adam_sizes = {12, 28};
  auto adam = std::make_unique<galois::AdamOptimizer>(adam_sizes, 2);
  auto gnn  = std::make_unique<galois::GraphNeuralNetwork<char, void>>(
      std::move(test_graph), std::move(adam), std::move(gnn_config));
  // for constancy set everything to 1
  gnn->SetAllLayerWeightsTo1();

  //////////////////////////////////////////////////////////////////////////////
  // forward phase
  //////////////////////////////////////////////////////////////////////////////
  const galois::PointerWithSize<galois::GNNFloat> fo_out = gnn->DoInference();

  // check output for layers to make sure it's as expected
  galois::PointerWithSize<galois::GNNFloat> lf0_out =
      gnn->GetIntermediateLayer(0)->GetForwardOutput();
  GALOIS_LOG_ASSERT(lf0_out.size() == 28);
  for (size_t i = 0; i < 4; i++) {
    GALOIS_LOG_ASSERT(lf0_out[0 + i] == 3);
  }
  for (size_t i = 0; i < 4; i++) {
    GALOIS_LOG_ASSERT(lf0_out[4 + i] == 6);
  }
  for (size_t i = 0; i < 4; i++) {
    GALOIS_LOG_ASSERT(lf0_out[8 + i] == 12);
  }
  for (size_t i = 0; i < 4; i++) {
    GALOIS_LOG_ASSERT(lf0_out[12 + i] == 18);
  }
  for (size_t i = 0; i < 4; i++) {
    GALOIS_LOG_ASSERT(lf0_out[16 + i] == 24);
  }
  for (size_t i = 0; i < 4; i++) {
    GALOIS_LOG_ASSERT(lf0_out[20 + i] == 30);
  }
  for (size_t i = 0; i < 4; i++) {
    GALOIS_LOG_ASSERT(lf0_out[24 + i] == 15);
  }

  // Disabled: this test worked in past because forward outputs were all
  // separate matrices, but due to space saving measures this forward output
  // gets messed with by the softmax call

  // const galois::PointerWithSize<galois::GNNFloat> lf1_out =
  //    gnn->GetIntermediateLayer(1)->GetForwardOutput();
  // GALOIS_LOG_ASSERT(lf1_out.size() == 49);
  // for (size_t i = 0; i < 7; i++) {
  //  GALOIS_LOG_VASSERT(lf1_out[0 + i] == 24, "{} vs {} (correct)", lf1_out[0 +
  //  i], 24);
  //}
  // for (size_t i = 0; i < 7; i++) {
  //  GALOIS_LOG_ASSERT(lf1_out[7 + i] == 60);
  //}
  // for (size_t i = 0; i < 7; i++) {
  //  GALOIS_LOG_ASSERT(lf1_out[14 + i] == 96);
  //}
  // for (size_t i = 0; i < 7; i++) {
  //  GALOIS_LOG_ASSERT(lf1_out[21 + i] == 144);
  //}
  // for (size_t i = 0; i < 7; i++) {
  //  GALOIS_LOG_ASSERT(lf1_out[28 + i] == 192);
  //}
  // for (size_t i = 0; i < 7; i++) {
  //  GALOIS_LOG_ASSERT(lf1_out[35 + i] == 156);
  //}
  // for (size_t i = 0; i < 7; i++) {
  //  GALOIS_LOG_ASSERT(lf1_out[42 + i] == 120);
  //}

  GALOIS_LOG_ASSERT(fo_out.size() == 49);
  // since row all same, prob distribution across row should be same
  for (size_t c = 0; c < 49; c += 7) {
    for (size_t i = 0; i < 6; i++) {
      GALOIS_LOG_VERBOSE("{}", fo_out[c + i]);
      GALOIS_LOG_ASSERT(fo_out[c + i] == fo_out[c + i + 1]);
    }
  }

  // train mode = last 2 should be masked off
  for (size_t c = 35; c < 49; c += 7) {
    for (size_t i = 0; i < 6; i++) {
      GALOIS_LOG_ASSERT(fo_out[c + i] == 0);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // backward phase; run it; verifying is difficult due to floating point
  // nature of softmax gradients
  //////////////////////////////////////////////////////////////////////////////

  gnn->GradientPropagation();

  //////////////////////////////////////////////////////////////////////////////
  // verify forward val and test masks
  //////////////////////////////////////////////////////////////////////////////
  gnn->SetLayerPhases(galois::GNNPhase::kValidate);
  gnn->SetAllLayerWeightsTo1();
  const galois::PointerWithSize<galois::GNNFloat> fo_out_val =
      gnn->DoInference();
  for (size_t c = 0; c < 49; c += 7) {
    for (size_t i = 0; i < 6; i++) {
      GALOIS_LOG_ASSERT(fo_out_val[c + i] == fo_out_val[c + i + 1]);
    }
  }
  // first 5 and last should be 0s
  for (size_t c = 0; c < 35; c += 7) {
    for (size_t i = 0; i < 6; i++) {
      GALOIS_LOG_ASSERT(fo_out_val[c + i] == 0);
    }
  }
  for (size_t c = 42; c < 49; c += 7) {
    for (size_t i = 0; i < 6; i++) {
      GALOIS_LOG_ASSERT(fo_out_val[c + i] == 0);
    }
  }

  // all but last should be 0s
  gnn->SetLayerPhases(galois::GNNPhase::kTest);
  gnn->SetAllLayerWeightsTo1();
  galois::PointerWithSize<galois::GNNFloat> fo_out_test = gnn->DoInference();
  for (size_t c = 0; c < 49; c += 7) {
    for (size_t i = 0; i < 6; i++) {
      GALOIS_LOG_ASSERT(fo_out_test[c + i] == fo_out_test[c + i + 1]);
    }
  }
  // first 5 and last should be 0s
  for (size_t c = 0; c < 42; c += 7) {
    for (size_t i = 0; i < 6; i++) {
      GALOIS_LOG_ASSERT(fo_out_test[c + i] == 0);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // run different config of gnn with dropout/activation
  //////////////////////////////////////////////////////////////////////////////

  GALOIS_LOG_VERBOSE("Running with different congifuration");

  test_graph = std::make_unique<galois::graphs::GNNGraph<char, void>>(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true, false);
  galois::GraphNeuralNetworkConfig gnn_config2(
      2, layer_types, layer_output_sizes, galois::GNNOutputLayerType::kSoftmax,
      dcon);
  auto adam2 = std::make_unique<galois::AdamOptimizer>(adam_sizes, 2);
  auto gnn2  = std::make_unique<galois::GraphNeuralNetwork<char, void>>(
      std::move(test_graph), std::move(adam2), std::move(gnn_config2));
  // run to make sure no crashes occur
  gnn2->DoInference();
  gnn2->GradientPropagation();
}
