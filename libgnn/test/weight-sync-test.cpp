#include "galois/Logging.h"
#include "galois/GraphNeuralNetwork.h"
#include "galois/layers/GraphConvolutionalLayer.h"

int main() {
  galois::DistMemSys G;

  if (galois::runtime::getSystemNetworkInterface().Num == 4) {
    GALOIS_LOG_ERROR("This test should be run with 4 hosts/processes");
    exit(1);
  }

  auto test_graph = std::make_unique<galois::graphs::GNNGraph>(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true);

  // create same layer from convlayer-test and make sure result is the same even
  // in multi-host environment
  galois::GNNLayerDimensions dimension_0;
  dimension_0.input_rows     = test_graph->size();
  dimension_0.input_columns  = 3;
  dimension_0.output_columns = 2;
  galois::GNNLayerConfig dcon;

  dcon.disable_aggregate_after_update = false;
  // create the layer, no norm factor
  std::unique_ptr<galois::GraphConvolutionalLayer> layer_0 =
      std::make_unique<galois::GraphConvolutionalLayer>(0, *(test_graph.get()),
                                                        dimension_0, dcon);
  layer_0->InitAllWeightsTo1();

  // backward pass checking; check the gradients out
  std::vector<galois::GNNFloat> dummy_ones_v(test_graph->size() * 2, 1);
  galois::PointerWithSize<galois::GNNFloat> dummy_ones(dummy_ones_v);
  layer_0->BackwardPhase(test_graph->GetLocalFeatures(), &dummy_ones);

  // gradient verification; average
  // host 0 has 18, 1 has 21, 2 has 12, 3 has 0s; averaged to 12.75
  const galois::PointerWithSize<galois::GNNFloat>& grads =
      layer_0->GetLayerWeightGradients();
  for (size_t i = 0; i < 6; i++) {
    GALOIS_LOG_ASSERT(grads[i] == 12.75);
  }

  // XXX CVC
}
