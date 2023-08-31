//! @file sigmoidlayer-test.cpp
//! Sigmoid layer test with a test graph
//! No automated ground truth checking; when this was written it was compared
//! manually with pytorch
//! TODO add in automated checking eventually; for now this just makes sure it
//! runs

#include "galois/Logging.h"
#include "galois/GNNMath.h"
#include "galois/layers/SigmoidLayer.h"

int main() {
  galois::DistMemSys G;

  galois::setActiveThreads(1);

  // load test graph
  galois::graphs::GNNGraph<char, void> test_graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, false, false);

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
  softmax_input[1]  = 1;
  softmax_input[2]  = 100000000000;
  softmax_input[3]  = 100000000000000000;
  softmax_input[4]  = -1000;
  softmax_input[5]  = -10;
  softmax_input[6]  = 1000000;
  softmax_input[8]  = 1;
  softmax_input[9]  = 1;
  softmax_input[10] = 1;
  softmax_input[16] = 1;
  softmax_input[17] = 1;
  softmax_input[18] = 1;
  softmax_input[24] = 0;
  softmax_input[32] = 0;
  softmax_input[40] = 0;
  softmax_input[48] = 0;

  std::vector<galois::GNNFloat> back_matrix(49);
  galois::PointerWithSize<galois::GNNFloat> p_back(back_matrix);

  // train mode
  auto output_layer = std::make_unique<galois::SigmoidLayer<char, void>>(
      3, test_graph, &p_back, dimension_0);
  output_layer->ForwardPhase(softmax_input);

  galois::PointerWithSize<galois::GNNFloat> asdf =
      output_layer->BackwardPhase(softmax_input, nullptr);
  printf("Output 1\n========\n");
  for (unsigned i = 0; i < asdf.size(); i++) {
    if (i % 7 == 0) {
      printf("--------------\n");
    }
    printf("%f\n", asdf[i]);
  }
}
