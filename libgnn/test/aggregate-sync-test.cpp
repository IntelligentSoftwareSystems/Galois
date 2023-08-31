#include "galois/Logging.h"
#include "galois/GraphNeuralNetwork.h"
#include "galois/layers/GraphConvolutionalLayer.h"

int main() {
  galois::DistMemSys G;

  if (galois::runtime::getSystemNetworkInterface().Num == 1) {
    GALOIS_LOG_WARN("This test should be run with multiple hosts/processes!");
  }

  auto test_graph = std::make_unique<galois::graphs::GNNGraph<char, void>>(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true, false);

  // print edges for sanity
  for (size_t node = 0; node < test_graph->size(); node++) {
    for (auto e = test_graph->edge_begin(node); e != test_graph->edge_end(node);
         e++) {
      galois::gPrint(test_graph->host_prefix(), "Edge ",
                     test_graph->GetGID(node), " ",
                     test_graph->GetGID(test_graph->GetEdgeDest(e)), "\n");
    }
  }
  for (auto own = test_graph->begin_owned(); own != test_graph->end_owned();
       own++) {
    galois::gPrint(test_graph->host_prefix(), "Node owned GID ",
                   test_graph->GetGID(*own), "\n");
  }

  // create same layer from convlayer-test and make sure result is the same even
  // in multi-host environment
  galois::GNNLayerDimensions dimension_0;
  dimension_0.input_rows     = test_graph->size();
  dimension_0.input_columns  = 3;
  dimension_0.output_columns = 2;
  galois::GNNLayerConfig l_config;
  l_config.DebugConfig();
  l_config.disable_aggregate_after_update = true;

  galois::PointerWithSize<galois::GNNFloat> p_null(nullptr, 0);
  std::vector<galois::GNNFloat> back_matrix(test_graph->size() * 3);
  galois::PointerWithSize<galois::GNNFloat> p_back(back_matrix);

  // create the layer, no norm factor
  std::unique_ptr<galois::GraphConvolutionalLayer<char, void>> layer_0 =
      std::make_unique<galois::GraphConvolutionalLayer<char, void>>(
          0, *(test_graph.get()), &p_null, dimension_0, l_config);
  layer_0->InitAllWeightsTo1();
  // make sure it runs in a sane manner
  galois::PointerWithSize<galois::GNNFloat> layer_0_forward_output =
      layer_0->ForwardPhase(test_graph->GetLocalFeatures());

  //////////////////////////////////////////////////////////////////////////////
  // sanity check output
  //////////////////////////////////////////////////////////////////////////////

  // check each row on each host: convert row into GID, and based on GID we
  // know what the ground truth is
  // row 0 = 3
  // row 1 = 6
  // row 2 = 12
  // row 3 = 18
  // row 4 = 24
  // row 5 = 30
  // row 6 = 15

  // row should correspond to LID
  for (size_t row = 0; row < test_graph->size(); row++) {
    // row -> GID
    size_t global_row = test_graph->GetGID(row);

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
      GALOIS_LOG_VASSERT(layer_0_forward_output[row * 2 + c] == ground_truth,
                         "should be {} not {}", ground_truth,
                         layer_0_forward_output[row * 2 + c]);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  std::vector<galois::GNNFloat> dummy_ones_v(test_graph->size() * 2, 1);
  galois::PointerWithSize<galois::GNNFloat> dummy_ones(dummy_ones_v);
  // backward pass checking
  // layer 0 means that an empty weight matrix is returned since there is no
  // point passing back anything
  galois::PointerWithSize<galois::GNNFloat> layer_0_backward_output =
      layer_0->BackwardPhase(test_graph->GetLocalFeatures(), &dummy_ones);

  //////////////////////////////////////////////////////////////////////////////
  // sanity check layer 0 backward output: empty
  //////////////////////////////////////////////////////////////////////////////

  GALOIS_LOG_ASSERT(layer_0_backward_output.size() == 0);

  //////////////////////////////////////////////////////////////////////////////
  // layer 1 to check backward output
  //////////////////////////////////////////////////////////////////////////////
  std::unique_ptr<galois::GraphConvolutionalLayer<char, void>> layer_1 =
      std::make_unique<galois::GraphConvolutionalLayer<char, void>>(
          1, *(test_graph.get()), &p_back, dimension_0, l_config);
  layer_1->InitAllWeightsTo1();
  galois::PointerWithSize<galois::GNNFloat> layer_1_forward_output =
      layer_1->ForwardPhase(test_graph->GetLocalFeatures());

  // same check for forward as before
  for (size_t row = 0; row < test_graph->size(); row++) {
    // row -> GID
    size_t global_row = test_graph->GetGID(row);

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
      GALOIS_LOG_ASSERT(layer_1_forward_output[row * 2 + c] == ground_truth);
    }
  }

  // since layer isn't 0 anymore, backward phase will actually return something
  dummy_ones_v.assign(test_graph->size() * 2, 1);
  galois::PointerWithSize<galois::GNNFloat> layer_1_backward_output =
      layer_1->BackwardPhase(test_graph->GetLocalFeatures(), &dummy_ones);

  for (size_t row = 0; row < test_graph->size(); row++) {
    // row -> GID
    size_t global_row = test_graph->GetGID(row);

    galois::GNNFloat ground_truth = 0.0;

    switch (global_row) {
    case 0:
    case 6:
      ground_truth = 2;
      break;
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
      ground_truth = 4;
      break;
    default:
      GALOIS_LOG_FATAL("bad global row for test graph");
      break;
    }

    // size 3 columns
    for (size_t c = 0; c < 3; c++) {
      GALOIS_LOG_ASSERT((layer_1_backward_output)[row * 3 + c] == ground_truth);
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  auto test_graph_2 = std::make_unique<galois::graphs::GNNGraph<char, void>>(
      "tester", galois::graphs::GNNPartitionScheme::kCVC, true, false);
  // print edges for sanity
  for (size_t node = 0; node < test_graph_2->size(); node++) {
    for (auto e = test_graph_2->edge_begin(node);
         e != test_graph_2->edge_end(node); e++) {
      galois::gPrint(test_graph_2->host_prefix(), "Edge ",
                     test_graph_2->GetGID(node), " ",
                     test_graph_2->GetGID(test_graph_2->GetEdgeDest(e)), "\n");
    }
  }
  for (auto own = test_graph_2->begin_owned(); own != test_graph_2->end_owned();
       own++) {
    galois::gPrint(test_graph_2->host_prefix(), "Node owned GID ",
                   test_graph_2->GetGID(*own), "\n");
  }

  // create same layer from convlayer-test and make sure result is the same even
  // in multi-host environment
  dimension_0.input_rows                  = test_graph_2->size();
  dimension_0.input_columns               = 3;
  dimension_0.output_columns              = 2;
  l_config.disable_aggregate_after_update = false;
  l_config.DebugConfig();

  // create the layer, no norm factor
  layer_0 = std::make_unique<galois::GraphConvolutionalLayer<char, void>>(
      0, *(test_graph_2.get()), &p_null, dimension_0, l_config);
  layer_0->InitAllWeightsTo1();

  // make sure it runs in a sane manner
  // galois::PointerWithSize<galois::GNNFloat> layer_0_forward_output =
  layer_0_forward_output =
      layer_0->ForwardPhase(test_graph_2->GetLocalFeatures());

  for (size_t row = 0; row < test_graph_2->size(); row++) {
    // row -> GID
    size_t global_row = test_graph_2->GetGID(row);

    if (global_row == 1) {
      galois::gPrint(test_graph_2->host_prefix(), "GID ", global_row, " local ",
                     row, " value ", layer_0_forward_output[row * 2], "\n");
    }
    if (global_row == 4) {
      galois::gPrint(test_graph_2->host_prefix(), "GID ", global_row, " local ",
                     row, " value ", layer_0_forward_output[row * 2], "\n");
    }
  }

  for (size_t row = 0; row < test_graph_2->size(); row++) {
    // row -> GID
    size_t global_row = test_graph_2->GetGID(row);

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
      GALOIS_LOG_VASSERT(layer_0_forward_output[row * 2 + c] == ground_truth,
                         "{} Row {} GID {} need to be {} not {}",
                         test_graph_2->host_prefix(), row, global_row,
                         ground_truth, layer_0_forward_output[row * 2 + c]);
    }
  }

  std::vector<galois::GNNFloat> back_matrix_2(test_graph_2->size() * 3);
  galois::PointerWithSize<galois::GNNFloat> p_back_2(back_matrix_2);

  layer_1 = std::make_unique<galois::GraphConvolutionalLayer<char, void>>(
      1, *(test_graph_2.get()), &p_back_2, dimension_0, l_config);
  layer_1->InitAllWeightsTo1();
  layer_1_forward_output =
      layer_1->ForwardPhase(test_graph_2->GetLocalFeatures());

  // same check for forward as before
  for (size_t row = 0; row < test_graph_2->size(); row++) {
    // row -> GID
    size_t global_row = test_graph_2->GetGID(row);

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
      GALOIS_LOG_ASSERT(layer_1_forward_output[row * 2 + c] == ground_truth);
    }
  }

  std::vector<galois::GNNFloat> dummy_ones_v2(test_graph_2->size() * 2, 1);
  galois::PointerWithSize<galois::GNNFloat> dummy_ones2(dummy_ones_v2);
  layer_1_backward_output =
      layer_1->BackwardPhase(test_graph_2->GetLocalFeatures(), &dummy_ones2);

  for (size_t row = 0; row < test_graph_2->size(); row++) {
    // row -> GID
    size_t global_row = test_graph_2->GetGID(row);

    galois::GNNFloat ground_truth = 0.0;

    switch (global_row) {
    case 0:
    case 6:
      ground_truth = 2;
      break;
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
      ground_truth = 4;
      break;
    default:
      GALOIS_LOG_FATAL("bad global row for test graph");
      break;
    }

    // size 3 columns
    for (size_t c = 0; c < 3; c++) {
      GALOIS_LOG_ASSERT((layer_1_backward_output)[row * 3 + c] == ground_truth);
    }
  }
}
