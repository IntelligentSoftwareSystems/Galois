//! @file convlayer-test.cpp
//! Conv layer test with a test graph

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
  galois::graphs::GNNGraph test_graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true);

  std::vector<galois::GNNFloat> feats = test_graph.GetLocalFeatures();
  //////////////////////////////////////////////////////////////////////////////
  // doubles as a test for reading as well
  GALOIS_LOG_ASSERT(7 == test_graph.size());
  GALOIS_LOG_ASSERT(21 == feats.size());
  GALOIS_LOG_ASSERT(0.0 == feats[0]);
  GALOIS_LOG_ASSERT(0.0 == feats[1]);
  GALOIS_LOG_ASSERT(0.0 == feats[2]);
  GALOIS_LOG_ASSERT(1.0 == feats[3]);
  GALOIS_LOG_ASSERT(1.0 == feats[4]);
  GALOIS_LOG_ASSERT(1.0 == feats[5]);
  GALOIS_LOG_ASSERT(2.0 == feats[6]);
  GALOIS_LOG_ASSERT(2.0 == feats[7]);
  GALOIS_LOG_ASSERT(2.0 == feats[8]);
  GALOIS_LOG_ASSERT(3.0 == feats[9]);
  GALOIS_LOG_ASSERT(3.0 == feats[10]);
  GALOIS_LOG_ASSERT(3.0 == feats[11]);
  GALOIS_LOG_ASSERT(4.0 == feats[12]);
  GALOIS_LOG_ASSERT(4.0 == feats[13]);
  GALOIS_LOG_ASSERT(4.0 == feats[14]);
  GALOIS_LOG_ASSERT(5.0 == feats[15]);
  GALOIS_LOG_ASSERT(5.0 == feats[16]);
  GALOIS_LOG_ASSERT(5.0 == feats[17]);
  GALOIS_LOG_ASSERT(6.0 == feats[18]);
  GALOIS_LOG_ASSERT(6.0 == feats[19]);
  GALOIS_LOG_ASSERT(6.0 == feats[20]);
  //////////////////////////////////////////////////////////////////////////////

  galois::GNNLayerDimensions dimension_0;
  dimension_0.input_rows     = 7;
  dimension_0.input_columns  = 3;
  dimension_0.output_columns = 2;

  galois::GNNConfig dcon;
  dcon.allow_aggregate_after_update = false;

  // create the layer, no norm factor
  std::unique_ptr<galois::GraphConvolutionalLayer> layer_0 =
      std::make_unique<galois::GraphConvolutionalLayer>(0, test_graph,
                                                        dimension_0, dcon);
  layer_0->InitAllWeightsTo1();
  // make sure it runs in a sane manner
  const std::vector<galois::GNNFloat>& layer_0_forward_output =
      layer_0->ForwardPhase(test_graph.GetLocalFeatures());

  //////////////////////////////////////////////////////////////////////////////
  // sanity check layer 0 output
  //////////////////////////////////////////////////////////////////////////////
  // since norm factors aren't invovled it is possible to do full assertions
  // 7 x 2
  GALOIS_LOG_ASSERT(layer_0_forward_output.size() == 14);
  GALOIS_LOG_ASSERT(layer_0_forward_output[0] == 3);
  GALOIS_LOG_ASSERT(layer_0_forward_output[1] == 3);
  GALOIS_LOG_ASSERT(layer_0_forward_output[2] == 6);
  GALOIS_LOG_ASSERT(layer_0_forward_output[3] == 6);
  GALOIS_LOG_ASSERT(layer_0_forward_output[4] == 12);
  GALOIS_LOG_ASSERT(layer_0_forward_output[5] == 12);
  GALOIS_LOG_ASSERT(layer_0_forward_output[6] == 18);
  GALOIS_LOG_ASSERT(layer_0_forward_output[7] == 18);
  GALOIS_LOG_ASSERT(layer_0_forward_output[8] == 24);
  GALOIS_LOG_ASSERT(layer_0_forward_output[9] == 24);
  GALOIS_LOG_ASSERT(layer_0_forward_output[10] == 30);
  GALOIS_LOG_ASSERT(layer_0_forward_output[11] == 30);
  GALOIS_LOG_ASSERT(layer_0_forward_output[12] == 15);
  GALOIS_LOG_ASSERT(layer_0_forward_output[13] == 15);
  //////////////////////////////////////////////////////////////////////////////

  // dummy 1 matrix
  std::vector<galois::GNNFloat> dummy_ones(14, 1);

  // backward pass checking
  // layer 0 means that an empty weight matrix is returned since there is no
  // point passing back anything
  std::vector<galois::GNNFloat>* layer_0_backward_output =
      layer_0->BackwardPhase(test_graph.GetLocalFeatures(), &dummy_ones);

  //////////////////////////////////////////////////////////////////////////////
  // sanity check layer 0 backward output; all 0 because layer 0
  //////////////////////////////////////////////////////////////////////////////
  // since norm factors aren't invovled it is possible to do full assertions
  // 7 x 3
  GALOIS_LOG_ASSERT(layer_0_backward_output->size() == 21);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[0] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[1] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[2] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[3] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[4] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[5] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[6] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[7] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[8] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[9] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[10] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[11] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[12] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[13] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[14] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[15] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[16] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[17] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[18] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[19] == 0);
  GALOIS_LOG_ASSERT((*layer_0_backward_output)[20] == 0);

  const std::vector<galois::GNNFloat> layer_0_weight_gradients =
      layer_0->GetLayerWeightGradients();
  // make sure they are sane
  GALOIS_LOG_ASSERT(layer_0_weight_gradients.size() == 6);
  GALOIS_LOG_ASSERT(layer_0_weight_gradients[0] == 21);
  GALOIS_LOG_ASSERT(layer_0_weight_gradients[1] == 21);
  GALOIS_LOG_ASSERT(layer_0_weight_gradients[2] == 21);
  GALOIS_LOG_ASSERT(layer_0_weight_gradients[3] == 21);
  GALOIS_LOG_ASSERT(layer_0_weight_gradients[4] == 21);

  layer_0.reset();

  //////////////////////////////////////////////////////////////////////////////

  // create layer 1 for testing backward prop actually giving weights back

  std::unique_ptr<galois::GraphConvolutionalLayer> layer_1 =
      std::make_unique<galois::GraphConvolutionalLayer>(1, test_graph,
                                                        dimension_0, dcon);
  layer_1->InitAllWeightsTo1();
  const std::vector<galois::GNNFloat>& layer_1_forward_output =
      layer_1->ForwardPhase(test_graph.GetLocalFeatures());
  // same check as before for sanity purposes
  GALOIS_LOG_ASSERT(layer_1_forward_output.size() == 14);
  GALOIS_LOG_ASSERT(layer_1_forward_output[0] == 3);
  GALOIS_LOG_ASSERT(layer_1_forward_output[1] == 3);
  GALOIS_LOG_ASSERT(layer_1_forward_output[2] == 6);
  GALOIS_LOG_ASSERT(layer_1_forward_output[3] == 6);
  GALOIS_LOG_ASSERT(layer_1_forward_output[4] == 12);
  GALOIS_LOG_ASSERT(layer_1_forward_output[5] == 12);
  GALOIS_LOG_ASSERT(layer_1_forward_output[6] == 18);
  GALOIS_LOG_ASSERT(layer_1_forward_output[7] == 18);
  GALOIS_LOG_ASSERT(layer_1_forward_output[8] == 24);
  GALOIS_LOG_ASSERT(layer_1_forward_output[9] == 24);
  GALOIS_LOG_ASSERT(layer_1_forward_output[10] == 30);
  GALOIS_LOG_ASSERT(layer_1_forward_output[11] == 30);
  GALOIS_LOG_ASSERT(layer_1_forward_output[12] == 15);
  GALOIS_LOG_ASSERT(layer_1_forward_output[13] == 15);

  // since layer isn't 0 anymore, backward phase will actually return something
  dummy_ones.assign(14, 1);
  std::vector<galois::GNNFloat>* layer_1_backward_output =
      layer_1->BackwardPhase(test_graph.GetLocalFeatures(), &dummy_ones);
  //////////////////////////////////////////////////////////////////////////////
  // check that multiplies go as expected
  //////////////////////////////////////////////////////////////////////////////
  GALOIS_LOG_ASSERT(layer_1_backward_output->size() == 21);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[0] == 2);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[1] == 2);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[2] == 2);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[3] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[4] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[5] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[6] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[7] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[8] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[9] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[10] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[11] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[12] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[13] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[14] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[15] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[16] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[17] == 4);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[18] == 2);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[19] == 2);
  GALOIS_LOG_ASSERT((*layer_1_backward_output)[20] == 2);

  const std::vector<galois::GNNFloat> layer_1_weight_gradients =
      layer_1->GetLayerWeightGradients();
  // make sure they are sane
  GALOIS_LOG_ASSERT(layer_1_weight_gradients.size() == 6);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[0] == 21);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[1] == 21);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[2] == 21);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[3] == 21);
  GALOIS_LOG_ASSERT(layer_1_weight_gradients[4] == 21);

  layer_1.reset();

  //////////////////////////////////////////////////////////////////////////////

  galois::GNNConfig config;
  config.do_dropout                   = true;
  config.do_activation                = true;
  config.do_normalization             = true;
  config.allow_aggregate_after_update = false;

  // finally, just make sure dropout and activation run without crashes
  // (verification requires floating point accuracy or setting a seed which I
  // don't have time for at the moment
  // TODO in future maybe add better unit test for this
  std::unique_ptr<galois::GraphConvolutionalLayer> layer_2 =
      std::make_unique<galois::GraphConvolutionalLayer>(1, test_graph,
                                                        dimension_0, config);
  const std::vector<galois::GNNFloat> l2_fo =
      layer_2->ForwardPhase(test_graph.GetLocalFeatures());
  GALOIS_LOG_ASSERT(l2_fo.size() == 14);
  GALOIS_LOG_VERBOSE("{}", l2_fo[0]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[1]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[2]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[3]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[4]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[5]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[6]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[7]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[8]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[9]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[10]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[11]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[12]);
  GALOIS_LOG_VERBOSE("{}", l2_fo[13]);

  std::vector<galois::GNNFloat>* l2_bo =
      layer_2->BackwardPhase(test_graph.GetLocalFeatures(), &dummy_ones);

  GALOIS_LOG_ASSERT(l2_bo->size() == 21);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[0]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[1]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[2]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[3]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[4]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[5]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[6]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[7]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[8]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[9]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[10]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[11]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[12]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[13]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[14]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[15]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[16]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[17]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[18]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[19]);
  GALOIS_LOG_VERBOSE("{}", (*l2_bo)[20]);

  return 0;
}
