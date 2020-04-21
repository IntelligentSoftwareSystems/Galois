/**
 * Based on the net.hpp file from Caffe deep learning framework.
 */

#include "deepgalois/net.h"

namespace deepgalois {

#ifndef GALOIS_USE_DIST
void Net::init(std::string dataset_str, unsigned epochs, unsigned hidden1, 
               bool selfloop, bool is_single) {
#else
void Net::init(std::string dataset_str, unsigned epochs, unsigned hidden1,
               bool selfloop, Graph* dGraph) {
#endif
#ifndef GALOIS_USE_DIST
  is_single_class = is_single;
  context = new deepgalois::Context();
  context->set_label_class(is_single);
  num_samples = context->read_graph(dataset_str, selfloop);
#else
  context = new deepgalois::DistContext();
  num_samples = dGraph->size();
  context->saveGraph(dGraph);
  // TODO self loop setup?
  context->initializeSyncSubstrate();
#endif

  // read graph, get num nodes
  num_classes = context->read_labels(dataset_str);
  num_epochs = epochs;

  //std::cout << "Reading label masks ... ";
  train_mask.resize(num_samples, 0);
  val_mask.resize(num_samples, 0);
  // get testing and validation sets
  if (dataset_str == "reddit") {
    train_begin = 0, train_count = 153431,
    train_end = train_begin + train_count;
    val_begin = 153431, val_count = 23831, val_end = val_begin + val_count;
    // TODO do all can be used below
#ifndef GALOIS_USE_DIST
    for (size_t i = train_begin; i < train_end; i++) train_mask[i] = 1;
    for (size_t i = val_begin; i < val_end; i++) val_mask[i] = 1;
#else
    // find local ID from global ID, set if it exists
    for (size_t i = train_begin; i < train_end; i++) {
      if (dGraph->isLocal(i)) {
        train_mask[dGraph->getLID(i)] = 1;
      }
    }
    for (size_t i = val_begin; i < val_end; i++) {
      if (dGraph->isLocal(i)) {
        val_mask[dGraph->getLID(i)] = 1;
      }
    }
#endif
  } else {
#ifndef GALOIS_USE_DIST
    train_count =
        read_masks(dataset_str, "train", train_begin, train_end, train_mask);
    val_count = read_masks(dataset_str, "val", val_begin, val_end, val_mask);
#else
    train_count =
        read_masks(dataset_str, "train", train_begin, train_end, train_mask,
                   dGraph);
    val_count = read_masks(dataset_str, "val", val_begin, val_end, val_mask,
                           dGraph);
#endif
  }
  //std::cout << "Done\n";

  // NOTE: train_begin/train_end are global IDs, train_mask is a local id
  // train count and val count are LOCAL counts

  num_layers = NUM_CONV_LAYERS + 1;
  // initialize feature metadata
  feature_dims.resize(num_layers + 1);
  feature_dims[0] =
      context->read_features(dataset_str); // input feature dimension: D
  feature_dims[1] = hidden1;               // hidden1 level embedding: 16
  feature_dims[2] = num_classes;           // output embedding: E
  feature_dims[3] = num_classes;           // normalized output embedding: E
  layers.resize(num_layers);
#ifndef CPU_ONLY
  context->copy_data_to_device(); // copy labels and input features to the device
#endif
}

void Net::train(optimizer* opt, bool need_validate) {
#ifdef GALOIS_USE_DIST
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
  std::string header = "[" + std::to_string(myID) + "] ";
  std::string seperator = "\n";
#else
  //std::string header = "[" + std::to_string(0) + "] ";
  //std::string seperator = "\n";
  std::string header = "";
  std::string seperator = " ";
#endif

  galois::gPrint("\nStart training...\n");
  galois::StatTimer Tupdate("Train-WeightUpdate");
  galois::StatTimer Tfw("Train-Forward");
  galois::StatTimer Tbw("Train-Backward");
  galois::StatTimer Tval("Validation");
  double total_train_time = 0.0;

  Timer t_epoch;
  // run epochs
  for (unsigned i = 0; i < num_epochs; i++) {
    galois::gPrint(header, "Epoch ", std::setw(3), i, seperator);
    t_epoch.Start();

    // training steps
    set_netphases(net_phase::train);
    acc_t train_loss = 0.0, train_acc = 0.0;

    // forward: after this phase, layer edges will contain intermediate features
    // for use during backprop
    Tfw.start();
    train_loss = Net::fprop(train_begin, train_end, train_count, &train_mask[0]); // forward
#ifdef CPU_ONLY
    Graph *g = context->getGraphPointer();
#else
	CSRGraph *g = context->getGpuGraphPointer();
#endif
    if (is_single_class) {
      train_acc = masked_accuracy(train_begin, train_end, train_count,
                                  &train_mask[0], g); // predict
    } else {
      train_acc = masked_multi_class_accuracy(train_begin, train_end, train_count,
                                              &train_mask[0], g); // predict
    }
    Tfw.stop();

    // backward: use intermediate features + ground truth to update layers
    // with feature gradients whcih are then used to calculate weight gradients
    Tbw.start();
    Net::bprop();
    Tbw.stop();

    // gradient update: use gradients stored on each layer to update model for
    // next epoch
    Tupdate.start();
    Net::update_weights(opt); // update parameters
    Tupdate.stop();

    // validation / testing
    set_netphases(net_phase::test);
    galois::gPrint(header, "train_loss ", std::setprecision(3), std::fixed, train_loss,
                   " train_acc ", train_acc, seperator);
    t_epoch.Stop();
    double epoch_time = t_epoch.Millisecs();
    total_train_time += epoch_time;
    if (need_validate) {
      // Validation
      acc_t val_loss = 0.0, val_acc = 0.0;
      Tval.start();
      double val_time = evaluate(val_begin, val_end, val_count, &val_mask[0],
                                 val_loss, val_acc);
      Tval.stop();
      galois::gPrint(header, "val_loss ", std::setprecision(3), std::fixed, val_loss,
                     " val_acc ", val_acc, seperator);
      galois::gPrint(header, "time ", std::setprecision(3), std::fixed, epoch_time + val_time, 
                     " ms (train_time ", epoch_time, " val_time ", val_time, ")\n");
    } else {
      galois::gPrint(header, "train_time ", std::fixed, epoch_time, " ms\n");
    }
  }
  double avg_train_time = total_train_time / (double)num_epochs;
  double throughput = 1000.0 * (double)num_epochs / total_train_time;
  galois::gPrint("\nAverage training time: ", avg_train_time, 
                 " ms. Throughput: ", throughput, " epoch/s\n");
}

// evaluate, i.e. inference or predict
double Net::evaluate(size_t begin, size_t end, size_t count, mask_t* masks,
                     acc_t& loss, acc_t& acc) {
  // TODO may need to do something for the dist case
  Timer t_eval;
  t_eval.Start();
  loss = fprop(begin, end, count, masks);
#ifdef CPU_ONLY
  Graph* g = context->getCpuGraphPointer();
#else
  CSRGraph* g = context->getGpuGraphPointer();
#endif
  if (is_single_class) {
    acc = masked_accuracy(begin, end, count, masks, g);
  } else {
    acc = masked_multi_class_accuracy(begin, end, count, masks, g);
  }
  t_eval.Stop();
  return t_eval.Millisecs();
}

void Net::construct_layers() {
  std::cout << "\nConstructing layers...\n";
  append_conv_layer(0, true);                    // first conv layer
  append_conv_layer(1);                          // hidden1 layer
  append_out_layer(2);                           // output layer
  layers[0]->set_in_data(context->get_in_ptr()); // feed input data
  context->norm_factor_counting();
  set_contexts();
}

//! Add an output layer to the network
void Net::append_out_layer(size_t layer_id) {
  assert(layer_id > 0); // can not be the first layer
  std::vector<size_t> in_dims(2), out_dims(2);
  in_dims[0] = out_dims[0] = num_samples;
  in_dims[1]               = get_in_dim(layer_id);
  out_dims[1]              = get_out_dim(layer_id);
  if (is_single_class)
    layers[layer_id] = new softmax_loss_layer(layer_id, in_dims, out_dims);
  else
    layers[layer_id] = new sigmoid_loss_layer(layer_id, in_dims, out_dims);
  connect(layers[layer_id - 1], layers[layer_id]);
}

//! Add a convolution layer to the network
void Net::append_conv_layer(size_t layer_id, bool act, bool norm, bool bias,
                            bool dropout, float_t dropout_rate) {
  assert(dropout_rate < 1.0);
  assert(layer_id < NUM_CONV_LAYERS);
  std::vector<size_t> in_dims(2), out_dims(2);
  in_dims[0] = out_dims[0] = num_samples;
  in_dims[1]               = get_in_dim(layer_id);
  out_dims[1]              = get_out_dim(layer_id);
  layers[layer_id] = new graph_conv_layer(layer_id, act, norm, bias, dropout,
                                          dropout_rate, in_dims, out_dims);
  if (layer_id > 0) connect(layers[layer_id - 1], layers[layer_id]);
}

#ifdef CPU_ONLY
/**
 *
 * @param begin GLOBAL begin
 * @param end GLOBAL end
 * @param count GLOBAL training count
 */
acc_t Net::masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, Graph* dGraph) {
#ifndef GALOIS_USE_DIST
  AccumF accuracy_all;
#else
  AccuracyAccum accuracy_all;
  galois::DGAccumulator<uint32_t> sampleCount;
  sampleCount.reset();
#endif

  accuracy_all.reset();

  galois::do_all(galois::iterate(begin, end), [&](const auto& i) {
#ifndef GALOIS_USE_DIST
    if (masks[i] == 1) {
      // get prediction
      int preds = argmax(num_classes,
      	    &(layers[NUM_CONV_LAYERS - 1]->next()->get_data()[i * num_classes]));
      // check prediction
      if ((label_t)preds == context->get_label(i))
        accuracy_all += 1.0;
    }
#else
    // only look at owned nodes (i.e. masters); the prediction for these
    // should only by handled on the owner
    if (dGraph->isOwned(i)) {
      sampleCount += 1;

      uint32_t localID = dGraph->getLID(i);
      if (masks[localID] == 1) {
        // get prediction
        int preds = argmax(num_classes,
        	    &(layers[NUM_CONV_LAYERS - 1]->next()->get_data()[localID * num_classes]));
        // check prediction
        if ((label_t)preds == context->get_label(localID))
          accuracy_all += 1.0;
      }
    }
#endif
  }, galois::loopname("getMaskedLoss"));

#ifdef GALOIS_USE_DIST
  count = sampleCount.reduce();
  galois::gDebug("sample count is ", count);
#endif

  // all hosts should get same accuracy
  return accuracy_all.reduce() / (acc_t)count;
}

acc_t Net::masked_multi_class_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, Graph* dGraph) {
  auto preds = layers[NUM_CONV_LAYERS - 1]->next()->get_data();
  auto ground_truth = context->get_labels_ptr();
  return deepgalois::masked_f1_score(begin, end, count, masks, num_classes, ground_truth, preds);
}
#endif

} // namespace deepgalois
