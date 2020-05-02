/**
 * Based on the net.hpp file from Caffe deep learning framework.
 */

#include "galois/Timer.h"
#include "deepgalois/net.h"
#include "deepgalois/utils.h"
#include "deepgalois/math_functions.hh"

namespace deepgalois {

void Net::init(std::string dataset_str, unsigned num_conv, unsigned epochs,
               unsigned hidden1, float lr, float dropout, float wd,
               bool selfloop, bool single, bool l2norm, bool dense, 
               unsigned neigh_sz, unsigned subg_sz) {
  assert(num_conv > 0);
  num_conv_layers = num_conv;
  num_epochs = epochs;
  learning_rate = lr;
  dropout_rate = dropout;
  weight_decay = wd;
  is_single_class = single;
  has_l2norm = l2norm;
  has_dense = dense;
  neighbor_sample_size = neigh_sz;
  subgraph_sample_size = subg_sz;
  val_interval = 1;
  galois::gPrint("Configuration: num_conv_layers ", num_conv_layers,
                 ", num_epochs ", num_epochs,
                 ", hidden1 ", hidden1,
                 ", learning_rate ", learning_rate,
                 ", dropout_rate ", dropout_rate,
                 ", weight_decay ", weight_decay, "\n");
#ifndef GALOIS_USE_DIST
  context = new deepgalois::Context();
  num_samples = context->read_graph(dataset_str, selfloop);
  context->set_label_class(is_single_class);
#endif

  // read graph, get num nodes
  num_classes = context->read_labels(dataset_str);

#ifndef GALOIS_USE_DIST
  //std::cout << "Reading label masks ... ";
  train_masks = new mask_t[num_samples];
  val_masks = new mask_t[num_samples];
  std::fill(train_masks, train_masks+num_samples, 0);
  std::fill(val_masks, val_masks+num_samples, 0);

  // get training and validation sets
  if (dataset_str == "reddit") {
    train_begin = 0, train_count = 153431,
    train_end = train_begin + train_count;
    val_begin = 153431, val_count = 23831, val_end = val_begin + val_count;
    // TODO do all can be used below
    for (size_t i = train_begin; i < train_end; i++) train_masks[i] = 1;
    for (size_t i = val_begin; i < val_end; i++) val_masks[i] = 1;
  } else {
    train_count = context->read_masks(dataset_str, "train", num_samples, train_begin, train_end, train_masks);
    val_count = context->read_masks(dataset_str, "val", num_samples, val_begin, val_end, val_masks);
  }
#endif

  if (subgraph_sample_size > train_count) {
    galois::gPrint("FATAL: subgraph size can not be larger than the size of training set\n");
    exit(1);
  }
  // NOTE: train_begin/train_end are global IDs, train_masks is a local id
  // train count and val count are LOCAL counts

  num_layers = num_conv_layers + 1;
  if (has_l2norm) num_layers ++;
  if (has_dense) num_layers ++;
  // initialize feature metadata
  feature_dims.resize(num_layers + 1);
  feature_dims[0] = context->read_features(dataset_str); // input feature dimension: D
  for (size_t i = 1; i < num_conv_layers; i++)
    feature_dims[i] = hidden1;                           // hidden1 level embedding: 16
  feature_dims[num_conv_layers] = num_classes;           // output embedding: E
  if (has_l2norm) 
      feature_dims[num_conv_layers+1] = num_classes;     // l2 normalized embedding: E
  if (has_dense) 
      feature_dims[num_layers-1] = num_classes;          // MLP embedding: E
  feature_dims[num_layers] = num_classes;                // normalized output embedding: E
  layers.resize(num_layers);

#ifdef CPU_ONLY
  context->set_use_subgraph(subgraph_sample_size > 0);
  if (subgraph_sample_size) sampler = new deepgalois::Sampler();
#else
  copy_masks_device(num_samples, train_masks, d_train_masks);
  copy_masks_device(num_samples, val_masks, d_val_masks);
  context->copy_data_to_device(); // copy labels and input features to the device
#endif
}

#ifdef GALOIS_USE_DIST
void Net::dist_init(Graph* graph) {
  dGraph = graph;
  context = new deepgalois::DistContext();
  num_samples = dGraph->size();
  context->saveGraph(dGraph);
  // TODO self loop setup?
  context->initializeSyncSubstrate();

  //std::cout << "Reading label masks ... ";
  train_masks = new mask_t[num_samples];
  val_masks = new mask_t[num_samples];
  std::fill(train_masks, train_masks+num_samples, 0);
  std::fill(val_masks, val_masks+num_samples, 0);

  if (dataset_str == "reddit") {
    train_begin = 0, train_count = 153431,
    train_end = train_begin + train_count;
    val_begin = 153431, val_count = 23831, val_end = val_begin + val_count;
    // find local ID from global ID, set if it exists
    for (size_t i = train_begin; i < train_end; i++) {
      if (dGraph->isLocal(i)) {
        train_masks[dGraph->getLID(i)] = 1;
      }
    }
    for (size_t i = val_begin; i < val_end; i++) {
      if (dGraph->isLocal(i)) {
        val_masks[dGraph->getLID(i)] = 1;
      }
    }
  } else {
    train_count = context->read_masks(dataset_str, "train", num_samples, train_begin, train_end, train_masks, dGraph);
    val_count = context->read_masks(dataset_str, "val", num_samples, val_begin, val_end, val_masks, dGraph);
  }
}
#endif

void Net::train(optimizer* opt, bool need_validate) {
  std::string header = "";
  std::string seperator = " ";
#ifdef GALOIS_USE_DIST
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
  header = "[" + std::to_string(myID) + "] ";
  seperator = "\n";
#endif

  galois::StatTimer Tupdate("Train-WeightUpdate");
  galois::StatTimer Tfw("Train-Forward");
  galois::StatTimer Tbw("Train-Backward");
  galois::StatTimer Tval("Validation");
  double total_train_time = 0.0;

  int num_subg_remain = 0;
#ifdef CPU_ONLY
  if (subgraph_sample_size) {
    galois::gPrint("\nConstruct training vertex set induced graph...\n");
    subgraph_masks = new mask_t[num_samples];
    sampler->set_masked_graph(train_begin, train_end, train_count, train_masks, context->getGraphPointer());
  }
#endif
  galois::gPrint("\nStart training...\n");
  Timer t_epoch;
  // run epochs
  for (unsigned ep = 0; ep < num_epochs; ep++) {
    galois::gPrint(header, "Epoch ", std::setw(3), ep, seperator);
    t_epoch.Start();

    if (subgraph_sample_size && num_subg_remain == 0) {
      for (size_t i = 0; i < num_layers; i++) layers[i]->update_dim_size(subgraph_sample_size);
#ifdef CPU_ONLY
      // generate subgraph
      context->createSubgraph();
      auto subgraph_ptr = context->getSubgraphPointer();
      sampler->subgraph_sample(subgraph_sample_size, *(subgraph_ptr), subgraph_masks);
      context->norm_factor_computing(1);
      for (size_t i = 0; i < num_conv_layers; i++) {
        layers[i]->set_graph_ptr(context->getSubgraphPointer());
        layers[i]->set_norm_consts_ptr(context->get_norm_factors_subg_ptr());
	  }
      // update labels for subgraph
      context->gen_subgraph_labels(subgraph_sample_size, subgraph_masks);
      layers[num_layers-1]->set_labels_ptr(context->get_labels_subg_ptr());

      // update features for subgraph
      context->gen_subgraph_feats(subgraph_sample_size, subgraph_masks);
      layers[0]->set_feats_ptr(context->get_feats_subg_ptr()); // feed input data
#endif
      num_subg_remain += 1; // num_threads
    }
    // training steps
    set_netphases(net_phase::train);
    acc_t train_loss = 0.0, train_acc = 0.0;

    // forward: after this phase, layer edges will contain intermediate features
    // for use during backprop
    Tfw.start();
    double fw_time = evaluate("train", train_loss, train_acc);
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
    if (need_validate && ep % val_interval == 0) {
      if (subgraph_sample_size) { // switch to the original graph
        for (size_t i = 0; i < num_layers; i++) layers[i]->update_dim_size(num_samples);
#ifdef CPU_ONLY
        for (size_t i = 0; i < num_conv_layers; i++) {
          layers[i]->set_graph_ptr(context->getGraphPointer());
          layers[i]->set_norm_consts_ptr(context->get_norm_factors_ptr());
	    }
        layers[num_layers-1]->set_labels_ptr(context->get_labels_ptr());
        layers[0]->set_feats_ptr(context->get_feats_ptr()); // feed input data
#endif
      }
      // Validation
      acc_t val_loss = 0.0, val_acc = 0.0;
      Tval.start();
      double val_time = evaluate("val", val_loss, val_acc);
      Tval.stop();
      galois::gPrint(header, "val_loss ", std::setprecision(3), std::fixed, val_loss,
                     " val_acc ", val_acc, seperator);
      galois::gPrint(header, "time ", std::setprecision(3), std::fixed, epoch_time + val_time, 
                     " ms (train_time ", epoch_time, " val_time ", val_time, ")\n");
    } else {
      galois::gPrint(header, "train_time ", std::fixed, epoch_time, 
                     " ms (fw ", fw_time, ", bw ", epoch_time - fw_time, ")\n");
    }
  }
  double avg_train_time = total_train_time / (double)num_epochs;
  double throughput = 1000.0 * (double)num_epochs / total_train_time;
  galois::gPrint("\nAverage training time: ", avg_train_time, 
                 " ms. Throughput: ", throughput, " epoch/s\n");
}

// evaluate, i.e. inference or predict
double Net::evaluate(std::string type, acc_t& loss, acc_t& acc) {
  // TODO may need to do something for the dist case
  Timer t_eval;
  t_eval.Start();
  size_t begin = 0, end = 0, count = 0;
  mask_t* masks = NULL;
  if (type == "train") {
    begin = train_begin;
    end = train_end;
    count = train_count;
    masks = train_masks;
    if (subgraph_sample_size) {
      // update masks for subgraph
      masks = NULL;
      begin = 0;
      end = subgraph_sample_size;
      count = subgraph_sample_size;
    }
  } else if (type == "val") {
    begin = val_begin;
    end = val_end;
    count = val_count;
    masks = val_masks;
  } else {
    begin = test_begin;
    end = test_end;
    count = test_count;
    masks = test_masks;
  }
#ifndef CPU_ONLY
  if (type == "train") {
    masks = d_train_masks;
  } else if (type == "val") {
    masks = d_val_masks;
  } else {
    masks = d_test_masks;
  }
#endif

  loss = fprop(begin, end, count, masks);
  float_t* predictions = layers[num_layers - 1]->next()->get_data();
  label_t* labels;
  if (subgraph_sample_size) {
    labels = context->get_labels_subg_ptr();
  } else {
    labels = context->get_labels_ptr();
  }
  if (is_single_class) {
    acc = masked_accuracy(begin, end, count, masks, predictions, labels);
  } else {
    acc = masked_multi_class_accuracy(begin, end, count, masks, predictions, labels);
  }
  t_eval.Stop();
  return t_eval.Millisecs();
}

//! forward propagation: [begin, end) is the range of samples used.
//! calls "forward" on the layers of the network and returns the loss of the
//! final layer
acc_t Net::fprop(size_t begin, size_t end, size_t count, mask_t* masks) {
  // set mask for the last layer
  layers[num_layers - 1]->set_sample_mask(begin, end, count, masks);
  // layer0: from N x D to N x 16
  // layer1: from N x 16 to N x E
  // layer2: from N x E to N x E (normalize only)
  for (size_t i = 0; i < num_layers; i++) {
    layers[i]->forward();
    // TODO need to sync model between layers here
  }
  // prediction error
  auto loss = layers[num_layers - 1]->get_prediction_loss();
  // Squared Norm Regularization to mitigate overfitting
  loss += weight_decay * layers[0]->get_weight_decay_loss();
  return loss;
}

void Net::bprop() {
  for (size_t i = num_layers; i != 0; i--) {
    layers[i - 1]->backward();
  }
}

// Scale gradient to counterbalance accumulation
void Net::normalize() {
}

// add weight decay
void Net::regularize() {
  size_t layer_id = 0;
  auto n = feature_dims[layer_id] * feature_dims[layer_id+1];
#ifdef CPU_ONLY
  // TODO: parallel
  math::axpy(n, weight_decay, layers[layer_id]->get_weights_ptr(), 
    layers[layer_id]->get_grads_ptr());
#else
  axpy_gpu(n, weight_decay, layers[layer_id]->get_weights_device_ptr(), 
    layers[layer_id]->get_grads_device_ptr());
#endif
}

void Net::update_weights(optimizer* opt) {
  normalize();
  regularize();
  for (size_t i = 0; i < num_layers; i++) {
    if (layers[i]->trainable()) {
      layers[i]->update_weight(opt);
    }
  }
}

void Net::construct_layers() {
  // append conv layers
  std::cout << "\nConstructing layers...\n";
  for (size_t i = 0; i < num_conv_layers-1; i++)
    append_conv_layer(i, true);                  // conv layers, act=true
  append_conv_layer(num_conv_layers-1);          // the last hidden layer, act=false
  if (has_l2norm)
    append_l2norm_layer(num_conv_layers);        // l2_norm layer
  if (has_dense)
    append_dense_layer(num_layers-2);            // dense layer
  append_out_layer(num_layers-1);                // output layer

  // allocate memory for intermediate features and gradients
  for (size_t i = 0; i < num_layers; i++) {
    layers[i]->add_edge();
  }
  for (size_t i = 1; i < num_layers; i++)
    connect(layers[i - 1], layers[i]);
  for (size_t i = 0; i < num_layers; i++)
    layers[i]->malloc_and_init();
  layers[0]->set_in_data(context->get_feats_ptr()); // feed input data
  // precompute the normalization constant based on graph structure
  context->norm_factor_computing(0);
  for (size_t i = 0; i < num_conv_layers; i++)
    layers[i]->set_norm_consts_ptr(context->get_norm_factors_ptr());
  set_contexts();
}

//! Add an l2_norm layer to the network
void Net::append_l2norm_layer(size_t layer_id) {
  assert(layer_id > 0); // can not be the first layer
  std::vector<size_t> in_dims(2), out_dims(2);
  in_dims[0]       = num_samples;
  in_dims[0]       = num_samples;
  in_dims[1]       = get_in_dim(layer_id);
  out_dims[1]      = get_out_dim(layer_id);
  layers[layer_id] = new l2_norm_layer(layer_id, in_dims, out_dims);
}

//! Add an dense layer to the network
void Net::append_dense_layer(size_t layer_id) {
  assert(layer_id > 0); // can not be the first layer
  std::vector<size_t> in_dims(2), out_dims(2);
  in_dims[0]       = num_samples;
  in_dims[0]       = num_samples;
  in_dims[1]       = get_in_dim(layer_id);
  out_dims[1]      = get_out_dim(layer_id);
  //layers[layer_id] = new dense_layer(layer_id, in_dims, out_dims);
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
  layers[layer_id]->set_labels_ptr(context->get_labels_ptr());
}

//! Add a convolution layer to the network
void Net::append_conv_layer(size_t layer_id, bool act, bool norm, bool bias,
                            bool dropout) {
  assert(dropout_rate < 1.0);
  assert(layer_id < num_conv_layers);
  std::vector<size_t> in_dims(2), out_dims(2);
  in_dims[0] = out_dims[0] = num_samples;
  in_dims[1]               = get_in_dim(layer_id);
  out_dims[1]              = get_out_dim(layer_id);
  layers[layer_id] = new graph_conv_layer(layer_id, act, norm, bias, dropout,
                                          dropout_rate, in_dims, out_dims);
  layers[layer_id]->set_graph_ptr(context->getGraphPointer());
}

void Net::read_test_masks(std::string dataset) {
  test_masks = new mask_t[num_samples];
  if (dataset == "reddit") {
    test_begin = 177262;
    test_count = 55703;
    test_end   = test_begin + test_count;
#ifndef GALOIS_USE_DIST
    for (size_t i = test_begin; i < test_end; i++) test_masks[i] = 1;
#else
    for (size_t i = test_begin; i < test_end; i++)  {
      if (dGraph->isLocal(i)) {
        test_masks[dGraph->getLID(i)] = 1;
      }
    }
#endif
  } else {
#ifndef GALOIS_USE_DIST
    test_count = context->read_masks(dataset, "test", num_samples, test_begin, test_end, test_masks);
#else
    test_count = context->read_masks(dataset, "test", num_samples, test_begin, test_end, test_masks, dGraph);
#endif
  }
#ifndef CPU_ONLY
  copy_masks_device(num_samples, test_masks, d_test_masks);
#endif
}

#ifdef CPU_ONLY
/**
 *
 * @param begin GLOBAL begin
 * @param end GLOBAL end
 * @param count GLOBAL training count
 */
acc_t Net::masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, float_t* preds, label_t* ground_truth) {
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
    if (masks == NULL || masks[i] == 1) { // use sampled graph when masks is NULL
      // get prediction
      auto pred = math::argmax(num_classes, preds+i*num_classes);
      // check prediction
      if ((label_t)pred == ground_truth[i])
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
        auto preds = math::argmax(num_classes, preds+localID*num_classes);
        // check prediction
        if ((label_t)preds == ground_truth[localID])
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

acc_t Net::masked_multi_class_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, float_t* preds, label_t* ground_truth) {
  return deepgalois::masked_f1_score(begin, end, count, masks, num_classes, ground_truth, preds);
}
#endif

} // namespace deepgalois
