/**
 * Based on the net.hpp file from Caffe deep learning framework.
 */

#include "deepgalois/net.h"

namespace deepgalois {

#ifndef GALOIS_USE_DIST
void Net::init(std::string dataset_str, unsigned epochs, unsigned hidden1, bool selfloop) {
#else
void Net::init(std::string dataset_str, unsigned epochs, unsigned hidden1,
               bool selfloop, Graph* dGraph) {
#endif
#ifndef GALOIS_USE_DIST
  context = new deepgalois::Context();
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
  galois::gPrint("\nStart training...\n");
  galois::StatTimer Tupdate("Train-WeightUpdate");
  galois::StatTimer Tfw("Train-Forward");
  galois::StatTimer Tbw("Train-Backward");
  galois::StatTimer Tval("Validation");

  Timer t_epoch;
  // run epochs
  for (unsigned i = 0; i < num_epochs; i++) {
    galois::gPrint("Epoch ", std::setw(2), i, std::fixed, std::setprecision(3), ":");
    t_epoch.Start();

    // training steps
    set_netphases(net_phase::train);
    acc_t train_loss = 0.0, train_acc = 0.0;

    // forward: after this phase, layer edges will contain intermediate features
    // for use during backprop
    Tfw.start();
    train_loss =
        Net::fprop(train_begin, train_end, train_count, &train_mask[0]); // forward
    train_acc = masked_accuracy(train_begin, train_end, train_count,
                                &train_mask[0]); // predict
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
    galois::gPrint("train_loss = ", std::setw(5), train_loss, " train_acc = ",
                   std::setw(5), train_acc);
    t_epoch.Stop();
    double epoch_time = t_epoch.Millisecs();
    if (need_validate) {
      // Validation
      acc_t val_loss = 0.0, val_acc = 0.0;
      Tval.start();
      double val_time = evaluate(val_begin, val_end, val_count, &val_mask[0],
                                 val_loss, val_acc);
      Tval.stop();
      galois::gPrint(" val_loss = ", std::setw(5), val_loss, " val_acc = ",
                     std::setw(5), val_acc);
      galois::gPrint(" time = ", epoch_time + val_time, " ms (train_time = ",
                     epoch_time, " val_time = ", val_time, ")\n");
    } else {
      galois::gPrint(" train_time = ", epoch_time, " ms\n");
    }
  }
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

#ifdef CPU_ONLY
acc_t Net::masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks) {
  AccumF accuracy_all;
  accuracy_all.reset();
  galois::do_all(galois::iterate(begin, end), [&](const auto& i) {
    if (masks[i] == 1) {
      int preds = argmax(num_classes,
	    &(layers[NUM_CONV_LAYERS - 1]->next()->get_data()[i * num_classes]));
      if ((label_t)preds == context->get_label(i))
        accuracy_all += 1.0;
    }
  },
  galois::chunk_size<256>(), galois::steal(), galois::loopname("getMaskedLoss"));
  return accuracy_all.reduce() / (acc_t)count;
}
#endif

} // namespace deepgalois