/**
 * Based on the net.hpp file from Caffe deep learning framework.
 */

#include "galois/Timer.h"
#include "deepgalois/net.h"
#include "deepgalois/math_functions.hh"

namespace deepgalois {

#ifdef GALOIS_USE_DIST
void Net::dist_init(Graph* graph, std::string dataset_str) {
  dGraph = graph;
  context = new deepgalois::DistContext();
  num_samples = dGraph->size();
  context->saveGraph(dGraph);
  // TODO self loop setup?
  context->initializeSyncSubstrate();
  num_classes = context->read_labels();

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
    train_count = context->read_masks("train", num_samples, train_begin, train_end, train_masks, dGraph);
    val_count = context->read_masks("val", num_samples, val_begin, val_end, val_masks, dGraph);
  }

  feature_dims[0] = context->read_features(); // input feature dimension: D
  for (size_t i = 1; i < num_conv_layers; i++)
    feature_dims[i] = hidden1;                           // hidden1 level embedding: 16
  feature_dims[num_conv_layers] = num_classes;           // output embedding: E
  if (has_l2norm) 
      feature_dims[num_conv_layers+1] = num_classes;     // l2 normalized embedding: E
  if (has_dense) 
      feature_dims[num_layers-1] = num_classes;          // MLP embedding: E
  feature_dims[num_layers] = num_classes;                // normalized output embedding: E
  layers.resize(num_layers);
}
#endif

#ifdef CPU_ONLY
void Net::init() {
  if (subgraph_sample_size) sampler = new deepgalois::Sampler();
}

// add weight decay
void Net::regularize() {
  size_t layer_id = 0;
  auto n = feature_dims[layer_id] * feature_dims[layer_id+1];
  // TODO: parallel
  math::axpy(n, weight_decay, layers[layer_id]->get_weights_ptr(), 
    layers[layer_id]->get_grads_ptr());
}

// Scale gradient to counterbalance accumulation
void Net::normalize() {
}

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
        auto pred = math::argmax(num_classes, &preds[localID*num_classes]);
        // check prediction
        if ((label_t)pred == ground_truth[localID])
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
