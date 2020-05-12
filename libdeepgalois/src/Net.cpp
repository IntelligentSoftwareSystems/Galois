/**
 * Based on the net.hpp file from Caffe deep learning framework.
 */

#include "galois/Timer.h"
#include "galois/Galois.h"
#include "deepgalois/Net.h"
#include "deepgalois/math_functions.hh"

namespace deepgalois {

void Net::partitionInit(DGraph* graph, std::string dataset_str,
                        bool isSingleClassLabel) {
  this->dGraph      = graph;
  this->distContext = new deepgalois::DistContext();
  this->distContext->saveDistGraph(dGraph);
  this->distNumSamples = this->dGraph->size();

  // TODO self loop setup would have to be done before this during partitioning
  // or on master node only

  this->distContext->initializeSyncSubstrate();
  num_classes = this->distContext->read_labels(isSingleClassLabel, dataset_str);

  // std::cout << "Reading label masks ... ";
  this->distTrainMasks = new mask_t[this->distNumSamples];
  this->distValMasks   = new mask_t[this->distNumSamples];
  std::fill(this->distTrainMasks, this->distTrainMasks + this->distNumSamples,
            0);
  std::fill(this->distValMasks, this->distValMasks + this->distNumSamples, 0);

  if (dataset_str == "reddit") {
    // this->globalTrainBegin = 0;
    // this->globalTrainCount = 153431;
    // this->globalTrainEnd = this->globalTrainBegin + this->globalTrainCount;
    // this->globalValBegin = 153431;
    // this->globalValCount = 23831;
    // this->globalValEnd = this->globalValBegin + this->globalValCount;

    // find local ID from global ID, set if it exists
    for (size_t i = globalTrainBegin; i < globalTrainEnd; i++) {
      if (this->dGraph->isLocal(i)) {
        this->distTrainMasks[this->dGraph->getLID(i)] = 1;
      }
    }
    for (size_t i = globalValBegin; i < globalValEnd; i++) {
      if (this->dGraph->isLocal(i)) {
        this->distValMasks[this->dGraph->getLID(i)] = 1;
      }
    }
  } else {
    globalTrainCount = this->distContext->read_masks(
        dataset_str, "train", this->distNumSamples, globalTrainBegin,
        globalTrainEnd, this->distTrainMasks, this->dGraph);
    globalValCount = this->distContext->read_masks(
        dataset_str, "val", this->distNumSamples, globalValBegin, globalValEnd,
        this->distValMasks, this->dGraph);
  }

  // input feature dimension: D
  feature_dims[0] = this->distContext->read_features(dataset_str);

  feature_dims[num_conv_layers] = num_classes; // output embedding: E
  if (this->has_l2norm) {
    // l2 normalized embedding: E
    feature_dims[num_conv_layers + 1] = num_classes;
  }
  if (this->has_dense) {
    // MLP embedding: E
    feature_dims[num_layers - 1] = num_classes;
  }
  feature_dims[num_layers] = num_classes; // normalized output embedding: E
}

void Net::init() {
  if (subgraph_sample_size)
    sampler = new deepgalois::Sampler();
}

// add weight decay
void Net::regularize() {
  size_t layer_id = 0;
  auto n          = feature_dims[layer_id] * feature_dims[layer_id + 1];
  // TODO: parallel
  math::axpy(n, weight_decay, layers[layer_id]->get_weights_ptr(),
             layers[layer_id]->get_grads_ptr());
}

/**
 *
 * @param begin GLOBAL begin
 * @param end GLOBAL end
 * @param masks: GLOBAL masks
 * @param count GLOBAL training count
 */
acc_t Net::masked_accuracy(size_t begin, size_t end, size_t count,
                           mask_t* masks, float_t* preds,
                           label_t* ground_truth) {
  galois::DGAccumulator<acc_t> accuracy_all;
  galois::DGAccumulator<uint32_t> sampleCount;
  accuracy_all.reset();
  sampleCount.reset();

  // TODO figure this out for distributed case
  galois::do_all(
      galois::iterate(begin, end),
      [&](const auto& i) {
#ifndef GALOIS_USE_DIST
        if (masks == NULL ||
            masks[i] == 1) { // use sampled graph when masks is NULL
          // get prediction
          auto pred = math::argmax(num_classes, preds + i * num_classes);
          // check prediction
          if ((label_t)pred == ground_truth[i])
            accuracy_all += 1.0;
        }
#else
        // TODO dist subraph

        // only look at owned nodes (i.e. masters); the prediction for these
        // should only by handled on the owner
        if (this->dGraph->isOwned(i)) {
          sampleCount += 1;

          uint32_t localID = this->dGraph->getLID(i);
          if (masks == NULL) {
            // GALOIS_DIE("subgraphs not implemented for dist yet");
            // subgraph here: TODO
            auto pred =
                math::argmax(num_classes, &preds[localID * num_classes]);
            // check prediction
            if ((label_t)pred == ground_truth[localID])
              accuracy_all += 1.0;
          } else {
            if (masks[localID] == 1) {
              // get prediction
              auto pred =
                  math::argmax(num_classes, &preds[localID * num_classes]);
              // check prediction
              if ((label_t)pred == ground_truth[localID])
                accuracy_all += 1.0;
            }
          }
        }
#endif
      },
      galois::loopname("getMaskedLoss"));

  count = sampleCount.reduce();
  galois::gDebug("sample count is ", count);

  // all hosts should get same accuracy
  return accuracy_all.reduce() / (acc_t)count;
}

acc_t Net::masked_multi_class_accuracy(size_t begin, size_t end, size_t count,
                                       mask_t* masks, float_t* preds,
                                       label_t* ground_truth) {
  // TODO dist version
  return deepgalois::masked_f1_score(begin, end, count, masks, num_classes,
                                     ground_truth, preds);
}

} // namespace deepgalois
