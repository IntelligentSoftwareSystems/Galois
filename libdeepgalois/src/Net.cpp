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

  // load the training/val masks
  if (dataset_str == "reddit") {
    // find local ID from global ID, set if it exists
    for (size_t i = this->globalTrainBegin; i < this->globalTrainEnd; i++) {
      if (this->dGraph->isLocal(i)) {
        this->distTrainMasks[this->dGraph->getLID(i)] = 1;
      }
    }
    for (size_t i = this->globalValBegin; i < this->globalValEnd; i++) {
      if (this->dGraph->isLocal(i)) {
        this->distValMasks[this->dGraph->getLID(i)] = 1;
      }
    }
  } else {
    globalTrainCount = this->distContext->read_masks(
        dataset_str, "train", this->distNumSamples, this->globalTrainBegin,
        this->globalTrainEnd, this->distTrainMasks, this->dGraph);
    globalValCount = this->distContext->read_masks(
        dataset_str, "val", this->distNumSamples, this->globalValBegin,
        this->globalValEnd, this->distValMasks, this->dGraph);
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

void Net::allocateSubgraphsMasks(int num_subgraphs) {
  subgraphs_masks = new mask_t[distNumSamples * num_subgraphs];
}

// add weight decay
void Net::regularize() {
  size_t layer_id = 0;
  auto n          = feature_dims[layer_id] * feature_dims[layer_id + 1];
  // TODO: parallel
  math::axpy(n, weight_decay, layers[layer_id]->get_weights_ptr(),
             layers[layer_id]->get_grads_ptr());
}

void Net::read_test_masks(std::string dataset) {
  if (dataset == "reddit") {
    globalTestBegin = 177262;
    globalTestCount = 55703;
    globalTestEnd   = globalTestBegin + globalTestCount;
    for (size_t i = globalTestBegin; i < globalTestEnd; i++) {
      globalTestMasks[i] = 1;
    }
  } else {
    globalTestCount = graphTopologyContext->read_masks(
        "test", globalSamples, globalTestBegin, globalTestEnd, globalTestMasks);
  }
}

void Net::readDistributedTestMasks(std::string dataset) {
  distTestMasks = new mask_t[distNumSamples];
  if (dataset == "reddit") {
    globalTestBegin = 177262;
    globalTestCount = 55703;
    globalTestEnd   = globalTestBegin + globalTestCount;
    for (size_t i = globalTestBegin; i < globalTestEnd; i++) {
      if (dGraph->isLocal(i))
        distTestMasks[dGraph->getLID(i)] = 1;
    }
  } else {
    globalTestCount = distContext->read_masks(
        dataset, std::string("test"), globalSamples, globalTestBegin,
        globalTestEnd, distTestMasks, dGraph);
  }
}

/**
 * @param gBegin GLOBAL begin
 * @param gEnd GLOBAL end
 * @param gMasks: GLOBAL masks
 * @param gCount GLOBAL training count
 */
acc_t Net::masked_accuracy(size_t gBegin, size_t gEnd, size_t gCount,
                           mask_t* gMasks, float_t* preds,
                           label_t* localGroundTruth) {
  galois::DGAccumulator<acc_t> accuracy_all;
  galois::DGAccumulator<uint32_t> sampleCount;
  accuracy_all.reset();
  sampleCount.reset();

  galois::do_all(
      galois::iterate(gBegin, gEnd),
      [&](const auto& i) {
        // only look at owned nodes (i.e. masters); the prediction for these
        // should only by handled on the owner
        if (this->dGraph->isOwned(i)) {
          sampleCount += 1;

          uint32_t localID = this->dGraph->getLID(i);
          if (gMasks == NULL) {
            auto pred =
                math::argmax(num_classes, &preds[localID * num_classes]);
            // check prediction
            if ((label_t)pred == localGroundTruth[localID])
              accuracy_all += 1.0;
          } else {
            // TODO masks needs to be local id
            if (gMasks[localID] == 1) {
              // get prediction
              auto pred =
                  math::argmax(num_classes, &preds[localID * num_classes]);
              // check prediction
              if ((label_t)pred == localGroundTruth[localID])
                accuracy_all += 1.0;
            }
          }
        }
      },
      galois::loopname("getMaskedLoss"));

  gCount = sampleCount.reduce();
  galois::gDebug("Total sample count is ", gCount);

  // all hosts should get same accuracy
  return accuracy_all.reduce() / (acc_t)gCount;
}

acc_t Net::masked_multi_class_accuracy(size_t gBegin, size_t gEnd,
                                       size_t gCount, mask_t* gMasks,
                                       float_t* preds,
                                       label_t* localGroundTruth) {
  // TODO fix this
  if (galois::runtime::getSystemNetworkInterface().Num > 1) {
    GALOIS_DIE(
        "Multi-class accuracy not yet implemented for distributed setting\n");
  }

  return deepgalois::masked_f1_score(gBegin, gEnd, gCount, gMasks, num_classes,
                                     localGroundTruth, preds);
}

} // namespace deepgalois
