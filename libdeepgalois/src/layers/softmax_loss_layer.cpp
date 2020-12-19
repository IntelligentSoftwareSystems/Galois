#include "deepgalois/layers/softmax_loss_layer.h"
#include "deepgalois/math_functions.hh"
#include "galois/Galois.h"

namespace deepgalois {

softmax_loss_layer::softmax_loss_layer(unsigned level,
                                       std::vector<size_t> in_dims,
                                       std::vector<size_t> out_dims)
    : layer(level, in_dims, out_dims) {
  trainable_ = false;
  name_      = layer_type() + "_" + std::to_string(level);
}

softmax_loss_layer::~softmax_loss_layer() { delete[] loss; }

void softmax_loss_layer::malloc_and_init() {
  loss = new float_t[input_dims[0]]; // error for each sample
}

inline label_t softmax_loss_layer::get_label(size_t i) {
  return labels[i];
  // return context->get_label(i);
}

// TODO: need kernel fusion optimization
// 𝑦[i] = 𝑒^𝑥[i] / Σ 𝑒^𝑥[𝑘]
void softmax_loss_layer::forward_propagation(const float_t* in_data,
                                             float_t* out_data) {
  // size_t numSamples = input_dims;
  size_t featLen = input_dims[1];
  // zero out the output vector
  for (unsigned i = 0; i < input_dims[0]; i++) {
    for (unsigned j = 0; j < featLen; j++) {
      out_data[i * featLen + j] = 0.0;
    }
  }

  galois::do_all(
      galois::iterate(begin_, end_),
      [&](const unsigned gid) {
        // if no mask used it means all are fair game
        if (!use_mask || masks_[gid] == 1) {
          if (this->context->isLocal(gid)) {
            unsigned lid = this->context->getLID(gid);
            // output is normalized input for this layer
            math::softmax(featLen, &in_data[featLen * lid],
                          &out_data[featLen * lid]); // normalize using softmax
            // one hot encoded vector for the labels
            vec_t groundTruth(output_dims[1], 0.0); // ground truth
            // labels are local
            groundTruth[get_label(lid)] = 1.0; // one-hot
            // loss calculation
            loss[lid] = math::cross_entropy(featLen, &groundTruth[0],
                                            &out_data[featLen * lid]);
          }
        }
      },
      galois::chunk_size<64>(), galois::steal(),
      galois::loopname("softmax-loss-fw"));

  // no sync required in distributed execution since no graph topology used
  // in this forward pass; only a post-process pretty much
}

void softmax_loss_layer::back_propagation(const float_t* in_data,
                                          const float_t* out_data, float_t*,
                                          float_t* in_grad) {
  // note: out_grad is ignored because it shouldn't exist (this is output layer)
  size_t featLen = layer::input_dims[1];

  for (unsigned i = 0; i < input_dims[0]; i++) {
    for (unsigned j = 0; j < featLen; j++) {
      in_grad[i * featLen + j] = 0.0;
    }
  }

  galois::do_all(
      galois::iterate(layer::begin_, layer::end_),
      [&](const auto& gid) {
        if (!use_mask || masks_[gid] == 1) { // masked
          if (this->context->isLocal(gid)) {
            unsigned lid = this->context->getLID(gid);
            vec_t norm_grad(featLen);
            std::vector<acc_t> groundTruth(featLen, 0.0);
            groundTruth[get_label(lid)] = 1.0;
            // use ground truth to determine derivative of cross entropy
            math::d_cross_entropy(featLen, &groundTruth[0],
                                  &out_data[featLen * lid], &norm_grad[0]);
            // derviative softmax to gradient used in the next layer
            math::d_softmax(featLen, &in_data[featLen * lid],
                            &out_data[featLen * lid], &in_grad[featLen * lid],
                            &norm_grad[0]);
          }
        }
      },
      galois::chunk_size<64>(), galois::steal(),
      galois::loopname("softmax-loss-bw"));

  // no weight sync required: this is all local graph information
}

acc_t softmax_loss_layer::get_prediction_loss() {
  galois::GAccumulator<acc_t> total_loss;
  galois::GAccumulator<size_t> valid_sample_count;
  total_loss.reset();
  valid_sample_count.reset();

  galois::do_all(
      galois::iterate(layer::begin_, layer::end_),
      [&](const auto& gid) {
        if (!use_mask || masks_[gid]) {
          if (this->context->isLocal(gid)) {
            unsigned lid = this->context->getLID(gid);
            total_loss += this->loss[lid];
            valid_sample_count += 1;
          }
        }
      },
      galois::chunk_size<256>(), galois::steal(),
      galois::loopname("getMaskedLoss"));

  size_t c = valid_sample_count.reduce();
  if (c > 0) {
    return total_loss.reduce() / (acc_t)valid_sample_count.reduce();
  } else {
    return 0;
  }
}

} // namespace deepgalois
