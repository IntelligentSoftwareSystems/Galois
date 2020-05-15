#include "deepgalois/layers/sigmoid_loss_layer.h"
#include "deepgalois/math_functions.hh"
#include "galois/Galois.h"

namespace deepgalois {

sigmoid_loss_layer::sigmoid_loss_layer(unsigned level,
                                       std::vector<size_t> in_dims,
                                       std::vector<size_t> out_dims)
    : layer(level, in_dims, out_dims) {
  trainable_ = false;
  name_      = layer_type() + "_" + std::to_string(level);
}

sigmoid_loss_layer::~sigmoid_loss_layer() { delete[] loss; }

void sigmoid_loss_layer::malloc_and_init() {
  loss = new float_t[input_dims[0]]; // error for each sample
}

inline label_t sigmoid_loss_layer::get_label(size_t i, size_t j) {
  // return context->get_label(i, j);
  return labels[i * input_dims[1] + j];
}

void sigmoid_loss_layer::forward_propagation(const float_t* in_data,
                                             float_t* out_data) {
  size_t featLen = input_dims[1];
  galois::do_all(
      galois::iterate(begin_, end_),
      [&](const auto& gid) {
        if (!use_mask || masks_[gid] == 1) { // masked
          // check if local to this host
          if (this->context->isLocal(gid)) {
            unsigned lid = this->context->getLID(gid);
            size_t idx   = featLen * lid;

            // output is normalized input for this layer
            math::sigmoid(featLen, &in_data[idx],
                          &out_data[idx]); // normalize using sigmoid

            // one hot encoded vector for the labels
            // TODO this is a bottleneck; big lock on memory allocator
            float_t* ground_truth = new float_t[featLen];
            for (size_t j = 0; j < featLen; j++)
              ground_truth[j] = (float_t)get_label(lid, j);
            // loss calculation
            this->loss[lid] =
                math::cross_entropy(featLen, ground_truth, &out_data[idx]);

            // TODO this is a bottleneck, lock on memory possibly
            delete[] ground_truth;
          }
        }
      },
      galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
      galois::loopname("sigmoid-loss-fw"));
}

void sigmoid_loss_layer::back_propagation(const float_t* in_data,
                                          const float_t* out_data, float_t*,
                                          float_t* in_grad) {
  size_t featLen = layer::input_dims[1];

  galois::do_all(
      galois::iterate(layer::begin_, layer::end_),
      [&](const auto& gid) {
        if (!use_mask || masks_[gid] == 1) { // masked
          if (this->context->isLocal(gid)) {
            unsigned lid = this->context->getLID(gid);

            size_t idx = featLen * lid;
            // TODO this is bad
            float_t* norm_grad    = new float_t[featLen];
            float_t* ground_truth = new float_t[featLen];
            for (size_t j = 0; j < featLen; j++)
              ground_truth[j] = (float_t)get_label(lid, j);
            // use ground truth to determine derivative of cross entropy
            math::d_cross_entropy(featLen, ground_truth, &out_data[idx],
                                  norm_grad);
            // derviative sigmoid to gradient used in the next layer
            math::d_sigmoid(featLen, &in_data[idx], &out_data[idx],
                            &in_grad[idx], norm_grad);
            // TODO this is bad
            delete[] norm_grad;
            delete[] ground_truth;
          }
        }
      },
      galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
      galois::loopname("sigmoid-loss-bw"));
}

acc_t sigmoid_loss_layer::get_prediction_loss() {
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
