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
  size_t len = input_dims[1];
  galois::do_all(
      galois::iterate(begin_, end_),
      [&](const auto& i) {
        if (!use_mask || masks_[i] == 1) { // masked
          size_t idx = len * i;
          // output is normalized input for this layer
          math::sigmoid(len, &in_data[idx],
                        &out_data[idx]); // normalize using sigmoid
          // one hot encoded vector for the labels
          float_t* ground_truth = new float_t[len];
          for (size_t j = 0; j < len; j++)
            ground_truth[j] = (float_t)get_label(i, j);
          // loss calculation
          loss[i] = math::cross_entropy(len, ground_truth, &out_data[idx]);
          delete[] ground_truth;
        }
      },
      galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
      galois::loopname("sigmoid-loss-fw"));
}

void sigmoid_loss_layer::back_propagation(const float_t* in_data,
                                          const float_t* out_data, float_t*,
                                          float_t* in_grad) {
  size_t len = layer::input_dims[1];
  galois::do_all(
      galois::iterate(layer::begin_, layer::end_),
      [&](const auto& i) {
        if (!use_mask || masks_[i] == 1) { // masked
          size_t idx            = len * i;
          float_t* norm_grad    = new float_t[len];
          float_t* ground_truth = new float_t[len];
          for (size_t j = 0; j < len; j++)
            ground_truth[j] = (float_t)get_label(i, j);
          // use ground truth to determine derivative of cross entropy
          math::d_cross_entropy(len, ground_truth, &out_data[idx], norm_grad);
          // derviative sigmoid to gradient used in the next layer
          math::d_sigmoid(len, &in_data[idx], &out_data[idx], &in_grad[idx],
                          norm_grad);
          delete[] norm_grad;
          delete[] ground_truth;
        }
      },
      galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
      galois::loopname("sigmoid-loss-bw"));
}

acc_t sigmoid_loss_layer::get_prediction_loss() {
  assert(count_ > 0);
  galois::GAccumulator<acc_t> total_loss;
  galois::GAccumulator<size_t> valid_sample_count;
  total_loss.reset();
  valid_sample_count.reset();
  galois::do_all(
      galois::iterate(layer::begin_, layer::end_),
      [&](const auto& i) {
        if (!use_mask || masks_[i]) {
          total_loss += loss[i];
          valid_sample_count += 1;
        }
      },
      galois::chunk_size<256>(), galois::steal(),
      galois::loopname("getMaskedLoss"));
  assert(valid_sample_count.reduce() == count_);
  return total_loss.reduce() / (acc_t)count_;
}

} // namespace deepgalois
