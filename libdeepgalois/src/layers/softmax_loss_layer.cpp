#include "deepgalois/layers/softmax_loss_layer.h"

namespace deepgalois {

#ifdef CPU_ONLY
softmax_loss_layer::softmax_loss_layer(unsigned level,
                                       std::vector<size_t> in_dims,
                                       std::vector<size_t> out_dims)
    : layer(level, in_dims, out_dims) {
  trainable_ = false;
  name_      = layer_type() + "_" + std::to_string(level);
  loss = new float_t[in_dims[0]]; // error for each sample
}

// TODO: need kernel fusion optimization
// 𝑦[i] = 𝑒^𝑥[i] / Σ 𝑒^𝑥[𝑘]
void softmax_loss_layer::forward_propagation(const float_t* in_data,
                                             float_t* out_data) {
  size_t len = input_dims[1];
  galois::do_all(galois::iterate(begin_, end_),
    [&](const auto& i) {
      if (masks_[i] == 1) { // masked
        // output is normalized input for this layer
        math::softmax(len, &in_data[len*i], &out_data[len*i]); // normalize using softmax
        // one hot encoded vector for the labels
        std::vector<acc_t> groundTruth(output_dims[1], 0.0); // ground truth
        groundTruth[context->get_label(i)] = 1.0;            // one-hot
        // loss calculation
        loss[i] = math::cross_entropy(len, &groundTruth[0], &out_data[len*i]);
      }
    }, galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
    galois::loopname("softmax-loss-fw"));

    // no sync required in distributed execution since no graph topology used
    // in this forward pass; only a post-process pretty much
}

void softmax_loss_layer::back_propagation(const float_t* in_data,
                                          const float_t* out_data,
                                          float_t* out_grad, float_t* in_grad) {
  // note: out_grad is ignored because it shouldn't exist (this is output layer)
  size_t len = layer::input_dims[1];
  galois::do_all(galois::iterate(layer::begin_, layer::end_),
    [&](const auto& i) {
      if (masks_[i] == 1) { // masked
        vec_t norm_grad(len);
        std::vector<acc_t> groundTruth(len, 0.0);
        groundTruth[context->get_label(i)] = 1.0;
        // use ground truth to determine derivative of cross entropy
        math::d_cross_entropy(len, &groundTruth[0], &out_data[len * i], &norm_grad[0]);
        // derviative softmax to gradient used in the next layer
        math::d_softmax(len, &in_data[len * i], &out_data[len * i],
                        &in_grad[len * i], &norm_grad[0]);
      }
    }, galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
    galois::loopname("softmax-loss-bw"));

  // no weight sync required: this is all local graph information
}

acc_t softmax_loss_layer::get_masked_loss() {
  assert(count_ > 0);
  AccumF total_loss;
  AccumU valid_sample_count;
  total_loss.reset();
  valid_sample_count.reset();
  galois::do_all(galois::iterate(layer::begin_, layer::end_),
    [&](const auto& i) {
      if (masks_[i]) {
        total_loss += loss[i];
        valid_sample_count += 1;
      }
    }, galois::chunk_size<256>(), galois::steal(),
    galois::loopname("getMaskedLoss"));
  assert(valid_sample_count.reduce() == count_);
  return total_loss.reduce() / (acc_t)count_;
}
#endif

} // namespace
