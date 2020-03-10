/**
 * Based on the net.hpp file from Caffe deep learning framework.
 */
#ifndef _MODEL_H_
#define _MODEL_H_

#include <random>
#include "galois/Timer.h"
#include "deepgalois/types.h"
#include "deepgalois/gtypes.h"
#include "deepgalois/layers/graph_conv_layer.h"
#include "deepgalois/layers/softmax_loss_layer.h"
#include "deepgalois/optimizer.h"
#ifndef GALOIS_USE_DIST
#include "deepgalois/context.h"
#else
#include "deepgalois/DistContext.h"
#endif



#define NUM_CONV_LAYERS 2

namespace deepgalois {

// N: number of vertices, D: feature vector dimentions,
// E: number of distinct labels, i.e. number of vertex classes
// layer 1: features N x D, weights D x 16, out N x 16 (hidden1=16)
// layer 2: features N x 16, weights 16 x E, out N x E
class Net {
public:
  Net() {}
  #ifndef GALOIS_USE_DIST
  void init(std::string dataset_str, unsigned epochs, unsigned hidden1, bool selfloop);
  #else
  void init(std::string dataset_str, unsigned epochs, unsigned hidden1,
            bool selfloop, Graph* dGraph);
  #endif
  size_t get_in_dim(size_t layer_id) { return feature_dims[layer_id]; }
  size_t get_out_dim(size_t layer_id) { return feature_dims[layer_id + 1]; }
  size_t get_nnodes() { return num_samples; }
  void train(optimizer* opt, bool need_validate); // training
  void construct_layers();
  //! Save the context object to all layers of the network
  void set_contexts() {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->set_context(context);
  }
  //! set netphases for all layers in this network
  void set_netphases(deepgalois::net_phase phase) {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->set_netphase(phase);
  }
  //! print all layers
  void print_layers_info() {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->print_layer_info();
  }

  //! Add a convolution layer to the network
  void append_conv_layer(size_t layer_id, bool act = false, bool norm = true,
                         bool bias = false, bool dropout = true,
                         float_t dropout_rate = 0.5) {
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

  //! Add an output layer to the network
  void append_out_layer(size_t layer_id) {
    assert(layer_id > 0); // can not be the first layer
    std::vector<size_t> in_dims(2), out_dims(2);
    in_dims[0] = out_dims[0] = num_samples;
    in_dims[1]               = get_in_dim(layer_id);
    out_dims[1]              = get_out_dim(layer_id);
    layers[layer_id] = new softmax_loss_layer(layer_id, in_dims, out_dims);
    connect(layers[layer_id - 1], layers[layer_id]);
  }

  //! forward propagation: [begin, end) is the range of samples used.
  //! calls "forward" on the layers of the network and returns the loss of the
  //! final layer
  acc_t fprop(size_t begin, size_t end, size_t count, mask_t* masks) {
    // set mask for the last layer
    layers[num_layers - 1]->set_sample_mask(begin, end, count, masks);
    // layer0: from N x D to N x 16
    // layer1: from N x 16 to N x E
    // layer2: from N x E to N x E (normalize only)
    for (size_t i = 0; i < num_layers; i++) {
      layers[i]->forward();
      // TODO need to sync model between layers here
    }
    return layers[num_layers - 1]->get_masked_loss();
  }

  // back propogation
  void bprop() {
    for (size_t i = num_layers; i != 0; i--) {
      layers[i - 1]->backward();
    }
  }

  // update trainable weights after back-propagation
  void update_weights(optimizer* opt) {
    for (size_t i = 0; i < num_layers; i++) {
      if (layers[i]->trainable()) {
        layers[i]->update_weight(opt);
      }
    }
  }

  // evaluate, i.e. inference or predict
  double evaluate(size_t begin, size_t end, size_t count, mask_t* masks,
                  acc_t& loss, acc_t& acc) {
    // TODO may need to do something for the dist case
    Timer t_eval;
    t_eval.Start();
    loss = fprop(begin, end, count, masks);
    acc  = masked_accuracy(begin, end, count, masks);
    t_eval.Stop();
    return t_eval.Millisecs();
  }

protected:
#ifndef GALOIS_USE_DIST
  deepgalois::Context* context;
#else
  deepgalois::DistContext* context;
#endif
  size_t num_samples;               // number of samples: N
  size_t num_classes;               // number of vertex classes: E
  size_t num_layers;                // for now hard-coded: NUM_CONV_LAYERS + 1
  unsigned num_epochs;              // number of epochs
  std::vector<size_t> feature_dims; // feature dimnesions for each layer
  std::vector<mask_t> train_mask, val_mask; // masks for traning and validation
  size_t train_begin, train_end, train_count, val_begin, val_end, val_count;
  std::vector<layer*> layers; // all the layers in the neural network
  // comparing outputs with the ground truth (labels)
  acc_t masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks);
};

} // namespace deepgalois

#endif
