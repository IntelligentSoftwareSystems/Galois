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
#include "deepgalois/layers/sigmoid_loss_layer.h"
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
  void init(std::string dataset_str, unsigned epochs, unsigned hidden1, 
            bool selfloop, bool is_single = true);
  #else
  void init(std::string dataset_str, unsigned epochs, unsigned hidden1,
            bool selfloop, Graph* dGraph);
  #endif
  size_t get_in_dim(size_t layer_id) { return feature_dims[layer_id]; }
  size_t get_out_dim(size_t layer_id) { return feature_dims[layer_id + 1]; }
  size_t get_nnodes() { return num_samples; }
  void construct_layers();
  void append_out_layer(size_t layer_id);
  void train(optimizer* opt, bool need_validate); // training
  double evaluate(size_t begin, size_t end, size_t count, 
                  mask_t* masks, acc_t& loss, acc_t& acc); // inference

  //! Add a convolution layer to the network
  void append_conv_layer(size_t layer_id, bool act = false, bool norm = true,
                         bool bias = false, bool dropout = true,
                         float_t dropout_rate = 0.5);

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

protected:
#ifndef GALOIS_USE_DIST
  deepgalois::Context* context;
#else
  deepgalois::DistContext* context;
#endif
  bool is_single_class;             // single-class (one-hot) or multi-class label
  size_t num_samples;               // number of samples: N
  size_t num_classes;               // number of vertex classes: E
  size_t num_layers;                // for now hard-coded: NUM_CONV_LAYERS + 1
  unsigned num_epochs;              // number of epochs

  std::vector<size_t> feature_dims; // feature dimnesions for each layer
  std::vector<mask_t> train_mask, val_mask; // masks for traning and validation
  size_t train_begin, train_end, train_count, val_begin, val_end, val_count;
  std::vector<layer*> layers; // all the layers in the neural network

  // comparing outputs with the ground truth (labels)
#ifdef CPU_ONLY
  acc_t masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, Graph* dGraph);
  acc_t masked_multi_class_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, Graph* dGraph);
#else
  acc_t masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, CSRGraph *gGraph);
  acc_t masked_multi_class_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, CSRGraph *gGraph);
#endif
};

} // namespace deepgalois

#endif
