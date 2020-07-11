/**
 * Based on the net.hpp file from Caffe deep learning framework.
 */
#pragma once
#include <random>
#include "deepgalois/types.h"
#include "deepgalois/layers/l2_norm_layer.h"
#include "deepgalois/layers/graph_conv_layer.h"
#include "deepgalois/layers/softmax_loss_layer.h"
#include "deepgalois/layers/sigmoid_loss_layer.h"
#include "deepgalois/optimizer.h"
#include "deepgalois/utils.h"
#include "deepgalois/Context.h"
#include "deepgalois/GraphTypes.h"
#include "deepgalois/DistContext.h"
#include "deepgalois/Sampler.h"

namespace deepgalois {

// N: number of vertices, D: feature vector dimentions,
// E: number of distinct labels, i.e. number of vertex classes
// layer 1: features N x D, weights D x 16, out N x 16 (hidden1=16)
// layer 2: features N x 16, weights 16 x E, out N x E
class Net {
  std::string header;
  bool is_single_class;          // single-class (one-hot) or multi-class label
  bool has_l2norm;               // whether the net contains an l2_norm layer
  bool has_dense;                // whether the net contains an dense layer
  unsigned neighbor_sample_size; // neighbor sampling
  unsigned subgraph_sample_size; // subgraph sampling
  int num_threads;               // number of threads
  size_t globalSamples;          // number of samples: N
  size_t distNumSamples;         // number of samples: N
  size_t num_classes;            // number of vertex classes: E
  size_t num_conv_layers;        // number of convolutional layers
  size_t num_layers;             // total number of layers (conv + output)
  int num_epochs;                // number of epochs
  unsigned h1;                   // hidden layer size
  float learning_rate;           // learning rate
  float dropout_rate;            // dropout rate
  float weight_decay;            // weighti decay for over-fitting
  // begins/ends below are global ids
  size_t globalTrainBegin;
  size_t globalTrainEnd;
  size_t globalTrainCount;
  size_t globalValBegin;
  size_t globalValEnd;
  size_t globalValCount;
  size_t globalTestBegin;
  size_t globalTestEnd;
  size_t globalTestCount;
  int val_interval;
  int num_subgraphs;
  unsigned subgraphNumVertices;
  bool is_selfloop;

  mask_t* globalTrainMasks; // masks for training
  mask_t* globalValMasks;   // masks for validation
  mask_t* globalTestMasks;  // masks for test
  // TODO it's looking like we may not even need these dist versions
  mask_t* distTrainMasks;
  mask_t* distValMasks;
  mask_t* distTestMasks; // masks for test, dst

  mask_t* d_train_masks; // masks for training on device
  mask_t* d_val_masks;   // masks for validation on device
  mask_t* d_test_masks;  // masks for test on device

  mask_t* subgraphs_masks; // masks for subgraphs; size of local graph
  // masks for subgraphs on device; size of local graph
  mask_t* d_subgraphs_masks;
  std::vector<size_t> feature_dims; // feature dimnesions for each layer
  std::vector<layer*> layers;       // all the layers in the neural network

  // one context is for entire graph; other is for partitioned graph
  // TODO optimize single host case

  //! context holds all of the graph data
  deepgalois::Context* graphTopologyContext;

  //! dist context holds graph data of the partitioned graph only
  deepgalois::DistContext* distContext;
  DGraph* dGraph;
  Sampler* sampler;

public:
  //! Default net constructor
  Net()
      : Net("reddit", 1, 2, 200, 16, 0.01, 0.5, 5e-4, false, true, false, false,
            25, 9000, 1) {}

  //! Net constructor
  Net(std::string dataset_str, int nt, unsigned n_conv, int epochs,
      unsigned hidden1, float lr, float dropout, float wd, bool selfloop,
      bool single, bool l2norm, bool dense, unsigned neigh_sz, unsigned subg_sz,
      int val_itv);

  // allocate memory for subgraph masks
  void allocateSubgraphsMasks(int num_subgraphs);

  //! Initializes metadata for the partition: loads data, labels, etc
  void partitionInit(DGraph* graph, std::string dataset_str,
                     bool isSingleClassLabel);
  size_t get_in_dim(size_t layer_id) { return feature_dims[layer_id]; }
  size_t get_out_dim(size_t layer_id) { return feature_dims[layer_id + 1]; }
  void regularize(); // add weight decay
  void train(optimizer* opt, bool need_validate);
  double evaluate(std::string type, acc_t& loss, acc_t& acc);

  //! read masks of test set for GLOBAL set
  void read_test_masks(std::string dataset);
  //! read test masks only for local nodes; assumes dist context is initialized
  void readDistributedTestMasks(std::string dataset);

  // void copy_test_masks_to_device();
  void construct_layers();

  //! Add an l2_norm layer to the network
  void append_l2norm_layer(size_t layer_id);

  //! Add an dense layer to the network
  void append_dense_layer(size_t layer_id);

  //! Add an output layer to the network
  void append_out_layer(size_t layer_id);

  //! Add a convolution layer to the network
  void append_conv_layer(size_t layer_id, bool act = false, bool norm = true,
                         bool bias = false, bool dropout = true);

  // update trainable weights after back-prop
  void update_weights(optimizer* opt);

  // forward propagation
  acc_t fprop(size_t gBegin, size_t gEnd, size_t gCount, mask_t* gMasks);
  void bprop();                        // back propagation
  void set_contexts();                 // Save the context
  void set_netphases(net_phase phase); // current phase: train or test
  void print_layers_info();            // print layer information
  void print_configs();                // print the configurations

  // comparing outputs with the ground truth (labels)
  acc_t masked_accuracy(size_t gBegin, size_t gEnd, size_t gCount,
                        mask_t* gMasks, float_t* preds,
                        label_t* localGroundTruth);
  acc_t masked_multi_class_accuracy(size_t gBegin, size_t gEnd, size_t gCount,
                                    mask_t* gMasks, float_t* preds,
                                    label_t* localGroundTruth);
};

} // namespace deepgalois
