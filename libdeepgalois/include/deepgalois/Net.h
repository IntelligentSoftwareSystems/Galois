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

#ifndef __GALOIS_HET_CUDA__
#include "deepgalois/Sampler.h"
#endif

namespace deepgalois {

// N: number of vertices, D: feature vector dimentions,
// E: number of distinct labels, i.e. number of vertex classes
// layer 1: features N x D, weights D x 16, out N x 16 (hidden1=16)
// layer 2: features N x 16, weights 16 x E, out N x E
class Net {
#ifdef __GALOIS_HET_CUDA__
  unsigned myID = 0;
#else
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
#endif
  std::string header    = "[" + std::to_string(myID) + "] ";
  std::string seperator = "\n";

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
  mask_t* distTrainMasks;
  mask_t* distValMasks;
  mask_t* test_masks; // masks for test

  mask_t* d_train_masks; // masks for training on device
  mask_t* d_val_masks;   // masks for validation on device
  mask_t* d_test_masks;  // masks for test on device

  mask_t* subgraphs_masks;          // masks for subgraphs; size of local graph
  std::vector<size_t> feature_dims; // feature dimnesions for each layer
  std::vector<layer*> layers;       // all the layers in the neural network

  // one context is for entire graph; other is for partitioned graph
  // TODO optimize single host case

  //! context holds all of the graph data
  deepgalois::Context* graphTopologyContext;

  //! dist context holds graph data of the partitioned graph only
  deepgalois::DistContext* distContext;
  DGraph* dGraph;

#ifndef __GALOIS_HET_CUDA__
  Sampler* sampler;
#endif

public:
  Net(std::string dataset_str, int nt, unsigned n_conv, int epochs,
      unsigned hidden1, float lr, float dropout, float wd, bool selfloop,
      bool single, bool l2norm, bool dense, unsigned neigh_sz, unsigned subg_sz, int val_itv)
      : is_single_class(single), has_l2norm(l2norm), has_dense(dense),
        neighbor_sample_size(neigh_sz), subgraph_sample_size(subg_sz),
        num_threads(nt), num_conv_layers(n_conv), num_epochs(epochs),
        h1(hidden1), learning_rate(lr), dropout_rate(dropout), weight_decay(wd),
        val_interval(val_itv), num_subgraphs(1), is_selfloop(selfloop) {
    // init some identifiers for this host
#ifndef __GALOIS_HET_CUDA__
    this->myID      = galois::runtime::getSystemNetworkInterface().ID;
#endif
    this->header    = "[" + std::to_string(myID) + "] ";
    this->seperator = " ";

    assert(n_conv > 0);

    // TODO use galois print
    std::cout << header << "Configuration: num_threads " << num_threads
              << ", num_conv_layers " << num_conv_layers << ", num_epochs "
              << num_epochs << ", hidden1 " << hidden1 << ", learning_rate "
              << learning_rate << ", dropout_rate " << dropout_rate
              << ", weight_decay " << weight_decay << "\n";
    this->num_layers = num_conv_layers + 1;

    // additional layers to add
    if (has_l2norm)
      this->num_layers++;
    if (has_dense)
      this->num_layers++;
    // initialize feature metadata
    feature_dims.resize(num_layers + 1);

    // initialze global graph context
    graphTopologyContext = new deepgalois::Context();
    graphTopologyContext->set_dataset(dataset_str);
    // read *entire* graph, get num nodes
    globalSamples = graphTopologyContext->read_graph(selfloop);
#ifdef __GALOIS_HET_CUDA__
    this->distContext = new deepgalois::DistContext();
    this->distContext->set_dataset(dataset_str);
    this->distNumSamples = this->distContext->read_graph(dataset_str, selfloop);
#endif

    // get training and validation sets: this is to create the training
    // subgraph in the sampler
    globalTrainMasks = new mask_t[globalSamples];
    globalValMasks   = new mask_t[globalSamples];
    std::fill(globalTrainMasks, globalTrainMasks + globalSamples, 0);
    std::fill(globalValMasks, globalValMasks + globalSamples, 0);

    // reddit is hard coded
    if (dataset_str == "reddit") {
      this->globalTrainBegin = 0;
      this->globalTrainCount = 153431;
      this->globalTrainEnd   = this->globalTrainBegin + this->globalTrainCount;
      this->globalValBegin   = 153431;
      this->globalValCount   = 23831;
      this->globalValEnd     = this->globalValBegin + this->globalValCount;

      // TODO do all can be used below
      for (size_t i = globalTrainBegin; i < globalTrainEnd; i++)
        globalTrainMasks[i] = 1;
      for (size_t i = globalValBegin; i < globalValEnd; i++)
        globalValMasks[i] = 1;
    } else {
      globalTrainCount = graphTopologyContext->read_masks(
          "train", globalSamples, globalTrainBegin, globalTrainEnd,
          globalTrainMasks);
      globalValCount = graphTopologyContext->read_masks(
          "val", globalSamples, globalValBegin, globalValEnd, globalValMasks);
    }

#ifndef __GALOIS_HET_CUDA__
    // make sure sampel size isn't greater than what we have to train with
    if (subgraph_sample_size > globalTrainCount) {
      GALOIS_DIE("subgraph size can not be larger than the size of training "
                 "set\n");
    }

    // features are read in distcontext, not this context (this context only
    // used for sampling)

    this->sampler = new Sampler();
#endif
  }

  //! Default net constructor
  // Net()
  //    : is_single_class(true), has_l2norm(false), has_dense(false),
  //      neighbor_sample_size(0), subgraph_sample_size(0), num_threads(1),
  //      globalSamples(0), num_classes(0), num_conv_layers(0), num_layers(0),
  //      num_epochs(0), learning_rate(0.0), dropout_rate(0.0),
  //      weight_decay(0.0), globalTrainBegin(0), globalTrainEnd(0),
  //      globalTrainCount(0), globalValBegin(0), globalValEnd(0),
  //      globalValCount(0), globalTestBegin(0), globalTestEnd(0),
  //      globalTestCount(0), val_interval(1), num_subgraphs(1),
  //      num_vertices_sg(9000), globalTrainMasks(NULL), globalValMasks(NULL),
  //      test_masks(NULL), context(NULL) {}

  void init();

  //! Initializes metadata for the partition
  void partitionInit(DGraph* graph, std::string dataset_str, bool isSingleClassLabel);
  size_t get_in_dim(size_t layer_id) { return feature_dims[layer_id]; }
  size_t get_out_dim(size_t layer_id) { return feature_dims[layer_id + 1]; }
  void regularize(); // add weight decay

  void train(optimizer* opt, bool need_validate) {
    double total_train_time = 0.0;
    int num_subg_remain     = 0;

    if (subgraph_sample_size) {
      distContext->allocateSubgraphs(num_subgraphs);
      subgraphs_masks = new mask_t[distNumSamples * num_subgraphs];
      std::cout << header << "Constructing training vertex set induced graph...\n";
#ifndef __GALOIS_HET_CUDA__
      sampler->initializeMaskedGraph(globalTrainCount, globalTrainMasks,
                                     graphTopologyContext->getGraphPointer(),
                                     distContext->getGraphPointer());
#endif
    }

    std::cout << header << "Start training...\n";

    Timer t_epoch;

    // run epochs
    for (int curEpoch = 0; curEpoch < num_epochs; curEpoch++) {
      t_epoch.Start();

      ////////////////////////////////////////////////////////////////////////////////
      // Sampling
      ////////////////////////////////////////////////////////////////////////////////
      if (subgraph_sample_size) {
        if (num_subg_remain == 0) {
          std::cout << header << "Generating " << num_subgraphs << " subgraph(s)\n";
          // TODO stat timer instead of this timer
          Timer t_subgen;
          t_subgen.Start();

          // generate subgraphs
#ifndef __GALOIS_HET_CUDA__
          for (int sid = 0; sid < num_subgraphs; sid++) {
            sampler->sampleSubgraph(
                subgraph_sample_size, *(distContext->getSubgraphPointer(sid)),
                &subgraphs_masks[sid * globalSamples], curEpoch);
          }
#endif
          num_subg_remain = num_subgraphs;
          t_subgen.Stop();
          // std::cout << "Done, time: " << t_subgen.Millisecs() << "\n";
        }
        // count their degrees
        for (int i = 0; i < num_subgraphs; i++) {
          auto sg_ptr = distContext->getSubgraphPointer(i);
          sg_ptr->degree_counting();
          // galois::gPrint("\tsubgraph[", i, "]: num_v ", sg_ptr->size(), "
          // num_e ", sg_ptr->sizeEdges(), "\n");
        }

        // choose a subgraph to use
        num_subg_remain--;
        int sg_id                 = num_subg_remain;
        auto subgraphPointer      = distContext->getSubgraphPointer(sg_id);
        this->subgraphNumVertices = subgraphPointer->size();

        // galois::gPrint("Subgraph num_vertices: ", subgraphNumVertices, ",
        // num_edges: ", subgraphPointer->sizeEdges(), "\n");
        for (size_t i = 0; i < num_layers; i++) {
          layers[i]->update_dim_size(this->subgraphNumVertices);
        }

        // TODO dist version where i need global degrees
        // change normalization constants
        distContext->constructNormFactorSub(sg_id);
        for (size_t i = 0; i < num_conv_layers; i++) {
          layers[i]->set_graph_ptr(subgraphPointer);
          layers[i]->set_norm_consts_ptr(
              distContext->get_norm_factors_subg_ptr());
        }

        // update labels for subgraph
        distContext->constructSubgraphLabels(
            this->subgraphNumVertices, &subgraphs_masks[sg_id * globalSamples]);
        layers[num_layers - 1]->set_labels_ptr(
            distContext->get_labels_subg_ptr());

        // update features for subgraph
        distContext->constructSubgraphFeatures(
            this->subgraphNumVertices, &subgraphs_masks[sg_id * globalSamples]);
        layers[0]->set_feats_ptr(
            distContext->get_feats_subg_ptr()); // feed input data

        // Graph* testing = distContext->getSubgraphPointer(sg_id);
        // for (size_t i = 0; i < testing->size(); i++) {
        //  for (auto j = testing->edge_begin(i); j < testing->edge_end(i); j++)
        //  {
        //    galois::gPrint(i, " ", testing->getEdgeDst(j), "\n");
        //  }
        //}
      } // end subgraph sample loop
      ////////////////////////////////////////////////////////////////////////////////

      // training steps
      std::cout << header << "Epoch " << std::setw(3) << curEpoch << seperator;
      set_netphases(net_phase::train);
      acc_t train_loss = 0.0, train_acc = 0.0;

      // forward: after this phase, layer edges will contain intermediate
      // features for use during backprop
      double fw_time = evaluate("train", train_loss, train_acc);

      // backward: use intermediate features + ground truth to update layers
      // with feature gradients whcih are then used to calculate weight
      // gradients
      Net::bprop();

      // gradient update: use gradients stored on each layer to update model for
      // next epoch
      Net::update_weights(opt); // update parameters

      // validation / testing
      set_netphases(net_phase::test);

      std::cout << header << "train_loss " << std::setprecision(3) << std::fixed
                << train_loss << " train_acc " << train_acc << seperator;

      t_epoch.Stop();

      double epoch_time = t_epoch.Millisecs();
      total_train_time += epoch_time;

      if (need_validate && curEpoch % val_interval == 0) {
        // Validation
        acc_t val_loss = 0.0, val_acc = 0.0;
        double val_time = evaluate("val", val_loss, val_acc);
        std::cout << header << "val_loss " << std::setprecision(3) << std::fixed
                  << val_loss << " val_acc " << val_acc << seperator;
        std::cout << header << "time " << std::setprecision(3) << std::fixed
                  << epoch_time + val_time << " ms (train_time " << epoch_time
                  << " val_time " << val_time << ")\n";
      } else {
        std::cout << header << "train_time " << std::fixed << epoch_time
                  << " ms (fw " << fw_time << ", bw " << epoch_time - fw_time
                  << ")\n";
      }
    } // epoch loop

    double avg_train_time = total_train_time / (double)num_epochs;
    double throughput     = 1000.0 * (double)num_epochs / total_train_time;
    std::cout << header << "Average training time per epoch: " << avg_train_time
              << " ms. Throughput: " << throughput << " epoch/s\n";
  }

  // evaluate, i.e. inference or predict
  double evaluate(std::string type, acc_t& loss, acc_t& acc) {
    Timer t_eval;
    t_eval.Start();
    size_t begin = 0, end = 0, count = 0;
    mask_t* masks = NULL;

    // TODO global here good for dist case?
    if (type == "train") {
      begin = globalTrainBegin;
      end   = globalTrainEnd;
      count = globalTrainCount;
      masks = globalTrainMasks;
      if (subgraph_sample_size) {
        // update masks for subgraph
        masks = NULL;
        begin = 0;
        end   = this->subgraphNumVertices;
        count = this->subgraphNumVertices;
      }
    } else if (type == "val") {
      begin = globalValBegin;
      end   = globalValEnd;
      count = globalValCount;
      masks = globalValMasks;
    } else {
      begin = globalTestBegin;
      end   = globalTestEnd;
      count = globalTestCount;
      masks = test_masks;
    }

#ifndef __GALOIS_HET_CUDA__
    // switch to the original graph if not training
    if (subgraph_sample_size && type != "train") {
      for (size_t i = 0; i < num_layers; i++)
        layers[i]->update_dim_size(distNumSamples);
      for (size_t i = 0; i < num_conv_layers; i++) {
        layers[i]->set_graph_ptr(distContext->getLGraphPointer());
        layers[i]->set_norm_consts_ptr(distContext->get_norm_factors_ptr());
      }
      layers[num_layers - 1]->set_labels_ptr(distContext->get_labels_ptr());
      layers[0]->set_feats_ptr(distContext->get_feats_ptr()); // feed input data
    }
#else
    if (type == "train") {
      masks = d_train_masks;
    } else if (type == "val") {
      masks = d_val_masks;
    } else {
      masks = d_test_masks;
    }
#endif

    loss                 = fprop(begin, end, count, masks);
    float_t* predictions = layers[num_layers - 1]->next()->get_data();

    // labels will be subgraph labels if applicable
    label_t* labels;
    if (type == "train" && subgraph_sample_size) {
      labels = distContext->get_labels_subg_ptr();
    } else {
      // note this grabs global labels; everything passed in should be global
      labels = distContext->get_labels_ptr();
    }

    if (is_single_class) {
      acc = masked_accuracy(begin, end, count, masks, predictions, labels);
    } else {
      acc = masked_multi_class_accuracy(begin, end, count, masks, predictions,
                                        labels);
    }

    t_eval.Stop();
    return t_eval.Millisecs();
  }

  // read masks of test set
  void read_test_masks(std::string dataset) {
    test_masks = new mask_t[distNumSamples];
    if (dataset == "reddit") {
      globalTestBegin = 177262;
      globalTestCount = 55703;
      globalTestEnd   = globalTestBegin + globalTestCount;
      for (size_t i = globalTestBegin; i < globalTestEnd; i++) {
#ifndef __GALOIS_HET_CUDA__
        if (dGraph->isLocal(i))
          test_masks[dGraph->getLID(i)] = 1;
#else
        // TODO: Read for GPU
#endif
      }
    } else {
      globalTestCount = distContext->read_masks(
          dataset, std::string("test"), globalSamples, globalTestBegin,
#ifdef __GALOIS_HET_CUDA__
          globalTestEnd, test_masks, NULL);
#else
          globalTestEnd, test_masks, dGraph);
#endif
    }
#ifdef __GALOIS_HET_CUDA__
    copy_test_masks_to_device();
#endif
  }
  void copy_test_masks_to_device();

  void construct_layers() {
    // append conv layers
    std::cout << "\nConstructing layers...\n";
    for (size_t i = 0; i < num_conv_layers - 1; i++) {
      append_conv_layer(i, true); // conv layers, act=true
    }

    append_conv_layer(num_conv_layers - 1); // the last hidden layer, act=false

    if (has_l2norm) {
      append_l2norm_layer(num_conv_layers); // l2_norm layer
    }

    if (has_dense) {
      append_dense_layer(num_layers - 2); // dense layer
    }

    append_out_layer(num_layers - 1); // output layer

    // allocate memory for intermediate features and gradients
    for (size_t i = 0; i < num_layers; i++) {
      layers[i]->add_edge();
    }
    for (size_t i = 1; i < num_layers; i++) {
      connect(layers[i - 1], layers[i]);
    }

    for (size_t i = 0; i < num_layers; i++) {
      layers[i]->malloc_and_init();
    }

    layers[0]->set_in_data(distContext->get_feats_ptr()); // feed input data
    // precompute the normalization constant based on graph structure
    // context->norm_factor_computing(false);
    distContext->constructNormFactor(graphTopologyContext);
    for (size_t i = 0; i < num_conv_layers; i++)
      layers[i]->set_norm_consts_ptr(distContext->get_norm_factors_ptr());
    set_contexts();
  }

  //! Add an l2_norm layer to the network
  void append_l2norm_layer(size_t layer_id) {
    assert(layer_id > 0); // can not be the first layer
    std::vector<size_t> in_dims(2), out_dims(2);
    in_dims[0]       = distNumSamples;
    in_dims[0]       = distNumSamples;
    in_dims[1]       = get_in_dim(layer_id);
    out_dims[1]      = get_out_dim(layer_id);
    layers[layer_id] = new l2_norm_layer(layer_id, in_dims, out_dims);
  }

  //! Add an dense layer to the network
  void append_dense_layer(size_t layer_id) {
    assert(layer_id > 0); // can not be the first layer
    std::vector<size_t> in_dims(2), out_dims(2);
    in_dims[0]  = distNumSamples;
    in_dims[0]  = distNumSamples;
    in_dims[1]  = get_in_dim(layer_id);
    out_dims[1] = get_out_dim(layer_id);
    // layers[layer_id] = new dense_layer(layer_id, in_dims, out_dims);
  }

  //! Add an output layer to the network
  void append_out_layer(size_t layer_id) {
    assert(layer_id > 0); // can not be the first layer
    std::vector<size_t> in_dims(2), out_dims(2);
    in_dims[0] = out_dims[0] = distNumSamples;
    in_dims[1]               = get_in_dim(layer_id);
    out_dims[1]              = get_out_dim(layer_id);

    if (is_single_class)
      layers[layer_id] = new softmax_loss_layer(layer_id, in_dims, out_dims);
    else
      layers[layer_id] = new sigmoid_loss_layer(layer_id, in_dims, out_dims);

    layers[layer_id]->set_labels_ptr(distContext->get_labels_ptr());
  }

  //! Add a convolution layer to the network
  void append_conv_layer(size_t layer_id, bool act = false, bool norm = true,
                         bool bias = false, bool dropout = true) {
    assert(dropout_rate < 1.0);
    assert(layer_id < num_conv_layers);
    std::vector<size_t> in_dims(2), out_dims(2);
    in_dims[0] = out_dims[0] = distNumSamples;
    in_dims[1]               = get_in_dim(layer_id);
    out_dims[1]              = get_out_dim(layer_id);
    layers[layer_id] = new graph_conv_layer(layer_id, act, norm, bias, dropout,
                                            dropout_rate, in_dims, out_dims);
    layers[layer_id]->set_graph_ptr(distContext->getLGraphPointer());
  }

  // update trainable weights after back-propagation
  void update_weights(optimizer* opt) {
    regularize();
    for (size_t i = 0; i < num_layers; i++) {
      if (layers[i]->trainable()) {
        layers[i]->update_weight(opt);
      }
    }
  }

  //! forward propagation: [begin, end) is the range of samples used.
  //! calls "forward" on each layer and returns the loss of the final layer
  acc_t fprop(size_t begin, size_t end, size_t count, mask_t* masks) {
    // set mask for the last layer; globals
    // TODO this should be distirbuted sample begin->end not global; fix later
    // seems to be unused in code right now anyways
    layers[num_layers - 1]->set_sample_mask(begin, end, count, masks);

    for (size_t i = 0; i < num_layers; i++) {
      layers[i]->forward();
    }
    // prediction error
    auto loss = layers[num_layers - 1]->get_prediction_loss();
    // Squared Norm Regularization to mitigate overfitting
    loss += weight_decay * layers[0]->get_weight_decay_loss();
    return loss;
  }

  void bprop() {
    for (size_t i = num_layers; i != 0; i--) {
      layers[i - 1]->backward();
    }
  }

  //! Save the context object to all layers of the network
  void set_contexts() {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->set_context(distContext);
  }
  //! set netphases for all layers in this network
  void set_netphases(net_phase phase) {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->set_netphase(phase);
  }
  //! print all layers
  void print_layers_info() {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->print_layer_info();
  }

  // comparing outputs with the ground truth (labels)
  acc_t masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks,
                        float_t* preds, label_t* ground_truth);
  acc_t masked_multi_class_accuracy(size_t begin, size_t end, size_t count,
                                    mask_t* masks, float_t* preds,
                                    label_t* ground_truth);
};

} // namespace deepgalois
