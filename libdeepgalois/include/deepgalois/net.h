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
#ifdef CPU_ONLY
#include "deepgalois/sampler.h"
#endif
#ifndef GALOIS_USE_DIST
#include "deepgalois/Context.h"
#else
#include "deepgalois/GraphTypes.h"
#include "deepgalois/DistContext.h"
#endif

namespace deepgalois {

// N: number of vertices, D: feature vector dimentions,
// E: number of distinct labels, i.e. number of vertex classes
// layer 1: features N x D, weights D x 16, out N x 16 (hidden1=16)
// layer 2: features N x 16, weights 16 x E, out N x E
class Net {
public:
  Net(std::string dataset_str, int nt, unsigned n_conv, int epochs,
      unsigned hidden1, float lr, float dropout, float wd, bool selfloop,
      bool single, bool l2norm, bool dense, unsigned neigh_sz, unsigned subg_sz,
      int val_itv)
      : is_single_class(single), has_l2norm(l2norm), has_dense(dense),
        neighbor_sample_size(neigh_sz), subgraph_sample_size(subg_sz),
        num_threads(nt), num_conv_layers(n_conv), num_epochs(epochs),
        learning_rate(lr), dropout_rate(dropout), weight_decay(wd),
        val_interval(val_itv), num_subgraphs(1), is_selfloop(selfloop) {
    assert(n_conv > 0);
    // TODO use galois print
    std::cout << "Configuration: num_threads " << num_threads
              << ", num_conv_layers " << num_conv_layers << ", num_epochs "
              << num_epochs << ", hidden1 " << hidden1 << ", learning_rate "
              << learning_rate << ", dropout_rate " << dropout_rate
              << ", weight_decay " << weight_decay << "\n";
    num_layers = num_conv_layers + 1;

    // additional layers to add
    if (has_l2norm)
      num_layers++;
    if (has_dense)
      num_layers++;

    // initialize feature metadata
    feature_dims.resize(num_layers + 1);

    // initialze context
    context = new deepgalois::Context();
    context->set_dataset(dataset_str);
    // read graph, get num nodes
    num_samples = context->read_graph(selfloop);
    context->set_label_class(is_single_class);
    // read ground truth labels
    num_classes = context->read_labels();

    // get training and validation sets
    train_masks = new mask_t[num_samples];
    val_masks   = new mask_t[num_samples];
    std::fill(train_masks, train_masks + num_samples, 0);
    std::fill(val_masks, val_masks + num_samples, 0);

    // reddit is hard coded
    if (dataset_str == "reddit") {
      train_begin = 0, train_count = 153431,
      train_end = train_begin + train_count;
      val_begin = 153431, val_count = 23831, val_end = val_begin + val_count;
      // TODO do all can be used below
      for (size_t i = train_begin; i < train_end; i++)
        train_masks[i] = 1;
      for (size_t i = val_begin; i < val_end; i++)
        val_masks[i] = 1;
    } else {
      train_count = context->read_masks("train", num_samples, train_begin,
                                        train_end, train_masks);
      val_count   = context->read_masks("val", num_samples, val_begin, val_end,
                                      val_masks);
    }

    // make sure sampel size isn't greater than what we have to train with
    if (subgraph_sample_size > train_count) {
      GALOIS_DIE("subgraph size can not be larger than the size of training "
                 "set\n");
    }

    // read features of vertices
    feature_dims[0] = context->read_features(); // input feature dimension: D

    for (size_t i = 1; i < num_conv_layers; i++)
      feature_dims[i] = hidden1;                 // hidden1 level embedding: 16

    feature_dims[num_conv_layers] = num_classes; // output embedding: E

    if (has_l2norm)
      feature_dims[num_conv_layers + 1] =
          num_classes; // l2 normalized embedding: E

    if (has_dense)
      feature_dims[num_layers - 1] = num_classes; // MLP embedding: E

    feature_dims[num_layers] = num_classes; // normalized output embedding: E
    layers.resize(num_layers);

    // set the subgraph boolean if sample size is greater than 0
    context->set_use_subgraph(subgraph_sample_size > 0);
  }

  //! Default net constructor
  //Net()
  //    : is_single_class(true), has_l2norm(false), has_dense(false),
  //      neighbor_sample_size(0), subgraph_sample_size(0), num_threads(1),
  //      num_samples(0), num_classes(0), num_conv_layers(0), num_layers(0),
  //      num_epochs(0), learning_rate(0.0), dropout_rate(0.0), weight_decay(0.0),
  //      train_begin(0), train_end(0), train_count(0), val_begin(0), val_end(0),
  //      val_count(0), test_begin(0), test_end(0), test_count(0),
  //      val_interval(1), num_subgraphs(1), num_vertices_sg(9000),
  //      train_masks(NULL), val_masks(NULL), test_masks(NULL), context(NULL) {}

  //! save graph pointer to context object
  void saveDistGraph(Graph* dGraph);

#ifdef GALOIS_USE_DIST
  void dist_init(Graph* graph, std::string dataset_str);
#endif

  size_t get_in_dim(size_t layer_id) { return feature_dims[layer_id]; }
  size_t get_out_dim(size_t layer_id) { return feature_dims[layer_id + 1]; }
  size_t get_nnodes() { return num_samples; }

  void normalize();  // Scale gradient to counterbalance accumulation
  void regularize(); // add weight decay

  void train(optimizer* opt, bool need_validate) {
    unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
    std::string header        = "[" + std::to_string(myID) + "] ";
    std::string seperator     = "\n";

    double total_train_time = 0.0;
    int num_subg_remain     = 0;
#ifdef CPU_ONLY
#ifndef GALOIS_USE_DIST
    if (subgraph_sample_size) {
      context->createSubgraphs(num_subgraphs);
      subgraphs_masks = new mask_t[num_samples * num_subgraphs];
      std::cout << "\nConstruct training vertex set induced graph...\n";
      sampler->set_masked_graph(train_begin, train_end, train_count,
                                train_masks, context->getGraphPointer());
    }
#endif
#endif
    std::cout << "\nStart training...\n";
    Timer t_epoch;
    // run epochs
    for (int ep = 0; ep < num_epochs; ep++) {
      t_epoch.Start();

      if (subgraph_sample_size) {
        if (num_subg_remain == 0) {
          std::cout << "Generating " << num_subgraphs << " subgraphs ";
          Timer t_subgen;
          t_subgen.Start();
          // generate subgraphs
#ifdef CPU_ONLY
#ifndef GALOIS_USE_DIST
          for (int sid = 0; sid < num_subgraphs; sid++) {
            // galois::do_all(galois::iterate(size_t(0),
            // size_t(num_subgraphs)),[&](const auto sid) {
            unsigned tid = 0;
            // tid = galois::substrate::ThreadPool::getTID();
            sampler->subgraph_sample(subgraph_sample_size,
                                     *(context->getSubgraphPointer(sid)),
                                     &subgraphs_masks[sid * num_samples], tid);
          } //, galois::loopname("subgraph_gen"));
#endif
#endif
          num_subg_remain = num_subgraphs;
          t_subgen.Stop();
          // std::cout << "Done, time: " << t_subgen.Millisecs() << "\n";
        }
#ifndef GALOIS_USE_DIST
        for (int i = 0; i < num_subgraphs; i++) {
          auto sg_ptr = context->getSubgraphPointer(i);
          sg_ptr->degree_counting();
          // galois::gPrint("\tsubgraph[", i, "]: num_v ", sg_ptr->size(), "
          // num_e ", sg_ptr->sizeEdges(), "\n");
        }
#endif // GALOIS_USE_DIST
        num_subg_remain--;
        int sg_id         = num_subg_remain;
        auto subgraph_ptr = context->getSubgraphPointer(sg_id);
        num_vertices_sg   = subgraph_ptr->size();
        // galois::gPrint("Subgraph num_vertices: ", num_vertices_sg, ",
        // num_edges: ", subgraph_ptr->sizeEdges(), "\n");
        for (size_t i = 0; i < num_layers; i++)
          layers[i]->update_dim_size(num_vertices_sg);
        context->norm_factor_computing(1, sg_id);
        for (size_t i = 0; i < num_conv_layers; i++) {
          layers[i]->set_graph_ptr(subgraph_ptr);
          layers[i]->set_norm_consts_ptr(context->get_norm_factors_subg_ptr());
        }
        // update labels for subgraph
        context->gen_subgraph_labels(num_vertices_sg,
                                     &subgraphs_masks[sg_id * num_samples]);
        layers[num_layers - 1]->set_labels_ptr(context->get_labels_subg_ptr());

        // update features for subgraph
        context->gen_subgraph_feats(num_vertices_sg,
                                    &subgraphs_masks[sg_id * num_samples]);
        layers[0]->set_feats_ptr(
            context->get_feats_subg_ptr()); // feed input data
      }

      // training steps
      std::cout << header << "Epoch " << std::setw(3) << ep << seperator;
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
      if (need_validate && ep % val_interval == 0) {
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
    }
    double avg_train_time = total_train_time / (double)num_epochs;
    double throughput     = 1000.0 * (double)num_epochs / total_train_time;
    std::cout << "\nAverage training time: " << avg_train_time
              << " ms. Throughput: " << throughput << " epoch/s\n";
  }

  // evaluate, i.e. inference or predict
  double evaluate(std::string type, acc_t& loss, acc_t& acc) {
    // TODO may need to do something for the dist case
    Timer t_eval;
    t_eval.Start();
    size_t begin = 0, end = 0, count = 0;
    mask_t* masks = NULL;
    if (type == "train") {
      begin = train_begin;
      end   = train_end;
      count = train_count;
      masks = train_masks;
      if (subgraph_sample_size) {
        // update masks for subgraph
        masks = NULL;
        begin = 0;
        end   = num_vertices_sg;
        count = num_vertices_sg;
      }
    } else if (type == "val") {
      begin = val_begin;
      end   = val_end;
      count = val_count;
      masks = val_masks;
    } else {
      begin = test_begin;
      end   = test_end;
      count = test_count;
      masks = test_masks;
    }
#ifdef CPU_ONLY
    if (subgraph_sample_size &&
        type != "train") { // switch to the original graph
      for (size_t i = 0; i < num_layers; i++)
        layers[i]->update_dim_size(num_samples);
      for (size_t i = 0; i < num_conv_layers; i++) {
        layers[i]->set_graph_ptr(context->getGraphPointer());
        layers[i]->set_norm_consts_ptr(context->get_norm_factors_ptr());
      }
      layers[num_layers - 1]->set_labels_ptr(context->get_labels_ptr());
      layers[0]->set_feats_ptr(context->get_feats_ptr()); // feed input data
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
    label_t* labels;
    if (type == "train" && subgraph_sample_size) {
      labels = context->get_labels_subg_ptr();
    } else {
      labels = context->get_labels_ptr();
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
    test_masks = new mask_t[num_samples];
    if (dataset == "reddit") {
      test_begin = 177262;
      test_count = 55703;
      test_end   = test_begin + test_count;
#ifndef GALOIS_USE_DIST
      for (size_t i = test_begin; i < test_end; i++)
        test_masks[i] = 1;
#else
      for (size_t i = test_begin; i < test_end; i++) {
        if (dGraph->isLocal(i)) {
          test_masks[dGraph->getLID(i)] = 1;
        }
      }
#endif
    } else {
#ifndef GALOIS_USE_DIST
      test_count = context->read_masks("test", num_samples, test_begin,
                                       test_end, test_masks);
#else
      test_count = context->read_masks("test", num_samples, test_begin,
                                       test_end, test_masks, dGraph);
#endif
    }
#ifndef CPU_ONLY
    copy_test_masks_to_device();
#endif
  }
  void copy_test_masks_to_device();

  void construct_layers() {
    // append conv layers
    std::cout << "\nConstructing layers...\n";
    for (size_t i = 0; i < num_conv_layers - 1; i++)
      append_conv_layer(i, true);           // conv layers, act=true
    append_conv_layer(num_conv_layers - 1); // the last hidden layer, act=false
    if (has_l2norm)
      append_l2norm_layer(num_conv_layers); // l2_norm layer
    if (has_dense)
      append_dense_layer(num_layers - 2); // dense layer
    append_out_layer(num_layers - 1);     // output layer

    // allocate memory for intermediate features and gradients
    for (size_t i = 0; i < num_layers; i++) {
      layers[i]->add_edge();
    }
    for (size_t i = 1; i < num_layers; i++)
      connect(layers[i - 1], layers[i]);
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->malloc_and_init();
    layers[0]->set_in_data(context->get_feats_ptr()); // feed input data
    // precompute the normalization constant based on graph structure
    context->norm_factor_computing(0);
    for (size_t i = 0; i < num_conv_layers; i++)
      layers[i]->set_norm_consts_ptr(context->get_norm_factors_ptr());
    set_contexts();
  }

  //! Add an l2_norm layer to the network
  void append_l2norm_layer(size_t layer_id) {
    assert(layer_id > 0); // can not be the first layer
    std::vector<size_t> in_dims(2), out_dims(2);
    in_dims[0]       = num_samples;
    in_dims[0]       = num_samples;
    in_dims[1]       = get_in_dim(layer_id);
    out_dims[1]      = get_out_dim(layer_id);
    layers[layer_id] = new l2_norm_layer(layer_id, in_dims, out_dims);
  }

  //! Add an dense layer to the network
  void append_dense_layer(size_t layer_id) {
    assert(layer_id > 0); // can not be the first layer
    std::vector<size_t> in_dims(2), out_dims(2);
    in_dims[0]  = num_samples;
    in_dims[0]  = num_samples;
    in_dims[1]  = get_in_dim(layer_id);
    out_dims[1] = get_out_dim(layer_id);
    // layers[layer_id] = new dense_layer(layer_id, in_dims, out_dims);
  }

  //! Add an output layer to the network
  void append_out_layer(size_t layer_id) {
    assert(layer_id > 0); // can not be the first layer
    std::vector<size_t> in_dims(2), out_dims(2);
    in_dims[0] = out_dims[0] = num_samples;
    in_dims[1]               = get_in_dim(layer_id);
    out_dims[1]              = get_out_dim(layer_id);
    if (is_single_class)
      layers[layer_id] = new softmax_loss_layer(layer_id, in_dims, out_dims);
    else
      layers[layer_id] = new sigmoid_loss_layer(layer_id, in_dims, out_dims);
    layers[layer_id]->set_labels_ptr(context->get_labels_ptr());
  }

  //! Add a convolution layer to the network
  void append_conv_layer(size_t layer_id, bool act = false, bool norm = true,
                         bool bias = false, bool dropout = true) {
    assert(dropout_rate < 1.0);
    assert(layer_id < num_conv_layers);
    std::vector<size_t> in_dims(2), out_dims(2);
    in_dims[0] = out_dims[0] = num_samples;
    in_dims[1]               = get_in_dim(layer_id);
    out_dims[1]              = get_out_dim(layer_id);
    layers[layer_id] = new graph_conv_layer(layer_id, act, norm, bias, dropout,
                                            dropout_rate, in_dims, out_dims);
    layers[layer_id]->set_graph_ptr(context->getGraphPointer());
  }

  // update trainable weights after back-propagation
  void update_weights(optimizer* opt) {
    normalize();
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
    // set mask for the last layer
    layers[num_layers - 1]->set_sample_mask(begin, end, count, masks);
    // layer0: from N x D to N x 16
    // layer1: from N x 16 to N x E
    // layer2: from N x E to N x E (normalize only)
    for (size_t i = 0; i < num_layers; i++) {
      layers[i]->forward();
      // TODO need to sync model between layers here
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
      layers[i]->set_context(context);
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

protected:
  bool is_single_class;          // single-class (one-hot) or multi-class label
  bool has_l2norm;               // whether the net contains an l2_norm layer
  bool has_dense;                // whether the net contains an dense layer
  unsigned neighbor_sample_size; // neighbor sampling
  unsigned subgraph_sample_size; // subgraph sampling
  int num_threads;               // number of threads
  size_t num_samples;            // number of samples: N
  size_t num_classes;            // number of vertex classes: E
  size_t num_conv_layers;        // number of convolutional layers
  size_t num_layers;             // total number of layers (conv + output)
  int num_epochs;                // number of epochs
  float learning_rate;           // learning rate
  float dropout_rate;            // dropout rate
  float weight_decay;            // weighti decay for over-fitting
  size_t train_begin, train_end, train_count;
  size_t val_begin, val_end, val_count;
  size_t test_begin, test_end, test_count;
  int val_interval;
  int num_subgraphs;
  int num_vertices_sg;
  bool is_selfloop;

  mask_t* train_masks;              // masks for training
  mask_t* d_train_masks;            // masks for training on device
  mask_t* val_masks;                // masks for validation
  mask_t* d_val_masks;              // masks for validation on device
  mask_t* test_masks;               // masks for test
  mask_t* d_test_masks;             // masks for test on device
  mask_t* subgraphs_masks;          // masks for subgraphs
  std::vector<size_t> feature_dims; // feature dimnesions for each layer
  std::vector<layer*> layers;       // all the layers in the neural network
#ifndef GALOIS_USE_DIST
  deepgalois::Context* context;
#else
  deepgalois::DistContext* context;
  Graph* dGraph;
#endif

#ifdef CPU_ONLY
#ifndef GALOIS_USE_DIST
  Sampler* sampler;
#endif
#endif
  // comparing outputs with the ground truth (labels)
  acc_t masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks,
                        float_t* preds, label_t* ground_truth);
  acc_t masked_multi_class_accuracy(size_t begin, size_t end, size_t count,
                                    mask_t* masks, float_t* preds,
                                    label_t* ground_truth);
};

} // namespace deepgalois
