#include "galois/Galois.h"
#include "deepgalois/Net.h"

namespace deepgalois {

Net::Net(std::string dataset_str, int nt, unsigned n_conv, int epochs,
         unsigned hidden1, float lr, float dropout, float wd, bool selfloop,
         bool single, bool l2norm, bool dense, unsigned neigh_sz,
         unsigned subg_sz, int val_itv)
    : //    globalSamples(0), num_classes(0), num_conv_layers(0), num_layers(0),
      //    globalTrainBegin(0), globalTrainEnd(0), globalTrainCount(0),
      //    globalValBegin(0), globalValEnd(0), globalValCount(0),
      //    globalTestBegin(0), globalTestEnd(0), globalTestCount(0),
      //    globalTrainMasks(NULL), globalValMasks(NULL), globalTestMasks(NULL)
      //    {}
      is_single_class(single), has_l2norm(l2norm), has_dense(dense),
      neighbor_sample_size(neigh_sz), subgraph_sample_size(subg_sz),
      num_threads(nt), num_conv_layers(n_conv), num_epochs(epochs), h1(hidden1),
      learning_rate(lr), dropout_rate(dropout), weight_decay(wd),
      val_interval(val_itv), num_subgraphs(1), is_selfloop(selfloop) {
  // init some identifiers for this host
  unsigned myID = 0;
#ifndef GALOIS_ENABLE_GPU
  myID = galois::runtime::getSystemNetworkInterface().ID;
#endif
  this->header = "[" + std::to_string(myID) + "] ";
  assert(n_conv > 0);
  this->num_layers = num_conv_layers + 1;

  // additional layers to add
  if (has_l2norm)
    this->num_layers++;
  if (has_dense)
    this->num_layers++;
  // initialize feature metadata
  feature_dims.resize(num_layers + 1);
  print_configs();

  // initialze global graph context
  graphTopologyContext = new deepgalois::Context();
  graphTopologyContext->set_dataset(dataset_str);
  // read *entire* graph, get num nodes
  globalSamples = graphTopologyContext->read_graph(selfloop);

  // get training and validation sets: this is to create the training
  // subgraph in the sampler
  globalTrainMasks = new mask_t[globalSamples];
  globalValMasks   = new mask_t[globalSamples];
  globalTestMasks  = new mask_t[globalSamples];
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
  // make sure sampel size isn't greater than what we have to train with
  assert(subgraph_sample_size <= globalTrainCount);

  layers.resize(num_layers);
  // hidden1 level embedding: 16
  for (size_t i = 1; i < num_conv_layers; i++)
    feature_dims[i] = this->h1;

  // features are read in distcontext, not this context (this context only
  // used for sampling)
  if (subgraph_sample_size)
    sampler = new deepgalois::Sampler();
}

void Net::train(optimizer* opt, bool need_validate) {
  galois::StatTimer train_timer("Timer_0");
  train_timer.start();
  std::string separator   = "\n";
  double total_train_time = 0.0;
  int num_subg_remain     = 0;
#ifndef GALOIS_ENABLE_GPU
  unsigned hostID = galois::runtime::getSystemNetworkInterface().ID;
#endif

  if (subgraph_sample_size) {
    galois::StatTimer construct_time("SubgraphAllocateTime");
    construct_time.start();
    distContext->allocateSubgraphs(num_subgraphs, subgraph_sample_size);
    allocateSubgraphsMasks(num_subgraphs);
    std::cout << header
              << "Constructing training vertex set induced graph...\n";
    // auto gg = distContext->getGraphPointer();
    auto gg =
        graphTopologyContext->getGraphPointer(); // gloabl graph in CPU mem
    sampler->initializeMaskedGraph(globalTrainCount, globalTrainMasks, gg,
                                   distContext->getGraphPointer());
    construct_time.stop();
  }

  galois::gPrint(header, "Start training...\n");

  Timer t_epoch;

  // run epochs
  for (int curEpoch = 0; curEpoch < num_epochs; curEpoch++) {
    t_epoch.Start();

    ////////////////////////////////////////////////////////////////////////////////
    // Sampling
    ////////////////////////////////////////////////////////////////////////////////
    if (subgraph_sample_size) {
      galois::StatTimer sample_time("SubgraphSampleTime");
      sample_time.start();
      if (num_subg_remain == 0) {
        std::cout << header << "Generating " << num_subgraphs
                  << " subgraph(s)\n";
        galois::StatTimer t_subgen("SubgraphGenerateTime");
        t_subgen.start();

        // generate subgraphs
        for (int sid = 0; sid < num_subgraphs; sid++) {
          VertexSet sampledSet;
          sampler->selectVertices(subgraph_sample_size, sampledSet,
                                  curEpoch); // m = 1000 by default
          sampler->generateSubgraph(sampledSet,
                                    subgraphs_masks + sid * globalSamples,
                                    distContext->getSubgraphPointer(sid));
        }
        num_subg_remain = num_subgraphs;
        t_subgen.stop();
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

      // std::cout << "Subgraph num_vertices: " << subgraphNumVertices
      //          << ", num_edges: " << subgraphPointer->sizeEdges() << "\n";
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
          this->subgraphNumVertices, subgraphs_masks + sg_id * globalSamples);
      layers[num_layers - 1]->set_labels_ptr(
          distContext->get_labels_subg_ptr());

      // update features for subgraph
      distContext->constructSubgraphFeatures(
          this->subgraphNumVertices, subgraphs_masks + sg_id * globalSamples);
      layers[0]->set_feats_ptr(
          distContext->get_feats_subg_ptr()); // feed input data

      // Graph* testing = distContext->getSubgraphPointer(sg_id);
      // for (size_t i = 0; i < testing->size(); i++) {
      //  for (auto j = testing->edge_begin(i); j < testing->edge_end(i); j++)
      //  {
      //    galois::gPrint(i, " ", testing->getEdgeDst(j), "\n");
      //  }
      //}
      sample_time.stop();
    } // end subgraph sample loop
    ////////////////////////////////////////////////////////////////////////////////

    // training steps
#ifdef GALOIS_ENABLE_GPU
    std::cout << header << "Epoch " << std::setw(3) << curEpoch << " ";
#else
    if (hostID == 0) {
      galois::gInfo("Epoch ", std::setw(3), curEpoch);
    }
#endif
    set_netphases(net_phase::train);
    acc_t train_loss = 0.0, train_acc = 0.0;

    // galois::gPrint(header, "Calling into eval for forward propagation\n");
    // forward: after this phase, layer edges will contain intermediate
    // features for use during backprop
    double fw_time = evaluate("train", train_loss, train_acc);
    // evaluate("train", train_loss, train_acc);

    // galois::gPrint(header, "Calling into backward propagation\n");
    // backward: use intermediate features + ground truth to update layers
    // with feature gradients whcih are then used to calculate weight
    // gradients
    Net::bprop();

    // galois::gPrint(header, "Weight update call\n");
    // gradient update: use gradients stored on each layer to update model
    // for next epoch
    Net::update_weights(opt); // update parameters

    t_epoch.Stop();

    // validation / testing
    set_netphases(net_phase::test);

#ifdef GALOIS_ENABLE_GPU
    std::cout << header << "train_loss " << std::setprecision(3) << std::fixed
              << train_loss << " train_acc " << train_acc << " ";
#else
    if (hostID == 0) {
      galois::gPrint(header, "train_loss ", std::setprecision(3), std::fixed,
                     train_loss, " train_acc ", train_acc, separator);
    }
#endif

    double epoch_time = t_epoch.Millisecs();
    total_train_time += epoch_time;

    // report current total time + accuracy as a stat
#ifndef GALOIS_ENABLE_GPU
    if (hostID == 0) {
      galois::runtime::reportParam(
          std::string("GNN"),
          "Epoch" + std::to_string(curEpoch) + "TestAccuracyAndTime",
          std::to_string(train_acc) + ";" + std::to_string(total_train_time));
    }
#endif

    if (need_validate && (curEpoch % val_interval == 0)) {
      // Validation
      acc_t val_loss = 0.0, val_acc = 0.0;
      double val_time = evaluate("val", val_loss, val_acc);
#ifdef GALOIS_ENABLE_GPU
      std::cout << header << "val_loss " << std::setprecision(3) << std::fixed
                << val_loss << " val_acc " << val_acc << " ";
      std::cout << header << "time " << std::setprecision(3) << std::fixed
                << epoch_time + val_time << " ms (train_time " << epoch_time
                << " val_time " << val_time << ")\n";
#else
      if (hostID == 0) {
        galois::gPrint(header, "val_loss ", std::setprecision(3), std::fixed,
                       val_loss, " val_acc ", val_acc, separator);
        galois::gPrint(header, "time ", std::setprecision(3), std::fixed,
                       epoch_time + val_time, " ms (train_time ", epoch_time,
                       " val_time ", val_time, ")\n");
      }
#endif
    } else {
#ifdef GALOIS_ENABLE_GPU
      std::cout << header << "train_time " << std::fixed << epoch_time
                << " ms (fw " << fw_time << ", bw " << epoch_time - fw_time
                << ")\n";
#else
      if (hostID == 0) {
        galois::gPrint(header, "train_time ", std::fixed, epoch_time,
                       " ms (fw ", fw_time, ", bw ", epoch_time - fw_time,
                       ")\n");
      }
#endif
    }
  } // epoch loop

  double avg_train_time = total_train_time / (double)num_epochs;
  double throughput     = 1000.0 * (double)num_epochs / total_train_time;
#ifdef GALOIS_ENABLE_GPU
  std::cout << "Average training time per epoch: " << avg_train_time
            << "ms. Throughput " << throughput << " epoch/s\n";
#else
  galois::gPrint(header, "Average training time per epoch: ", avg_train_time,
                 " ms. Throughput: ", throughput, " epoch/s\n");
#endif
  train_timer.stop();
}

// evaluate, i.e. inference or predict
double Net::evaluate(std::string type, acc_t& loss, acc_t& acc) {
  // TODO get rid of this timer
  Timer t_eval;
  t_eval.Start();

  galois::StatTimer eval_timer("EvaluateTime");
  eval_timer.start();

  size_t gBegin = 0, gEnd = 0, gCount = 0;
  mask_t* gMasks = NULL;

  // TODO global here good for dist case?
  if (type == "train") {
    gBegin = globalTrainBegin;
    gEnd   = globalTrainEnd;
    gCount = globalTrainCount;
    gMasks = globalTrainMasks;
    if (subgraph_sample_size) {
      // update gMasks for subgraph
      gMasks = NULL;
      gBegin = 0;
      gEnd   = this->subgraphNumVertices;
      gCount = this->subgraphNumVertices;
    }
  } else if (type == "val") {
    gBegin = globalValBegin;
    gEnd   = globalValEnd;
    gCount = globalValCount;
    gMasks = globalValMasks;
  } else {
    gBegin = globalTestBegin;
    gEnd   = globalTestEnd;
    gCount = globalTestCount;
    gMasks = globalTestMasks;
  }

  // switch to the original graph if not training
  if (subgraph_sample_size && type != "train") {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->update_dim_size(distNumSamples);
    for (size_t i = 0; i < num_conv_layers; i++) {
#ifdef GALOIS_ENABLE_GPU
      layers[i]->set_graph_ptr(distContext->getGraphPointer());
#else
      layers[i]->set_graph_ptr(distContext->getLGraphPointer());
#endif
      layers[i]->set_norm_consts_ptr(distContext->get_norm_factors_ptr());
    }
    layers[num_layers - 1]->set_labels_ptr(distContext->get_labels_ptr());
    layers[0]->set_feats_ptr(distContext->get_feats_ptr()); // feed input data
  }
#ifdef GALOIS_ENABLE_GPU
  if (type == "train") {
    gMasks = d_train_masks;
  } else if (type == "val") {
    gMasks = d_val_masks;
  } else {
    gMasks = d_test_masks;
  }
#endif

  // galois::gPrint(header, "Doing actual forward propagation\n");
  loss = fprop(gBegin, gEnd, gCount, gMasks);
  // galois::gPrint(header,
  //               "Forward propagation donne, going to check accuracy\n");
  float_t* predictions = layers[num_layers - 1]->next()->get_data();

  // labels will be subgraph labels if applicable
  label_t* localLabels;
  if (type == "train" && subgraph_sample_size) {
    localLabels = distContext->get_labels_subg_ptr();
  } else {
    // note this grabs local labels
    localLabels = distContext->get_labels_ptr();
  }

  if (is_single_class) {
    acc =
        masked_accuracy(gBegin, gEnd, gCount, gMasks, predictions, localLabels);
  } else {
    acc = masked_multi_class_accuracy(gBegin, gEnd, gCount, gMasks, predictions,
                                      localLabels);
  }

  eval_timer.stop();

  // TODO replace with stat timer
  t_eval.Stop();
  return t_eval.Millisecs();
}

void Net::construct_layers() {
  // append conv layers
  // galois::gPrint(header, "Constructing layers...\n");
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
void Net::append_l2norm_layer(size_t layer_id) {
  assert(layer_id > 0); // can not be the first layer
  std::vector<size_t> in_dims(2), out_dims(2);
  in_dims[0]       = distNumSamples;
  in_dims[0]       = distNumSamples;
  in_dims[1]       = get_in_dim(layer_id);
  out_dims[1]      = get_out_dim(layer_id);
  layers[layer_id] = new l2_norm_layer(layer_id, in_dims, out_dims);
}

//! Add an dense layer to the network
void Net::append_dense_layer(size_t layer_id) {
  assert(layer_id > 0); // can not be the first layer
  std::vector<size_t> in_dims(2), out_dims(2);
  in_dims[0]  = distNumSamples;
  in_dims[0]  = distNumSamples;
  in_dims[1]  = get_in_dim(layer_id);
  out_dims[1] = get_out_dim(layer_id);
  // layers[layer_id] = new dense_layer(layer_id, in_dims, out_dims);
}

//! Add an output layer to the network
void Net::append_out_layer(size_t layer_id) {
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
void Net::append_conv_layer(size_t layer_id, bool act, bool norm, bool bias,
                            bool dropout) {
  assert(dropout_rate < 1.0);
  assert(layer_id < num_conv_layers);
  std::vector<size_t> in_dims(2), out_dims(2);
  in_dims[0] = out_dims[0] = distNumSamples;
  in_dims[1]               = get_in_dim(layer_id);
  out_dims[1]              = get_out_dim(layer_id);
  layers[layer_id] = new graph_conv_layer(layer_id, act, norm, bias, dropout,
                                          dropout_rate, in_dims, out_dims);
#ifdef GALOIS_ENABLE_GPU
  layers[layer_id]->set_graph_ptr(distContext->getGraphPointer());
#else
  layers[layer_id]->set_graph_ptr(distContext->getLGraphPointer());
#endif
}

//! forward propagation: [begin, end) is the range of samples used.
//! calls "forward" on each layer and returns the loss of the final layer
acc_t Net::fprop(size_t gBegin, size_t gEnd, size_t gCount, mask_t* gMasks) {
  galois::StatTimer fprop_timer("ForwardPropTime");
  fprop_timer.start();
  // set mask for the last layer; globals
  // TODO this should be distirbuted sample gBegin->end not global; fix later
  // seems to be unused in code right now anyways
  // galois::gPrint(header, "fprop: set sample mask\n");
  layers[num_layers - 1]->set_sample_mask(gBegin, gEnd, gCount, gMasks);

  for (size_t i = 0; i < num_layers; i++) {
    galois::gPrint(header, "fprop: layer ", i, " forward call\n");
    layers[i]->forward();
  }

  // galois::gPrint(header, "fprop: getting loss\n");
  // prediction error
  acc_t loss = layers[num_layers - 1]->get_prediction_loss();
  // Squared Norm Regularization to mitigate overfitting
  loss += weight_decay * layers[0]->get_weight_decay_loss();
  fprop_timer.stop();
  return loss;
}

// back propagation
void Net::bprop() {
  galois::StatTimer bprop_timer("BackPropTime");
  bprop_timer.start();
  for (size_t i = num_layers; i != 0; i--) {
    layers[i - 1]->backward();
  }
  bprop_timer.stop();
}

// update trainable weights after back-propagation
void Net::update_weights(optimizer* opt) {
  regularize();
  for (size_t i = 0; i < num_layers; i++) {
    if (layers[i]->trainable()) {
      layers[i]->update_weight(opt);
    }
  }
}

//! Save the context object to all layers of the network
void Net::set_contexts() {
  for (size_t i = 0; i < num_layers; i++)
    layers[i]->set_context(distContext);
}

//! set netphases for all layers in this network
void Net::set_netphases(net_phase phase) {
  for (size_t i = 0; i < num_layers; i++)
    layers[i]->set_netphase(phase);
}

//! print all layers
void Net::print_layers_info() {
  for (size_t i = 0; i < num_layers; i++)
    layers[i]->print_layer_info();
}

// print the configurations
void Net::print_configs() {
  galois::gPrint(header, "Configuration: num_threads ", num_threads,
                 ", num_conv_layers ", num_conv_layers, ", num_epochs ",
                 num_epochs, ", hidden_feat_len ", h1, ", learning_rate ",
                 learning_rate, ", dropout_rate ", dropout_rate,
                 ", weight_decay ", weight_decay, "\n");
}

} // namespace deepgalois
