// XXX include net interface if necessary
#include "galois/Logging.h"
#include "galois/graphs/ReadGraph.h"
#include "galois/graphs/GNNGraph.h"
#include "galois/GNNMath.h"
#include <limits>

namespace {
//! Partitions a particular dataset given some partitioning scheme
std::unique_ptr<galois::graphs::GNNGraph::GNNDistGraph>
LoadPartition(const std::string& input_directory,
              const std::string& dataset_name,
              galois::graphs::GNNPartitionScheme partition_scheme) {
  // XXX input path
  std::string input_file = input_directory + dataset_name + ".csgr";
  GALOIS_LOG_VERBOSE("Partition loading: File to read is {}", input_file);

  // load partition
  switch (partition_scheme) {
  case galois::graphs::GNNPartitionScheme::kOEC:
    return galois::cuspPartitionGraph<GnnOEC, char, void>(
        input_file, galois::CUSP_CSR, galois::CUSP_CSR, true, "", "", false, 1);
  case galois::graphs::GNNPartitionScheme::kCVC:
    return galois::cuspPartitionGraph<GnnCVC, char, void>(
        input_file, galois::CUSP_CSR, galois::CUSP_CSR, true, "", "", false, 1);
  case galois::graphs::GNNPartitionScheme::kOCVC:
    return galois::cuspPartitionGraph<GenericCVC, char, void>(
        input_file, galois::CUSP_CSR, galois::CUSP_CSR, true, "", "", false, 1);
  default:
    GALOIS_LOG_FATAL("Error: partition scheme specified is invalid");
    return nullptr;
  }
}

} // end namespace

namespace galois {
namespace graphs {
GNNFloat* gnn_matrix_to_sync_            = nullptr;
size_t gnn_matrix_to_sync_column_length_ = 0;
galois::DynamicBitSet bitset_graph_aggregate;
#ifdef GALOIS_ENABLE_GPU
struct CUDA_Context* cuda_ctx_for_sync;
unsigned layer_number_to_sync;
#endif
} // namespace graphs
} // namespace galois

galois::graphs::GNNGraph::GNNGraph(const std::string& dataset_name,
                                   GNNPartitionScheme partition_scheme,
                                   bool has_single_class_label)
    : GNNGraph(galois::default_gnn_dataset_path, dataset_name, partition_scheme,
               has_single_class_label) {}

galois::graphs::GNNGraph::GNNGraph(const std::string& input_directory,
                                   const std::string& dataset_name,
                                   GNNPartitionScheme partition_scheme,
                                   bool has_single_class_label)
    : input_directory_(input_directory) {
  GALOIS_LOG_VERBOSE("[{}] Constructing partitioning for {}", host_id_,
                     dataset_name);
  // save host id
  host_id_ = galois::runtime::getSystemNetworkInterface().ID;
  host_prefix_ =
      std::string("[") +
      std::to_string(galois::runtime::getSystemNetworkInterface().ID) +
      std::string("] ");
  // load partition
  partitioned_graph_ =
      LoadPartition(input_directory_, dataset_name, partition_scheme);

  // read additional graph data
  ReadLocalLabels(dataset_name, has_single_class_label);
  ReadLocalFeatures(dataset_name);
  ReadLocalMasks(dataset_name);

  // init gluon from the partitioned graph
  sync_substrate_ =
      std::make_unique<galois::graphs::GluonSubstrate<GNNDistGraph>>(
          *partitioned_graph_, host_id_,
          galois::runtime::getSystemNetworkInterface().Num, false,
          partitioned_graph_->cartesianGrid());
  bitset_graph_aggregate.resize(partitioned_graph_->size());

  // read in entire graph topology
  ReadWholeGraph(dataset_name);
  // init norm factors using the whole graph topology
  InitNormFactor();

#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    // allocate/copy data structures over to GPU
    GALOIS_LOG_VERBOSE("[{}] Initializing GPU memory", host_id_);
    InitGPUMemory();

    // initialize CUDA context
    cuda_ctx_ = get_CUDA_context(host_id_);
    if (!init_CUDA_context(cuda_ctx_, ::gpudevice)) {
      GALOIS_DIE("Failed to initialize CUDA context");
    }
    PartitionedGraphInfo g_info;
    GetPartitionedGraphInfo(g_info);
    load_graph_CUDA_GNN(cuda_ctx_, g_info,
                        galois::runtime::getSystemNetworkInterface().Num);
  }
#endif
}

bool galois::graphs::GNNGraph::IsValidForPhaseCompleteRange(
    const unsigned lid, const galois::GNNPhase current_phase) const {
  // only use ranges if they're complete
  // convert to gid first
  size_t gid = partitioned_graph_->getGID(lid);

  // select range to use based on phase
  const GNNRange* range_to_use;
  switch (current_phase) {
  case GNNPhase::kTrain:
    range_to_use = &global_training_mask_range_;
    break;
  case GNNPhase::kValidate:
    range_to_use = &global_validation_mask_range_;
    break;
  case GNNPhase::kTest:
    range_to_use = &global_testing_mask_range_;
    break;
  default:
    GALOIS_LOG_FATAL("Invalid phase used");
    range_to_use = nullptr;
  }

  // if within range, it is valid
  // there is an assumption here that ranges are contiguous; may not
  // necessarily be the case in all inputs in which case using the mask is
  // required (but less cache efficient)
  if (range_to_use->begin <= gid && gid < range_to_use->end) {
    return true;
  } else {
    return false;
  }
}

bool galois::graphs::GNNGraph::IsValidForPhaseMasked(
    const unsigned lid, const galois::GNNPhase current_phase) const {
  // select mask to use based on phase
  const std::vector<char>* mask_to_use;
  switch (current_phase) {
  case GNNPhase::kTrain:
    mask_to_use = &local_training_mask_;
    break;
  case GNNPhase::kValidate:
    mask_to_use = &local_validation_mask_;
    break;
  case GNNPhase::kTest:
    mask_to_use = &local_testing_mask_;
    break;
  default:
    GALOIS_LOG_FATAL("Invalid phase used");
    mask_to_use = nullptr;
  }

  return (*mask_to_use)[lid];
}

void galois::graphs::GNNGraph::AggregateSync(
    GNNFloat* matrix_to_sync, const size_t matrix_column_size) const {
  // set globals for the sync substrate
  gnn_matrix_to_sync_               = matrix_to_sync;
  gnn_matrix_to_sync_column_length_ = matrix_column_size;
  sync_substrate_
      ->sync<writeSource, readAny, GNNSumAggregate, Bitset_graph_aggregate>(
          "GraphAggregateSync");
}

#ifdef GALOIS_ENABLE_GPU
void galois::graphs::GNNGraph::AggregateSync(
    GNNFloat* matrix_to_sync, const size_t matrix_column_size,
    const unsigned layer_number) const {
  size_t layer_input_mtx_column_size =
      getLayerInputMatrixColumnSize(cuda_ctx_, layer_number);
  size_t layer_output_mtx_column_size =
      getLayerOutputMatrixColumnSize(cuda_ctx_, layer_number);
  // set globals for the sync substrate
  gnn_matrix_to_sync_               = matrix_to_sync;
  gnn_matrix_to_sync_column_length_ = matrix_column_size;
  cuda_ctx_for_sync                 = cuda_ctx_;
  layer_number_to_sync              = layer_number;
  // TODO bitset setting
  // call sync
  cudaSetLayerInputOutput(cuda_ctx_, matrix_to_sync, matrix_column_size, size(),
                          layer_number);

  if (gnn_matrix_to_sync_column_length_ == layer_input_mtx_column_size) {
    sync_substrate_->sync<writeSource, readAny, GNNSumAggregate_layer_input>(
        "GraphAggregateSync", gnn_matrix_to_sync_column_length_);
  } else if (gnn_matrix_to_sync_column_length_ ==
             layer_output_mtx_column_size) {
    sync_substrate_->sync<writeSource, readAny, GNNSumAggregate_layer_output>(
        "GraphAggregateSync", gnn_matrix_to_sync_column_length_);
  } else {
    GALOIS_LOG_FATAL("Column size of the synchronized matrix does not"
                     " match to the column size of the CUDA context");
  }
}
#endif

void galois::graphs::GNNGraph::UniformNodeSample(float droprate) {
  galois::do_all(
      galois::iterate(begin_owned(), end_owned()), [&](const NodeIterator& x) {
        partitioned_graph_->getData(*x) = sample_rng_.DoBernoulli(droprate);
      });
  // TODO(loc) GPU
  // TODO(loc) sync the flags across all machines to have same sample on all of
  // them
}

// TODO(loc) does not work in a distributed setting: assumes the partitioned
// graph is the entire graph
void galois::graphs::GNNGraph::GraphSAINTSample(size_t num_roots,
                                                size_t walk_depth) {
  // reset sample
  galois::do_all(galois::iterate(begin(), end()),
                 [&](size_t n) { partitioned_graph_->getData(n) = 0; });

  galois::on_each([&](size_t thread_id, size_t num_threads) {
    size_t my_start = 0;
    size_t my_end   = 0;
    std::tie(my_start, my_end) =
        galois::block_range(size_t{0}, num_roots, thread_id, num_threads);
    size_t thread_roots = my_end - my_start;
    size_t train_range  = global_training_mask_range_.size;
    // init RNG
    drand48_data seed_struct;
    srand48_r(sample_rng_.GetRandomNumber() * thread_id * num_threads,
              &seed_struct);

    for (size_t root_num = 0; root_num < thread_roots; root_num++) {
      // pick a random training node root at random (with replacement);
      size_t root = 0;
      while (true) {
        long int rand_num;
        lrand48_r(&seed_struct, &rand_num);
        root = global_training_mask_range_.begin + (rand_num % train_range);
        if (IsValidForPhase(root, GNNPhase::kTrain)) {
          break;
        }
      }
      // mark this root as sampled
      SetSampledNode(root);
      assert(IsInSampledGraph(root));

      // sample more nodes based on depth of the walk
      for (size_t current_depth = 0; current_depth < walk_depth;
           current_depth++) {
        // pick random edge, mark sampled, swap roots
        EdgeIterator first_edge = EdgeBegin(root);
        size_t num_edges        = std::distance(first_edge, EdgeEnd(root));
        if (num_edges == 0) {
          break;
        }

        // must select training neighbor: if it doesn't, then ignore and
        // continue
        // To prevent infinite loop in case node has NO training neighbor,
        // this implementation will not loop until one is found and will
        // not find full depth if it doesn't find any training nodes randomly
        long int rand_num;
        lrand48_r(&seed_struct, &rand_num);
        EdgeIterator selected_edge = first_edge + (rand_num % num_edges);
        size_t candidate_dest      = EdgeDestination(selected_edge);

        // TODO(loc) another possibility is to just pick it anyways regardless
        // but don't mark it as sampled, though this would lead to disconnected
        // graph
        if (IsValidForPhase(candidate_dest, GNNPhase::kTrain)) {
          SetSampledNode(candidate_dest);
          assert(IsInSampledGraph(candidate_dest));
          root = candidate_dest;
        }
      }
    }
  });
}

void galois::graphs::GNNGraph::ReadLocalLabels(const std::string& dataset_name,
                                               bool has_single_class_label) {
  GALOIS_LOG_VERBOSE("[{}] Reading labels from disk...", host_id_);
  std::string filename;
  if (has_single_class_label) {
    filename = input_directory_ + dataset_name + "-labels.txt";
  } else {
    filename = input_directory_ + dataset_name + "-mlabels.txt";
  }

  // read file header, save num label classes while at it
  std::ifstream file_stream;
  file_stream.open(filename, std::ios::in);
  size_t num_nodes;
  file_stream >> num_nodes >> num_label_classes_ >> std::ws;
  assert(num_nodes == partitioned_graph_->globalSize());
  if (host_id_ == 0) {
    galois::gInfo("Number of label classes is ", num_label_classes_);
  }

  // allocate memory for labels
  if (has_single_class_label) {
    // single-class (one-hot) label for each vertex: N x 1
    using_single_class_labels_ = true;
    local_ground_truth_labels_.resize(partitioned_graph_->size());
  } else {
    // multi-class label for each vertex: N x num classes
    using_single_class_labels_ = false;
    local_ground_truth_labels_.resize(partitioned_graph_->size() *
                                      num_label_classes_);
  }

  size_t cur_gid              = 0;
  size_t found_local_vertices = 0;
  // each line contains a set of 0s and 1s
  std::string read_line;

  // loop through all labels of the graph
  while (std::getline(file_stream, read_line)) {
    // only process label if this node is local
    if (partitioned_graph_->isLocal(cur_gid)) {
      uint32_t cur_lid = partitioned_graph_->getLID(cur_gid);
      // read line as bitset of 0s and 1s
      std::istringstream label_stream(read_line);
      int cur_bit;
      // bitset size is # of label classes
      for (size_t cur_class = 0; cur_class < num_label_classes_; ++cur_class) {
        // read a bit
        label_stream >> cur_bit;

        if (has_single_class_label) {
          // no label
          if (cur_bit == -1) {
            local_ground_truth_labels_[cur_lid] = num_label_classes_;
            break;
          }

          // in single class, only 1 bit is set in bitset; that represents the
          // class to take
          if (cur_bit != 0) {
            // set class and break (assumption is that's the only bit that is
            // set)
            local_ground_truth_labels_[cur_lid] = cur_class;
            break;
          }
        } else {
          // else the entire bitset needs to be copied over to the label array
          // TODO this can possibly be saved all at once rather than bit by bit?
          local_ground_truth_labels_[cur_lid * num_label_classes_ + cur_class] =
              cur_bit;
        }
      }
      found_local_vertices++;
    }
    // always increment cur_gid
    cur_gid++;
  }

  file_stream.close();

  galois::gInfo(host_prefix_, "Read ", found_local_vertices, " labels (",
                local_ground_truth_labels_.size() * double{4} / (1 << 30),
                " GB)");
  GALOIS_LOG_ASSERT(found_local_vertices == partitioned_graph_->size());
}

void galois::graphs::GNNGraph::ReadLocalFeatures(
    const std::string& dataset_name) {
  GALOIS_LOG_VERBOSE("[{}] Reading features from disk...", host_id_);

  // read in dimensions of features, specifically node feature length
  size_t num_global_vertices;

  std::string file_dims = input_directory_ + dataset_name + "-dims.txt";
  std::ifstream ifs;
  ifs.open(file_dims, std::ios::in);
  ifs >> num_global_vertices >> node_feature_length_;
  ifs.close();

  GALOIS_LOG_ASSERT(num_global_vertices == partitioned_graph_->globalSize());
  GALOIS_LOG_VERBOSE("[{}] N x D: {} x {}", host_id_, num_global_vertices,
                     node_feature_length_);

  // memory for all features of all nodes in graph
  // TODO read features without loading entire feature file into memory; this
  // is quite inefficient
  std::unique_ptr<GNNFloat[]> full_feature_set =
      std::make_unique<GNNFloat[]>(num_global_vertices * node_feature_length_);

  // read in all features
  std::ifstream file_stream;
  std::string feature_file = input_directory_ + dataset_name + "-feats.bin";
  file_stream.open(feature_file, std::ios::binary | std::ios::in);
  file_stream.read((char*)full_feature_set.get(), sizeof(GNNFloat) *
                                                      num_global_vertices *
                                                      node_feature_length_);
  file_stream.close();

  // allocate memory for local features
  local_node_features_.resize(partitioned_graph_->size() *
                              node_feature_length_);

  // copy over features for local nodes only
  size_t num_kept_vertices = 0;
  for (size_t gid = 0; gid < num_global_vertices; gid++) {
    if (partitioned_graph_->isLocal(gid)) {
      // copy over feature vector
      std::copy(full_feature_set.get() + gid * node_feature_length_,
                full_feature_set.get() + (gid + 1) * node_feature_length_,
                &local_node_features_[partitioned_graph_->getLID(gid) *
                                      node_feature_length_]);
      num_kept_vertices++;
    }
  }
  full_feature_set.reset();

  galois::gInfo(host_prefix_, "Read ", local_node_features_.size(),
                " features (",
                local_node_features_.size() * double{4} / (1 << 30), " GB)");
  GALOIS_LOG_ASSERT(num_kept_vertices == partitioned_graph_->size());
}

//! Helper function to read masks from file into the appropriate structures
//! given a name, mask type, and arrays to save into
size_t galois::graphs::GNNGraph::ReadLocalMasksFromFile(
    const std::string& dataset_name, const std::string& mask_type,
    GNNRange* mask_range, char* masks) {
  size_t range_begin;
  size_t range_end;

  // read mask range
  std::string mask_filename =
      input_directory_ + dataset_name + "-" + mask_type + "_mask.txt";
  std::ifstream mask_stream;
  mask_stream.open(mask_filename, std::ios::in);
  mask_stream >> range_begin >> range_end >> std::ws;
  GALOIS_LOG_ASSERT(range_begin <= range_end);

  // set the range object
  mask_range->begin = range_begin;
  mask_range->end   = range_end;
  mask_range->size  = range_end - range_begin;

  size_t cur_line_num = 0;
  // valid nodes on this host
  size_t local_sample_count = 0;
  // this tracks TOTAL # of valid nodes in this group (not necessarily valid
  // ones on this host)
  size_t valid_count = 0;
  std::string line;
  // each line is a number signifying if mask is set for the vertex
  while (std::getline(mask_stream, line)) {
    std::istringstream mask_stream(line);
    // only examine vertices/lines in range
    if (cur_line_num >= range_begin && cur_line_num < range_end) {
      unsigned mask = 0;
      mask_stream >> mask;
      if (mask == 1) {
        valid_count++;
        if (partitioned_graph_->isLocal(cur_line_num)) {
          masks[partitioned_graph_->getLID(cur_line_num)] = 1;
          local_sample_count++;
        }
      }
    }
    cur_line_num++;
  }
  mask_stream.close();

  if (valid_count != mask_range->size) {
    // overlapping masks: need to actually check the masks rather than use
    // ranges
    if (!incomplete_masks_) {
      galois::gInfo(
          "Masks are not contained in range: must actually check mask");
    }
    incomplete_masks_ = true;
  }

  return valid_count;
}

void galois::graphs::GNNGraph::ReadLocalMasks(const std::string& dataset_name) {
  // allocate the memory for the local masks
  local_training_mask_.resize(partitioned_graph_->size());
  local_validation_mask_.resize(partitioned_graph_->size());
  local_testing_mask_.resize(partitioned_graph_->size());

  if (dataset_name == "reddit") {
    // TODO reddit is hardcode handled at the moment; better way to not do
    // this?
    global_training_mask_range_   = {.begin = 0, .end = 153431, .size = 153431};
    global_validation_mask_range_ = {
        .begin = 153431, .end = 153431 + 23831, .size = 23831};
    global_testing_mask_range_ = {
        .begin = 177262, .end = 177262 + 55703, .size = 55703};

    // training
    for (size_t i = global_training_mask_range_.begin;
         i < global_training_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_training_mask_[partitioned_graph_->getLID(i)] = 1;
      }
    }

    // validation
    for (size_t i = global_validation_mask_range_.begin;
         i < global_validation_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_validation_mask_[partitioned_graph_->getLID(i)] = 1;
      }
    }

    // testing
    for (size_t i = global_testing_mask_range_.begin;
         i < global_testing_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_testing_mask_[partitioned_graph_->getLID(i)] = 1;
      }
    }
  } else {
    // XXX i can get local sample counts from here if i need it
    size_t valid_train = ReadLocalMasksFromFile(dataset_name, "train",
                                                &global_training_mask_range_,
                                                local_training_mask_.data());
    size_t valid_val   = ReadLocalMasksFromFile(dataset_name, "val",
                                              &global_validation_mask_range_,
                                              local_validation_mask_.data());
    size_t valid_test  = ReadLocalMasksFromFile(dataset_name, "test",
                                               &global_testing_mask_range_,
                                               local_testing_mask_.data());
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gInfo("Valid # training nodes is ", valid_train);
      galois::gInfo("Valid # validation nodes is ", valid_val);
      galois::gInfo("Valid # test nodes is ", valid_test);
    }
  }
}

void galois::graphs::GNNGraph::ReadWholeGraph(const std::string& dataset_name) {
  std::string input_file = input_directory_ + dataset_name + ".csgr";
  GALOIS_LOG_VERBOSE("[{}] Reading entire graph: file to read is {}", host_id_,
                     input_file);
  galois::graphs::readGraph(whole_graph_, input_file);
}

void galois::graphs::GNNGraph::InitNormFactor() {
  GALOIS_LOG_VERBOSE("[{}] Initializing norm factors", host_id_);
  norm_factors_.resize(partitioned_graph_->size(), 0.0);
  degree_norm_.resize(partitioned_graph_->size(), 0.0);
  CalculateFullNormFactor();
}

void galois::graphs::GNNGraph::CalculateFullNormFactor() {
  norm_factors_.assign(partitioned_graph_->size(), 0.0);

  // get the norm factor contribution for each node based on the GLOBAL graph
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), partitioned_graph_->size()),
      [&](size_t local_id) {
        // translate lid into gid to get global degree
        size_t global_id = partitioned_graph_->getGID(local_id);
        // +1 because simulated self edge
        size_t global_degree = whole_graph_.edge_end(global_id) -
                               whole_graph_.edge_begin(global_id) + 1;
        // only set if non-zero
        if (global_degree != 0) {
          norm_factors_[local_id] =
              1.0 / std::sqrt(static_cast<float>(global_degree));
          degree_norm_[local_id] = 1.0 / static_cast<float>(global_degree);
        }
      },
      galois::loopname("CalculateFullNormFactor"));
}

void galois::graphs::GNNGraph::CalculateSpecialNormFactor(bool is_sampled,
                                                          bool is_inductive) {
  if (galois::runtime::getSystemNetworkInterface().Num > 1) {
    GALOIS_LOG_FATAL("cannot run special norm factor in dist setting yet");
  }

  norm_factors_.assign(partitioned_graph_->size(), 0.0);

  // get the norm factor contribution for each node based on the GLOBAL graph
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), partitioned_graph_->size()),
      [&](size_t local_id) {
        // ignore node if not valid
        if (is_sampled && is_inductive) {
          if (!IsValidForPhase(local_id, GNNPhase::kTrain) ||
              !IsInSampledGraph(local_id)) {
            return;
          }
        } else if (is_sampled) {
          if (!IsInSampledGraph(local_id)) {
            return;
          }
        } else if (is_inductive) {
          if (!IsValidForPhase(local_id, GNNPhase::kTrain)) {
            return;
          }
        }

        size_t degree = 0;

        // TODO(loc) make this work in a distributed setting; assuming
        // whole graph is present on single host at the moment
        for (EdgeIterator e = EdgeBegin(local_id); e != EdgeEnd(local_id);
             e++) {
          size_t dest = EdgeDestination(e);
          if (is_sampled && is_inductive) {
            if (!IsValidForPhase(dest, GNNPhase::kTrain) ||
                !IsInSampledGraph(dest)) {
              continue;
            }
          } else if (is_sampled) {
            if (!IsInSampledGraph(dest)) {
              continue;
            }
          } else if (is_inductive) {
            if (!IsValidForPhase(dest, GNNPhase::kTrain)) {
              continue;
            }
          } else {
            GALOIS_LOG_WARN(
                "Why is special norm factor called if not sampled/inductive?");
          }
          degree += 1;
        }

        // only set if non-zero
        if (degree != 0) {
          norm_factors_[local_id] = 1.0 / std::sqrt(static_cast<float>(degree));
          degree_norm_[local_id]  = 1.0 / static_cast<float>(degree);
        }
      },
      galois::loopname("CalculateSpecialNormFactor"));
}

float galois::graphs::GNNGraph::GetGlobalAccuracy(
    PointerWithSize<GNNFloat> predictions, GNNPhase phase) {
  // No GPU version yet, but this is where it would be
  return GetGlobalAccuracy(predictions, phase, false);
}

float galois::graphs::GNNGraph::GetGlobalAccuracy(
    PointerWithSize<GNNFloat> predictions, GNNPhase phase, bool sampling) {
  // No GPU version yet, but this is where it would be
  return GetGlobalAccuracyCPU(predictions, phase, sampling);
}

float galois::graphs::GNNGraph::GetGlobalAccuracyCPU(
    PointerWithSize<GNNFloat> predictions, GNNPhase phase, bool sampling) {
  if (is_single_class_label()) {
    return GetGlobalAccuracyCPUSingle(predictions, phase, sampling);
  } else {
    return GetGlobalAccuracyCPUMulti(predictions, phase, sampling);
  }
}

float galois::graphs::GNNGraph::GetGlobalAccuracyCPUSingle(
    PointerWithSize<GNNFloat> predictions, GNNPhase phase, bool sampling) {
  // check owned nodes' accuracy
  assert((num_label_classes_ * size()) == predictions.size());
  num_correct_.reset();
  total_checked_.reset();

  galois::do_all(
      galois::iterate(begin_owned(), end_owned()),
      [&](const unsigned lid) {
        if (IsValidForPhase(lid, phase)) {
          if (sampling) {
            if (phase == GNNPhase::kTrain && !IsInSampledGraph(lid)) {
              return;
            }
          }

          total_checked_ += 1;
          // get prediction by getting max
          size_t predicted_label = galois::MaxIndex(
              num_label_classes_, &(predictions[lid * num_label_classes_]));
          // check against ground truth and track accordingly
          // TODO static cast used here is dangerous
          if (predicted_label ==
              static_cast<size_t>(GetSingleClassLabel(lid))) {
            num_correct_ += 1;
          }
        }
      },
      // steal on as some threads may have nothing to work on
      galois::steal(), galois::loopname("GlobalAccuracy"));

  size_t global_correct = num_correct_.reduce();
  size_t global_checked = total_checked_.reduce();

  GALOIS_LOG_VERBOSE("Accuracy: {} / {}", global_correct, global_checked);

  return static_cast<float>(global_correct) /
         static_cast<float>(global_checked);
}

float galois::graphs::GNNGraph::GetGlobalAccuracyCPUMulti(
    PointerWithSize<GNNFloat> predictions, GNNPhase phase, bool sampling) {

  const GNNLabel* full_ground_truth = GetMultiClassLabel(0);
  assert(predictions.size() == (num_label_classes_ * size()));

  size_t global_true_positive  = 0;
  size_t global_true_negative  = 0;
  size_t global_false_positive = 0;
  size_t global_false_negative = 0;
  size_t global_f1_score       = 0;

  // per class check
  for (size_t label_class = 0; label_class < num_label_classes_;
       label_class++) {
    local_true_positive_.reset();
    local_true_negative_.reset();
    local_false_positive_.reset();
    local_false_negative_.reset();

    // loop through all *owned* nodes (do not want to overcount)
    galois::do_all(
        galois::iterate(begin_owned(), end_owned()),
        [&](const unsigned lid) {
          if (IsValidForPhase(lid, phase)) {
            if (sampling) {
              if (phase == GNNPhase::kTrain && !IsInSampledGraph(lid)) {
                return;
              }
            }

            size_t label_index = lid * num_label_classes_ + label_class;

            GNNLabel true_label = full_ground_truth[label_index];
            GNNLabel prediction_is_positive =
                (predictions[label_index] > 0.5) ? 1 : 0;

            if (true_label && prediction_is_positive) {
              local_true_positive_ += 1;
            } else if (true_label && !prediction_is_positive) {
              local_false_negative_ += 1;
            } else if (!true_label && prediction_is_positive) {
              local_false_positive_ += 1;
            } else if (!true_label && !prediction_is_positive) {
              local_true_negative_ += 1;
            } else {
              // all cases should be covered with clauses above, so it should
              // NEVER get here; adding it here just for sanity purposes
              GALOIS_LOG_FATAL(
                  "Logic error with true label and prediction label");
            }
          }
          total_checked_ += 1;
        },
        galois::steal(), galois::loopname("GlobalMultiAccuracy"));

    // reduce from accumulators across all hosts for this particular class
    size_t class_true_positives  = local_true_positive_.reduce();
    size_t class_false_positives = local_false_positive_.reduce();
    size_t class_true_negatives  = local_true_negative_.reduce();
    size_t class_false_negatives = local_false_negative_.reduce();

    // add to global counts
    global_true_positive += class_true_positives;
    global_false_positive += class_false_positives;
    global_true_negative += class_true_negatives;
    global_false_negative += class_false_negatives;

    // calculate precision, recall, and f1 score for this class
    // ternery op used to avoid division by 0
    double class_precision =
        (class_true_positives + class_true_negatives) > 0
            ? static_cast<double>(class_true_positives) /
                  (class_true_positives + class_false_positives)
            : 0.0;
    double class_recall =
        (class_true_positives + class_false_negatives) > 0
            ? static_cast<double>(class_true_positives) /
                  (class_true_positives + class_false_negatives)
            : 0.0;
    double class_f1_score = (class_precision + class_recall) > 0
                                ? (2.0 * (class_precision * class_recall)) /
                                      (class_precision + class_recall)
                                : 0.0;

    global_f1_score += class_f1_score;
  } // end label class loop

  // GALOIS_LOG_WARN("{} {} {} {}", global_true_positive, global_true_negative,
  // global_false_positive, global_false_negative);

  // double global_f1_macro_score = global_f1_score / num_label_classes_;

  // micro = considers all classes for precision/recall
  double global_micro_precision =
      (global_true_positive + global_true_negative) > 0
          ? static_cast<double>(global_true_positive) /
                (global_true_positive + global_false_positive)
          : 0.0;
  double global_micro_recall =
      (global_true_positive + global_false_negative) > 0
          ? static_cast<double>(global_true_positive) /
                (global_true_positive + global_false_negative)
          : 0.0;

  double global_f1_micro_score =
      (global_micro_precision + global_micro_recall) > 0
          ? (2.0 * (global_micro_precision * global_micro_recall)) /
                (global_micro_precision + global_micro_recall)
          : 0.0;

  return global_f1_micro_score;
}

#ifdef GALOIS_ENABLE_GPU
void galois::graphs::GNNGraph::InitGPUMemory() {
  // create int casted CSR
  uint64_t* e_index_ptr = partitioned_graph_->row_start_ptr();
  uint32_t* e_dest_ptr  = partitioned_graph_->edge_dst_ptr();

  // + 1 because first element is 0 in BLAS CSRs
  std::vector<int> e_index(partitioned_graph_->size() + 1);
  std::vector<int> e_dest(partitioned_graph_->sizeEdges());

  // set in parallel
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), partitioned_graph_->size() + 1),
      [&](size_t index) {
        if (index != 0) {
          if (e_index_ptr[index - 1] >
              static_cast<size_t>(std::numeric_limits<int>::max())) {
            GALOIS_LOG_FATAL("{} is too big a number for int arrays on GPUs",
                             e_index_ptr[index - 1]);
          }
          e_index[index] = static_cast<int>(e_index_ptr[index - 1]);
        } else {
          e_index[index] = 0;
        }
      },
      galois::loopname("GPUEdgeIndexConstruction"));
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), partitioned_graph_->sizeEdges()),
      [&](size_t edge) {
        if (e_dest_ptr[edge] >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
          GALOIS_LOG_FATAL("{} is too big a number for int arrays on GPUs",
                           e_dest_ptr[edge]);
        }

        e_dest[edge] = static_cast<int>(e_dest_ptr[edge]);
      },
      galois::loopname("GPUEdgeDestConstruction"));

  gpu_memory_.SetGraphTopology(e_index, e_dest);
  e_index.clear();
  e_dest.clear();

  gpu_memory_.SetFeatures(local_node_features_, node_feature_length_);
  gpu_memory_.SetLabels(local_ground_truth_labels_);
  gpu_memory_.SetMasks(local_training_mask_, local_validation_mask_,
                       local_testing_mask_);
  gpu_memory_.SetNormFactors(norm_factors_);
}

void galois::graphs::GNNGraph::InitLayerVectorMetaObjects(
    size_t layer_number, unsigned num_hosts, size_t infl_in_size,
    size_t infl_out_size) {
  init_CUDA_layer_vector_meta_obj(cuda_ctx_, layer_number, num_hosts, size(),
                                  infl_in_size, infl_out_size);
}

void galois::graphs::GNNGraph::ResizeLayerVector(size_t num_layers) {
  resize_CUDA_layer_vector(cuda_ctx_, num_layers);
}
#endif
