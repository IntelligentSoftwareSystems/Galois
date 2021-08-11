// XXX include net interface if necessary
#include "galois/Logging.h"
#include "galois/graphs/ReadGraph.h"
#include "galois/graphs/GNNGraph.h"
#include "galois/GNNMath.h"
#include "galois/graphs/DegreeSyncStructures.h"
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

// Sync structure variables; global to get around sync structure
// limitations at the moment
namespace galois {
namespace graphs {
GNNFloat* gnn_matrix_to_sync_            = nullptr;
size_t gnn_matrix_to_sync_column_length_ = 0;
size_t subgraph_size_                    = 0;
//! For synchronization of graph aggregations
galois::DynamicBitSet bitset_graph_aggregate;
galois::LargeArray<uint32_t>* gnn_lid_to_sid_pointer_ = nullptr;
size_t num_active_layer_rows_                         = 0;
uint32_t* gnn_degree_vec_1_;
uint32_t* gnn_degree_vec_2_;

galois::DynamicBitSet bitset_sample_flag_;

//! For synchronization of sampled degrees
galois::DynamicBitSet bitset_sampled_degrees_;
std::vector<galois::LargeArray<uint32_t>>* gnn_sampled_out_degrees_;

#ifdef GALOIS_ENABLE_GPU
struct CUDA_Context* cuda_ctx_for_sync;
struct CUDA_Context* cuda_ctx;
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
  // reverse edges
  partitioned_graph_->ConstructIncomingEdges();

  galois::gInfo(host_prefix_, "Number of local proxies is ",
                partitioned_graph_->size());
  galois::gInfo(host_prefix_, "Number of local edges is ",
                partitioned_graph_->sizeEdges());

  // read additional graph data
  if (dataset_name != "ogbn-papers100M-remap") {
    ReadLocalLabels(dataset_name, has_single_class_label);
  } else {
    galois::gInfo("Remapped ogbn 100M");
    ReadLocalLabelsBin(dataset_name);
  }
  ReadLocalFeatures(dataset_name);
  ReadLocalMasks(dataset_name);

  // init gluon from the partitioned graph
  sync_substrate_ =
      std::make_unique<galois::graphs::GluonSubstrate<GNNDistGraph>>(
          *partitioned_graph_, host_id_,
          galois::runtime::getSystemNetworkInterface().Num, false,
          partitioned_graph_->cartesianGrid());
  bitset_graph_aggregate.resize(partitioned_graph_->size());

  // init norm factors (involves a sync call)
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
  case GNNPhase::kOther:
    GALOIS_LOG_FATAL("no range for other");
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
  const GNNMask* mask_to_use;
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
  case GNNPhase::kOther:
    if (valid_other_ == 0) {
      return false;
    }
    mask_to_use = &other_mask_;
    break;
  case GNNPhase::kBatch:
    mask_to_use = &local_minibatch_mask_;
    break;
  default:
    GALOIS_LOG_FATAL("Invalid phase used");
    mask_to_use = nullptr;
  }

  return (*mask_to_use)[lid];
}

void galois::graphs::GNNGraph::AggregateSync(
    GNNFloat* matrix_to_sync, const size_t matrix_column_size, bool is_backward,
    uint32_t active_row_boundary) const {
  gnn_matrix_to_sync_               = matrix_to_sync;
  gnn_matrix_to_sync_column_length_ = matrix_column_size;
  subgraph_size_                    = active_size();
  num_active_layer_rows_            = active_row_boundary;
  if (!use_subgraph_ && !use_subgraph_view_) {
    // set globals for the sync substrate
    if (!is_backward) {
      if (use_timer_) {
        sync_substrate_->sync<writeSource, readAny, GNNSumAggregate,
                              Bitset_graph_aggregate>("GraphAggregateSync");
      } else {
        sync_substrate_->sync<writeSource, readAny, GNNSumAggregate,
                              Bitset_graph_aggregate>("Ignore");
      }
    } else {
      galois::StatTimer clubbed_timer("Sync_BackwardSync", "Gluon");
      clubbed_timer.start();
      sync_substrate_->sync<writeDestination, readAny, GNNSumAggregate,
                            Bitset_graph_aggregate>(
          "BackwardGraphAggregateSync");
      clubbed_timer.stop();
    }
  } else {
    // setup the SID to LID map for the sync substrate to use (SID != LID)
    gnn_lid_to_sid_pointer_ = subgraph_->GetLIDToSIDPointer();

    if (!is_backward) {
      if (use_timer_) {
        sync_substrate_->sync<writeSource, readAny, GNNSampleSumAggregate,
                              Bitset_graph_aggregate>("GraphAggregateSync");
      } else {
        sync_substrate_->sync<writeSource, readAny, GNNSampleSumAggregate,
                              Bitset_graph_aggregate>("Ignore");
      }
    } else {
      galois::StatTimer clubbed_timer("Sync_BackwardSync", "Gluon");
      clubbed_timer.start();
      sync_substrate_->sync<writeDestination, readAny, GNNSampleSumAggregate,
                            Bitset_graph_aggregate>(
          "BackwardGraphAggregateSync");
      clubbed_timer.stop();
    }
  }
}

#ifdef GALOIS_ENABLE_GPU
void galois::graphs::GNNGraph::AggregateSyncGPU(
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

  // XXX no timer if use_timer is off
  if (gnn_matrix_to_sync_column_length_ == layer_input_mtx_column_size) {
    if (use_timer_) {
      sync_substrate_->sync<writeSource, readAny, GNNSumAggregate_layer_input>(
          "GraphAggregateSync", gnn_matrix_to_sync_column_length_);
    } else {
      sync_substrate_->sync<writeSource, readAny, GNNSumAggregate_layer_input>(
          "Ignore", gnn_matrix_to_sync_column_length_);
    }
  } else if (gnn_matrix_to_sync_column_length_ ==
             layer_output_mtx_column_size) {
    if (use_timer_) {
      sync_substrate_->sync<writeSource, readAny, GNNSumAggregate_layer_output>(
          "GraphAggregateSync", gnn_matrix_to_sync_column_length_);
    } else {
      sync_substrate_->sync<writeSource, readAny, GNNSumAggregate_layer_output>(
          "Ignore", gnn_matrix_to_sync_column_length_);
    }
  } else {
    GALOIS_LOG_FATAL("Column size of the synchronized matrix does not"
                     " match to the column size of the CUDA context");
  }
}
#endif
void galois::graphs::GNNGraph::ReadLocalLabelsBin(
    const std::string& dataset_name) {
  GALOIS_LOG_VERBOSE("[{}] Reading labels from disk...", host_id_);

  std::ifstream file_stream;
  file_stream.open(input_directory_ + dataset_name + "-labels-dims.txt",
                   std::ios::in);
  size_t num_nodes;
  file_stream >> num_nodes >> num_label_classes_ >> std::ws;
  assert(num_nodes == partitioned_graph_->globalSize());
  if (host_id_ == 0) {
    galois::gInfo("Number of label classes is ", num_label_classes_);
  }
  file_stream.close();

  std::string filename = input_directory_ + dataset_name + "-labels.bin";
  std::ifstream file_stream_bin;
  file_stream_bin.open(filename, std::ios::binary | std::ios::in);

  std::vector<GNNLabel> all_labels(num_nodes);
  // read all labels into a vector
  file_stream_bin.read((char*)all_labels.data(), sizeof(GNNLabel) * num_nodes);

  using_single_class_labels_ = true;
  local_ground_truth_labels_.resize(partitioned_graph_->size());

  galois::GAccumulator<size_t> found_local_vertices;
  found_local_vertices.reset();

  // save only local ones; can do in parallel as well
  // assumes -1 already dealt with
  galois::do_all(galois::iterate(size_t{0}, partitioned_graph_->size()),
                 [&](size_t lid) {
                   local_ground_truth_labels_[lid] = all_labels[GetGID(lid)];
                   found_local_vertices += 1;
                 });

  size_t fli = found_local_vertices.reduce();
  galois::gInfo(host_prefix_, "Read ", fli, " labels (",
                local_ground_truth_labels_.size() * double{4} / (1 << 30),
                " GB)");
  GALOIS_LOG_ASSERT(fli == partitioned_graph_->size());
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
  galois::GAccumulator<size_t> num_kept_vertices;
  num_kept_vertices.reset();
  galois::do_all(
      galois::iterate(size_t{0}, num_global_vertices), [&](size_t gid) {
        if (partitioned_graph_->isLocal(gid)) {
          // copy over feature vector
          std::copy(full_feature_set.get() + gid * node_feature_length_,
                    full_feature_set.get() + (gid + 1) * node_feature_length_,
                    &local_node_features_[partitioned_graph_->getLID(gid) *
                                          node_feature_length_]);
          num_kept_vertices += 1;
        }
      });
  full_feature_set.reset();

  galois::gInfo(host_prefix_, "Read ", local_node_features_.size(),
                " features (",
                local_node_features_.size() * double{4} / (1 << 30), " GB)");
  GALOIS_LOG_ASSERT(num_kept_vertices.reduce() == partitioned_graph_->size());
}

//! Helper function to read masks from file into the appropriate structures
//! given a name, mask type, and arrays to save into
size_t galois::graphs::GNNGraph::ReadLocalMasksFromFile(
    const std::string& dataset_name, const std::string& mask_type,
    GNNRange* mask_range, std::vector<char>* masks) {
  size_t range_begin;
  size_t range_end;

  // read mask range
  std::string mask_filename =
      input_directory_ + dataset_name + "-" + mask_type + "_mask.txt";
  bool train_is_on = false;
  if (mask_type == "train") {
    train_is_on = true;
  }

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
          (*masks)[partitioned_graph_->getLID(cur_line_num)] = 1;
          local_sample_count++;
        }
        if (train_is_on) {
          global_training_mask_[cur_line_num] = 1;
        }
      }
    }
    cur_line_num++;
  }
  mask_stream.close();

  if (train_is_on) {
    global_training_count_ = valid_count;
  }

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

size_t galois::graphs::GNNGraph::FindOtherMask() {
  galois::GAccumulator<size_t> other_accum;
  other_accum.reset();
  other_mask_.resize(partitioned_graph_->size());

  galois::do_all(
      galois::iterate(size_t{0}, partitioned_graph_->size()),
      [&](size_t local_id) {
        if (!IsValidForPhase(local_id, GNNPhase::kTrain) &&
            !IsValidForPhase(local_id, GNNPhase::kValidate) &&
            !IsValidForPhase(local_id, GNNPhase::kTest)) {
          other_mask_[local_id] = 1;
          other_accum += 1;
        }
      },
      galois::loopname("FindOtherMask"));
  return other_accum.reduce();
}

void galois::graphs::GNNGraph::ReadLocalMasks(const std::string& dataset_name) {
  // allocate the memory for the local masks
  global_training_mask_.resize(partitioned_graph_->globalSize());
  local_training_mask_.resize(partitioned_graph_->size());
  local_validation_mask_.resize(partitioned_graph_->size());
  local_testing_mask_.resize(partitioned_graph_->size());

  if (dataset_name == "reddit") {
    global_training_count_ = 153431;

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
      global_training_mask_[i] = 1;
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
  } else if (dataset_name == "ogbn-papers100M-remap") {
    global_training_count_ = 1207178;

    global_training_mask_range_ = {.begin = 0, .end = 1207178, .size = 1207178};
    global_validation_mask_range_ = {
        .begin = 1207178, .end = 1207178 + 125264, .size = 125264};
    global_testing_mask_range_ = {
        .begin = 1332442, .end = 1332442 + 214337, .size = 214337};
    // training
    for (size_t i = global_training_mask_range_.begin;
         i < global_training_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_training_mask_[partitioned_graph_->getLID(i)] = 1;
      }
      global_training_mask_[i] = 1;
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
    valid_other_ = FindOtherMask();
    GALOIS_LOG_ASSERT(valid_other_ <= 109513177);
  } else {
    size_t valid_train = ReadLocalMasksFromFile(dataset_name, "train",
                                                &global_training_mask_range_,
                                                &local_training_mask_);
    size_t valid_val   = ReadLocalMasksFromFile(dataset_name, "val",
                                              &global_validation_mask_range_,
                                              &local_validation_mask_);
    size_t valid_test  = ReadLocalMasksFromFile(dataset_name, "test",
                                               &global_testing_mask_range_,
                                               &local_testing_mask_);
    valid_other_       = FindOtherMask();
    // the "other" set of nodes that don't fall into any classification
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gInfo("Valid # training nodes is ", valid_train);
      galois::gInfo("Valid # validation nodes is ", valid_val);
      galois::gInfo("Valid # test nodes is ", valid_test);
      galois::gInfo("Valid # other nodes is ", valid_other_);
    }
  }
}

void galois::graphs::GNNGraph::InitNormFactor() {
  GALOIS_LOG_VERBOSE("[{}] Initializing norm factors", host_id_);
  global_degrees_.resize(partitioned_graph_->size(), 0.0);
  global_train_degrees_.resize(partitioned_graph_->size(), 0.0);
  CalculateFullNormFactor();
#ifdef GALOIS_ENABLE_GPU
  if (device_personality == DevicePersonality::GPU_CUDA) {
    gpu_memory_.InitNormFactor(partitioned_graph_->size());
  }
#endif
}

void galois::graphs::GNNGraph::CalculateFullNormFactor() {
  // TODO(loc) reset all degrees if this is called multiple times?
  // get the norm factor contribution for each node based on the GLOBAL graph
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), partitioned_graph_->size()),
      [&](size_t src) {
        for (auto edge_iter = partitioned_graph_->edge_begin(src);
             edge_iter != partitioned_graph_->edge_end(src); edge_iter++) {
          // count degrees for all + train/other
          size_t dest = GetEdgeDest(edge_iter);
          if (IsValidForPhase(dest, GNNPhase::kTrain) ||
              IsValidForPhase(dest, GNNPhase::kOther)) {
            global_train_degrees_[src] += 1;
          }
          global_degrees_[src] += 1;
        }
      },
      galois::loopname("CalculateLocalDegrees"));
  // degree sync
  gnn_degree_vec_1_ = global_train_degrees_.data();
  gnn_degree_vec_2_ = global_degrees_.data();
  sync_substrate_->sync<writeSource, readAny, InitialDegreeSync>(
      "InitialDegreeSync");
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
    PointerWithSize<GNNFloat> predictions, GNNPhase phase, bool) {
  // check owned nodes' accuracy
  num_correct_.reset();
  total_checked_.reset();

  galois::do_all(
      // will only loop over sampled nodes if sampling is on
      galois::iterate(begin_owned(), end_owned()),
      // this is possibly the subgraph id
      [&](const unsigned node_id) {
        if (IsValidForPhase(node_id, phase)) {
          total_checked_ += 1;
          // get prediction by getting max
          // note the use of node_id here: lid only used to check original
          // labels
          size_t predicted_label = galois::MaxIndex(
              num_label_classes_, &(predictions[node_id * num_label_classes_]));
          // check against ground truth and track accordingly
          // TODO static cast used here is dangerous
          if (predicted_label ==
              static_cast<size_t>(GetSingleClassLabel(node_id))) {
            num_correct_ += 1;
          }
        }
      },
      // steal on as some threads may have nothing to work on
      galois::steal());

  size_t global_correct = num_correct_.reduce();
  size_t global_checked = total_checked_.reduce();

  GALOIS_LOG_DEBUG("Sub: {}, Accuracy: {} / {}", use_subgraph_, global_correct,
                   global_checked);

  return static_cast<float>(global_correct) /
         static_cast<float>(global_checked);
}
std::pair<uint32_t, uint32_t> galois::graphs::GNNGraph::GetBatchAccuracy(
    PointerWithSize<GNNFloat> predictions) {
  // check owned nodes' accuracy
  num_correct_.reset();
  total_checked_.reset();

  galois::do_all(
      // will only loop over sampled nodes if sampling is on
      galois::iterate(begin_owned(), end_owned()),
      // this is possibly the subgraph id
      [&](const unsigned node_id) {
        if (IsValidForPhase(node_id, GNNPhase::kBatch)) {
          total_checked_ += 1;
          size_t predicted_label = galois::MaxIndex(
              num_label_classes_, &(predictions[node_id * num_label_classes_]));
          if (predicted_label ==
              static_cast<size_t>(GetSingleClassLabel(node_id))) {
            num_correct_ += 1;
          }
        }
      },
      // steal on as some threads may have nothing to work on
      galois::steal(), galois::loopname("GlobalAccuracy"));

  size_t global_correct = num_correct_.reduce();
  size_t global_checked = total_checked_.reduce();

  return std::make_pair(global_correct, global_checked);
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

////////////////////////////////////////////////////////////////////////////////

void galois::graphs::GNNGraph::InitializeSamplingData(size_t num_layers,
                                                      bool choose_all) {
  subgraph_ = std::make_unique<GNNSubgraph>(partitioned_graph_->size());
  sample_node_timestamps_.create(partitioned_graph_->size(),
                                 std::numeric_limits<uint32_t>::max());
  edge_sample_status_.resize(num_layers);
  for (size_t i = 0; i < num_layers; i++) {
    edge_sample_status_[i].resize(partitioned_graph_->sizeEdges());
  }
  sampled_edges_.resize(partitioned_graph_->sizeEdges());
  // this is to hold the degree of a sampled graph considering all hosts; yes,
  // memory wise this is slightly problematic possibly, but each layer is its
  // own subgraph
  if (!choose_all) {
    sampled_out_degrees_.resize(num_layers);
    for (galois::LargeArray<uint32_t>& array : sampled_out_degrees_) {
      array.create(partitioned_graph_->size());
    }
  } else {
    subgraph_choose_all_ = true;
  }
  definitely_sampled_nodes_.resize(partitioned_graph_->size());
  master_offset_accum_.resize(num_layers + 1);
  mirror_offset_accum_.resize(num_layers + 1);
  sample_master_offsets_.resize(num_layers + 1, 0);
  sample_mirror_offsets_.resize(num_layers + 1, 0);
}

size_t galois::graphs::GNNGraph::SetupNeighborhoodSample(GNNPhase seed_phase) {
  DisableSubgraph();

  if (!bitset_sample_flag_.size()) {
    bitset_sample_flag_.resize(size());
  }
  bitset_sample_flag_.ParallelReset();
  definitely_sampled_nodes_.ParallelReset();

  galois::do_all(
      galois::iterate(begin_owned(), end_owned()),
      [&](const NodeIterator& x) {
        if (IsValidForPhase(*x, seed_phase)) {
          SetSampledNode(*x);
          bitset_sample_flag_.set(*x);
          definitely_sampled_nodes_.set(*x);
        } else {
          UnsetSampledNode(*x);
        }
      },
      galois::loopname("InitialSeedSetting"));
  // unsets nodes set in previous iterations; for some reason they get
  // synchronized along  with everything else even though bitset sample flag
  // should prevent it (that, or it's because they don't get sync'd that they
  // remain the same)
  galois::do_all(galois::iterate(end_owned(), end()),
                 [&](const NodeIterator& x) { UnsetSampledNode(*x); });

  // clear node timestamps
  galois::StatTimer fill_time("ClearFillTime");
  fill_time.start();
  galois::ParallelSTL::fill(sample_node_timestamps_.begin(),
                            sample_node_timestamps_.end(),
                            std::numeric_limits<uint32_t>::max());
  galois::ParallelSTL::fill(sample_master_offsets_.begin(),
                            sample_master_offsets_.end(), 0);
  galois::ParallelSTL::fill(sample_mirror_offsets_.begin(),
                            sample_mirror_offsets_.end(), 0);
  fill_time.stop();

  for (unsigned i = 0; i < master_offset_accum_.size(); i++) {
    master_offset_accum_[i].reset();
    mirror_offset_accum_[i].reset();
  }

  // clear all sampled edges
  galois::StatTimer ctime("ClearSampleEdges");
  ctime.start();
  for (galois::DynamicBitSet& edge_layer : edge_sample_status_) {
    edge_layer.ParallelReset();
  }
  ctime.stop();
  //  galois::do_all(
  //      galois::iterate(edge_sample_status_.begin(),
  //      edge_sample_status_.end()),
  //      [&](galois::DynamicBitSet& edge_layer) { edge_layer.reset(); },
  //      galois::loopname("ClearSampleEdges"));

  sampled_edges_.ParallelReset();

  // reset all degrees
  if (!subgraph_choose_all_) {
    galois::StatTimer cad_timer("ClearAllDegrees");
    cad_timer.start();
    for (galois::LargeArray<uint32_t>& array : sampled_out_degrees_) {
      galois::ParallelSTL::fill(array.begin(), array.end(), 0);
    }
    cad_timer.stop();
  }

  if (!bitset_sampled_degrees_.size()) {
    bitset_sampled_degrees_.resize(partitioned_graph_->size());
  }
  bitset_sampled_degrees_.reset();

  // Seed nodes sync
  if (use_timer_) {
    sync_substrate_
        ->sync<writeSource, readAny, SampleFlagSync, SampleFlagBitset>(
            "SeedNodeSample");
  } else {
    sync_substrate_
        ->sync<writeSource, readAny, SampleFlagSync, SampleFlagBitset>(
            "Ignore");
  }

  galois::GAccumulator<unsigned> local_seed_count;
  local_seed_count.reset();
  galois::GAccumulator<unsigned> master_offset;
  master_offset.reset();
  galois::GAccumulator<unsigned> mirror_offset;
  mirror_offset.reset();
  // count # of seed nodes
  galois::do_all(
      galois::iterate(begin(), end()),
      [&](const NodeIterator& x) {
        if (IsInSampledGraph(x)) {
          if (*x < *end_owned()) {
            master_offset += 1;
          } else {
            // mirror
            mirror_offset += 1;
          }

          // galois::gInfo(host_prefix_, "Seed node is ", GetGID(*x));
          local_seed_count += 1;
          // 0 = seed node
          sample_node_timestamps_[*x] = 0;
        }
      },
      galois::loopname("SeedNodeOffsetCounting"));

  sample_master_offsets_[0] = master_offset.reduce();
  sample_mirror_offsets_[0] = mirror_offset.reduce();

  return local_seed_count.reduce();
}

size_t galois::graphs::GNNGraph::SampleAllEdges(size_t agg_layer_num,
                                                bool inductive_subgraph,
                                                size_t timestamp) {
  DisableSubgraph();

  galois::do_all(
      galois::iterate(begin(), end()),
      [&](const NodeIterator& src_iter) {
        // only operate on if sampled
        if (IsInSampledGraph(src_iter)) {
          // marks ALL edges of nodes that connect to train/other nodes
          for (auto edge_iter : partitioned_graph_->edges(*src_iter)) {
            // total += 1;
            if (inductive_subgraph) {
              if (!IsValidForPhase(partitioned_graph_->getEdgeDst(edge_iter),
                                   GNNPhase::kTrain) &&
                  !IsValidForPhase(partitioned_graph_->getEdgeDst(edge_iter),
                                   GNNPhase::kOther)) {
                continue;
              }
            }

            MakeEdgeSampled(edge_iter, agg_layer_num);
            uint32_t dest = partitioned_graph_->getEdgeDst(edge_iter);
            if (!IsInSampledGraph(dest)) {
              bitset_sample_flag_.set(dest);
            }
            definitely_sampled_nodes_.set(*src_iter);
            definitely_sampled_nodes_.set(dest);
          }
        }
      },
      galois::steal(), galois::loopname("ChooseAllEdges"));

  // update nodes, then communicate update to all hosts so that they can
  // continue the exploration
  galois::do_all(
      galois::iterate(size_t{0}, bitset_sample_flag_.size()),
      [&](uint32_t new_node_id) {
        if (bitset_sample_flag_.test(new_node_id)) {
          SetSampledNode(new_node_id);
        }
      },
      galois::loopname("NeighborhoodSampleSet"));

  if (use_timer_) {
    sync_substrate_
        ->sync<writeDestination, readAny, SampleFlagSync, SampleFlagBitset>(
            "SampleFlag");
  } else {
    sync_substrate_
        ->sync<writeDestination, readAny, SampleFlagSync, SampleFlagBitset>(
            "Ignore");
  }

  galois::GAccumulator<unsigned> local_sample_count;
  local_sample_count.reset();
  // count # of seed nodes
  galois::do_all(galois::iterate(begin(), end()), [&](const NodeIterator& x) {
    if (IsInSampledGraph(x)) {
      local_sample_count += 1;
      if (sample_node_timestamps_[*x] == std::numeric_limits<uint32_t>::max()) {
        if (x < end_owned()) {
          // owned nodes that are activated on other hosts shoudl always
          // be activated because it's responsible for keeping others in
          // sync during comms; ignoring it = bad
          // TODO(gluon) make it so you don't have to deal with this
          // and just use host as a reducer point
          definitely_sampled_nodes_.set(*x);
        }
        sample_node_timestamps_[*x] = timestamp;
      }
    }
  });

  EnableSubgraphChooseAll();
  return local_sample_count.reduce();
}

size_t galois::graphs::GNNGraph::SampleEdges(size_t sample_layer_num,
                                             size_t num_to_sample,
                                             bool inductive_subgraph,
                                             size_t timestamp) {
  use_subgraph_      = false;
  use_subgraph_view_ = false;

  galois::do_all(
      galois::iterate(begin(), end()),
      [&](const NodeIterator& src_iter) {
        // only operate on if sampled
        if (IsInSampledGraph(src_iter)) {
          // chance of not uniformly choosing an edge of this node num_to_sample
          // times (degree norm is 1 / degree)
          double probability_of_reject;
          if (!inductive_subgraph) {
            probability_of_reject =
                std::pow(1 - GetGlobalDegreeNorm(*src_iter), num_to_sample);
          } else {
            probability_of_reject = std::pow(
                1 - GetGlobalTrainDegreeNorm(*src_iter), num_to_sample);
          }

          // loop through edges, turn "on" edge with some probability
          for (auto edge_iter : partitioned_graph_->edges(*src_iter)) {
            if (sample_rng_.DoBernoulli(probability_of_reject)) {
              if (inductive_subgraph) {
                // only take if node is training node or a node not classified
                // into train/test/val
                if (!IsValidForPhase(partitioned_graph_->getEdgeDst(edge_iter),
                                     GNNPhase::kTrain) &&
                    !IsValidForPhase(partitioned_graph_->getEdgeDst(edge_iter),
                                     GNNPhase::kOther)) {
                  continue;
                }
              }

              uint32_t edge_dst = partitioned_graph_->getEdgeDst(edge_iter);
              // if here, it means edge accepted; set sampled on, mark
              // as part of next set
              MakeEdgeSampled(edge_iter, sample_layer_num);
              if (!IsInSampledGraph(edge_dst)) {
                bitset_sample_flag_.set(edge_dst);
              }
              bitset_sampled_degrees_.set(*src_iter);
              definitely_sampled_nodes_.set(*src_iter);
              definitely_sampled_nodes_.set(edge_dst);
              // degree increment
              sampled_out_degrees_[sample_layer_num][*src_iter]++;
            }
          }
        }
      },
      galois::steal(), galois::loopname("NeighborhoodSample"));

  // update nodes, then communicate update to all hosts so that they can
  // continue the exploration
  galois::do_all(
      galois::iterate(size_t{0}, bitset_sample_flag_.size()),
      [&](uint32_t new_node_id) {
        if (bitset_sample_flag_.test(new_node_id)) {
          SetSampledNode(new_node_id);
        }
      },
      galois::loopname("NeighborhoodSampleSet"));

  // why not read source? even if it doesn't need to sample anything, it needs
  // to know that it's active so that subgraph construction can proceed
  // correctly
  if (use_timer_) {
    sync_substrate_
        ->sync<writeDestination, readAny, SampleFlagSync, SampleFlagBitset>(
            "SampleFlag");
  } else {
    sync_substrate_
        ->sync<writeDestination, readAny, SampleFlagSync, SampleFlagBitset>(
            "Ignore");
  }

  // count sampled node size
  galois::GAccumulator<unsigned> local_sample_count;
  local_sample_count.reset();
  // count # of seed nodes
  galois::do_all(galois::iterate(begin(), end()), [&](const NodeIterator& x) {
    if (IsInSampledGraph(x)) {
      local_sample_count += 1;
      if (sample_node_timestamps_[*x] == std::numeric_limits<uint32_t>::max()) {
        if (x < end_owned()) {
          // owned nodes that are activated on other hosts shoudl always
          // be activated because it's responsible for keeping others in
          // sync during comms; ignoring it = bad
          // TODO(gluon) make it so you don't have to deal with this
          // and just use host as a reducer point
          definitely_sampled_nodes_.set(*x);
        }
        sample_node_timestamps_[*x] = timestamp;
      }
    }
  });

  DisableSubgraphChooseAll();
  return local_sample_count.reduce();
}

//! Construct the subgraph from sampled edges and corresponding nodes
std::vector<unsigned>
galois::graphs::GNNGraph::ConstructSampledSubgraph(size_t num_sampled_layers,
                                                   bool use_view) {
  // false first so that the build process can use functions to access the
  // real graph
  DisableSubgraph();

  gnn_sampled_out_degrees_ = &sampled_out_degrees_;

  // first, sync the degres of the sampled edges across all hosts
  // read any because destinations need it to for reverse phase
  if (use_timer_) {
    sync_substrate_
        ->sync<writeSource, readAny, SubgraphDegreeSync, SubgraphDegreeBitset>(
            "SubgraphDegree");
  } else {
    sync_substrate_
        ->sync<writeSource, readAny, SubgraphDegreeSync, SubgraphDegreeBitset>(
            "Ignore");
  }

  galois::StatTimer offsets_n_rows_time("OffsetRowSubgraphTime");
  offsets_n_rows_time.start();
  galois::do_all(
      galois::iterate(begin(), end()),
      [&](const NodeIterator& x) {
        if (IsActiveInSubgraph(*x)) {
          if (sample_node_timestamps_[*x] !=
              std::numeric_limits<uint32_t>::max()) {
            if (*x < *end_owned()) {
              // master
              master_offset_accum_[sample_node_timestamps_[*x]] += 1;
            } else {
              // mirror
              mirror_offset_accum_[sample_node_timestamps_[*x]] += 1;
            }
          } else {
            GALOIS_LOG_FATAL(
                "should have been timestamped at some point if active");
          }
        }
      },
      galois::loopname("MasterMirrorOffset"));

  std::vector<unsigned> new_rows(master_offset_accum_.size());
  for (unsigned i = 0; i < master_offset_accum_.size(); i++) {
    sample_master_offsets_[i] = master_offset_accum_[i].reduce();
    sample_mirror_offsets_[i] = mirror_offset_accum_[i].reduce();
    new_rows[i] = sample_master_offsets_[i] + sample_mirror_offsets_[i];
    if (i > 0) {
      new_rows[i] += new_rows[i - 1];
    }
  }

  offsets_n_rows_time.stop();

  if (!use_view) {
    subgraph_->BuildSubgraph(*this, num_sampled_layers);
  } else {
    // a view only has lid<->sid mappings
    subgraph_->BuildSubgraphView(*this, num_sampled_layers);
  }

  sync_substrate_->SetupSubgraphMirrors(subgraph_->GetSubgraphMirrors(),
                                        use_timer_);

  // after this, this graph is a subgraph
  if (!use_view) {
    use_subgraph_ = true;
  } else {
    use_subgraph_view_ = true;
  }

  return new_rows;
}

size_t galois::graphs::GNNGraph::PrepareNextTrainMinibatch() {
  train_batcher_->GetNextMinibatch(&local_minibatch_mask_);
#ifndef NDEBUG
  size_t count = 0;
  // galois::gPrint("Minibatch : ");
  for (unsigned i = 0; i < local_minibatch_mask_.size(); i++) {
    if (local_minibatch_mask_[i]) {
      // galois::gPrint(partitioned_graph_->getGID(i), ",");
      count++;
    }
  }
  // galois::gPrint("\n");
  galois::gInfo(host_prefix(), "Batched nodes ", count);
#endif
  return SetupNeighborhoodSample(GNNPhase::kBatch);
}

size_t galois::graphs::GNNGraph::PrepareNextTestMinibatch() {
  test_batcher_->GetNextMinibatch(&local_minibatch_mask_);
  return SetupNeighborhoodSample(GNNPhase::kBatch);
}

////////////////////////////////////////////////////////////////////////////////

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
  gpu_memory_.AllocAggregateBitset(partitioned_graph_->size());
  gpu_memory_.SetGlobalTrainDegrees(global_train_degrees_);
  gpu_memory_.SetGlobalDegrees(global_degrees_);
}

void galois::graphs::GNNGraph::InitLayerVectorMetaObjects(
    size_t layer_number, unsigned num_hosts, size_t infl_in_size,
    size_t infl_out_size) {
  init_CUDA_layer_vector_meta_obj(cuda_ctx_, layer_number, num_hosts, size(),
                                  infl_in_size, infl_out_size);
}

void galois::graphs::GNNGraph::ResizeGPULayerVector(size_t num_layers) {
  resize_CUDA_layer_vector(cuda_ctx_, num_layers);
}
#endif
void galois::graphs::GNNGraph::ContiguousRemap(const std::string& new_name) {
  node_remapping_.resize(partitioned_graph_->size());

  uint32_t new_node_id = 0;

  // serial loops because new ID needs to be kept consistent
  // first, train nodes
  for (size_t cur_node = 0; cur_node < partitioned_graph_->size(); cur_node++) {
    if (IsValidForPhase(cur_node, GNNPhase::kTrain)) {
      node_remapping_[new_node_id++] = cur_node;
    }
  }
  galois::gInfo("Train nodes are from 0 to ", new_node_id);

  // second, val nodes
  uint32_t val_start = new_node_id;
  for (size_t cur_node = 0; cur_node < partitioned_graph_->size(); cur_node++) {
    if (IsValidForPhase(cur_node, GNNPhase::kValidate)) {
      node_remapping_[new_node_id++] = cur_node;
    }
  }
  galois::gInfo("Val nodes are from ", val_start, " to ", new_node_id, "(",
                new_node_id - val_start, ")");

  // third, test nodes
  uint32_t test_start = new_node_id;
  for (size_t cur_node = 0; cur_node < partitioned_graph_->size(); cur_node++) {
    if (IsValidForPhase(cur_node, GNNPhase::kTest)) {
      node_remapping_[new_node_id++] = cur_node;
    }
  }
  galois::gInfo("Test nodes are from ", test_start, " to ", new_node_id, "(",
                new_node_id - test_start, ")");

  // last, everything else
  uint32_t other_start = new_node_id;
  for (size_t cur_node = 0; cur_node < partitioned_graph_->size(); cur_node++) {
    if (IsValidForPhase(cur_node, GNNPhase::kOther)) {
      node_remapping_[new_node_id++] = cur_node;
    }
  }
  galois::gInfo("Other nodes are from ", other_start, " to ", new_node_id, "(",
                new_node_id - other_start, ")");
  GALOIS_LOG_ASSERT(new_node_id == partitioned_graph_->size());

  // remap features to match new node mapping, save to disk
  // std::vector<GNNFeature> remapped_features(local_node_features_.size());
  //// do all works because can copy in parallel
  // galois::do_all(
  //  galois::iterate(size_t{0}, partitioned_graph_->size()),
  //  [&] (size_t remap_node_id) {
  //    std::memcpy(
  //        &(remapped_features[remap_node_id * node_feature_length_]),
  //        &((local_node_features_.data())[node_remapping_[remap_node_id] *
  //        node_feature_length_]), node_feature_length_ * sizeof(GNNFeature));
  //  }
  //);
  //// sanity check
  // galois::do_all(
  //  galois::iterate(size_t{0}, partitioned_graph_->size()),
  //  [&] (size_t remap_node_id) {
  //    for (size_t i = 0; i < node_feature_length_; i++) {
  //      GALOIS_LOG_ASSERT(remapped_features[remap_node_id *
  //      node_feature_length_ + i] ==
  //                        local_node_features_[node_remapping_[remap_node_id]
  //                        * node_feature_length_ + i]);
  //    }
  //  }
  //);
  //// save to disk
  // std::ofstream write_file_stream;
  // std::string feature_file = input_directory_ + new_name + "-feats.bin";
  // galois::gPrint(feature_file, "\n");
  // write_file_stream.open(feature_file, std::ios::binary | std::ios::out);
  // write_file_stream.write((char*)remapped_features.data(), sizeof(GNNFeature)
  // *
  //                                                   partitioned_graph_->size()
  //                                                   * node_feature_length_);
  // write_file_stream.close();

  // std::ifstream file_stream;
  // file_stream.open(feature_file, std::ios::binary | std::ios::in);
  // file_stream.read((char*)remapped_features.data(), sizeof(GNNFloat) *
  //                                                  partitioned_graph_->size()
  //                                                  * node_feature_length_);
  // file_stream.close();
  //// sanity check again
  // galois::do_all(
  //  galois::iterate(size_t{0}, partitioned_graph_->size()),
  //  [&] (size_t remap_node_id) {
  //    for (size_t i = 0; i < node_feature_length_; i++) {
  //      GALOIS_LOG_ASSERT(remapped_features[remap_node_id *
  //      node_feature_length_ + i] ==
  //                        local_node_features_[node_remapping_[remap_node_id]
  //                        * node_feature_length_ + i]);
  //    }
  //  }
  //);
  // remapped_features.clear();

  // std::vector<GNNLabel> remapped_labels(local_ground_truth_labels_.size());
  //// save new labels order to disk (binary file)
  // galois::do_all(
  //  galois::iterate(size_t{0}, partitioned_graph_->size()),
  //  [&] (size_t remap_node_id) {
  //    remapped_labels[remap_node_id] =
  //    local_ground_truth_labels_[node_remapping_[remap_node_id]];
  //  }
  //);

  // std::string label_filename = input_directory_ + new_name + "-labels.bin";
  // std::ofstream label_write_stream;
  // label_write_stream.open(label_filename, std::ios::binary | std::ios::out);
  // label_write_stream.write((char*)remapped_labels.data(), sizeof(GNNLabel) *
  //                                                        partitioned_graph_->size());
  // label_write_stream.close();

  // galois::do_all(
  //  galois::iterate(size_t{0}, partitioned_graph_->size()),
  //  [&] (size_t remap_node_id) {
  //    remapped_labels[remap_node_id] =
  //    local_ground_truth_labels_[remap_node_id];
  //  }
  //);
  // ReadLocalLabelsBin(new_name);
  // galois::do_all(
  //  galois::iterate(size_t{0}, partitioned_graph_->size()),
  //  [&] (size_t remap_node_id) {
  //    GALOIS_LOG_ASSERT(local_ground_truth_labels_[remap_node_id] ==
  //    remapped_labels[node_remapping_[remap_node_id]]);
  //  }
  //);

  // save the mapping to a binary file for use by graph convert to deal with
  // the gr
  std::string label_filename = input_directory_ + new_name + "-mapping.bin";
  std::ofstream label_write_stream;
  label_write_stream.open(label_filename, std::ios::binary | std::ios::out);
  label_write_stream.write((char*)node_remapping_.data(),
                           sizeof(uint32_t) * node_remapping_.size());
  label_write_stream.close();
}
