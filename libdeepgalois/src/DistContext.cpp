#include "deepgalois/DistContext.h"
#include "deepgalois/utils.h"
#include "deepgalois/configs.h"

namespace deepgalois {
DistContext::DistContext() : DistContext(false) { syncSubstrate = NULL; }

DistContext::~DistContext() {}

void DistContext::saveDistGraph(DGraph* a) {
  partitionedGraph = a;

  // construct lgraph from underlying lc csr graph
  // TODO fix this so i don't have more than 1 copy of graph in memory
  this->lGraph = new Graph();
  this->lGraph->allocateFrom(a->size(), a->sizeEdges());
  this->lGraph->constructNodes();

  galois::do_all(
      galois::iterate((size_t)0, a->size()),
      [&](const auto src) {
        this->lGraph->fixEndEdge(src, *a->edge_end(src));
        index_t idx = *(a->edge_begin(src));

        for (auto e = a->edge_begin(src); e != a->edge_end(src); e++) {
          const auto dst = a->getEdgeDst(e);
          this->lGraph->constructEdge(idx++, dst, 0);
        }
      },
      galois::loopname("lgraphcopy"));
}

// TODO move to reader class
size_t DistContext::read_labels(bool isSingleClassLabel,
                                std::string dataset_str) {
  DGraph* dGraph         = DistContext::partitionedGraph;
  this->usingSingleClass = isSingleClassLabel;
  unsigned myID          = galois::runtime::getSystemNetworkInterface().ID;
  galois::gPrint("[", myID, "] Reading labels from disk...\n");

  std::string filename = path + dataset_str + "-labels.txt";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  size_t m;
  // read file header
  in >> m >> this->num_classes >> std::ws;
  assert(m == dGraph->globalSize());

  // size of labels should be # local nodes
  if (isSingleClassLabel) {
    galois::gPrint("[", myID, "] One hot labels...\n");
    // single-class (one-hot) label for each vertex: N x 1
    this->h_labels = new label_t[dGraph->size()];
  } else {
    galois::gPrint("[", myID, "] Multi-class labels...\n");
    this->h_labels = new label_t[dGraph->size() * this->num_classes];
    // multi-class label for each vertex: N x E
  }

  uint32_t foundVertices = 0;
  unsigned v             = 0;
  // each line contains a set of 0s and 1s
  while (std::getline(in, line)) {
    // only bother if local node
    if (dGraph->isLocal(v)) {
      std::istringstream label_stream(line);
      unsigned x;
      // for each class
      for (size_t idx = 0; idx < this->num_classes; ++idx) {
        // check if that class is labeled
        label_stream >> x;

        // diff between single and multi class
        if (isSingleClassLabel) {
          if (x != 0) {
            // set local id
            this->h_labels[dGraph->getLID(v)] = idx;
            foundVertices++;
            break;
          }
        } else {
          this->h_labels[dGraph->getLID(v) * this->num_classes + idx] = x;
          foundVertices++;
        }
      }
    }
    // always increment v
    v++;
  }

  in.close();

  // print the number of vertex classes
  galois::gPrint("[", myID,
                 "] Done with labels, unique label counts: ", num_classes,
                 "; set ", foundVertices, " nodes\n");

  return num_classes;
}

// TODO move to reader class
size_t DistContext::read_features(std::string dataset_str) {
  DGraph* dGraph = DistContext::partitionedGraph;
  unsigned myID  = galois::runtime::getSystemNetworkInterface().ID;
  galois::gPrint("[", myID, "] Reading features from disk...\n");

  std::string filename = path + dataset_str + ".ft";
  std::ifstream in;
  size_t m; // m = number of vertices
  // dimension read
  std::string file_dims = path + dataset_str + "-dims.txt";
  std::ifstream ifs;
  ifs.open(file_dims, std::ios::in);
  ifs >> m >> this->feat_len >> std::ws;
  ifs.close();

  galois::gPrint("[", myID, "] N x D: ", m, " x ", feat_len, "\n");

  // TODO read in without using 2 in-memory buffers
  // full read feats to load into h_feats
  float_t* fullFeats = new float_t[m * feat_len];
  // actual stored feats
  h_feats = new float_t[dGraph->size() * feat_len];

  // read in full feats
  filename = path + dataset_str + "-feats.bin";
  in.open(filename, std::ios::binary | std::ios::in);
  in.read((char*)fullFeats, sizeof(float_t) * m * feat_len);
  in.close();

  // get the local ids we want
  size_t count = 0;
  for (size_t i = 0; i < m; i++) {
    if (dGraph->isLocal(i)) {
      // h_feats[count * feat_len] = fullFeats[i];
      std::copy(fullFeats + i * DistContext::feat_len,
                fullFeats + (i + 1) * DistContext::feat_len,
                &this->h_feats[dGraph->getLID(i) * DistContext::feat_len]);
      count++;
    }
  }
  GALOIS_ASSERT(count == dGraph->size());
  free(fullFeats);

  galois::gPrint("[", myID, "] Done with features, feature length: ", feat_len,
                 "\n");

  return feat_len;
}

// TODO move to reader class/reuse reader class somehow
size_t DistContext::read_masks(std::string dataset_str, std::string mask_type,
                               size_t n, size_t& begin, size_t& end,
                               mask_t* masks, DGraph* dGraph) {
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;

  bool dataset_found = false;
  for (int i = 0; i < NUM_DATASETS; i++) {
    if (dataset_str == dataset_names[i]) {
      dataset_found = true;
      break;
    }
  }
  if (!dataset_found) {
    GALOIS_DIE("Dataset currently not supported");
  }
  size_t i             = 0;
  size_t sample_count  = 0;
  std::string filename = path + dataset_str + "-" + mask_type + "_mask.txt";

  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  in >> begin >> end >> std::ws;
  while (std::getline(in, line)) {
    std::istringstream mask_stream(line);
    if (i >= begin && i < end) {
      unsigned mask = 0;
      mask_stream >> mask;
      if (mask == 1) {
        // only bother if it's local
        if (dGraph->isLocal(i)) {
          masks[dGraph->getLID(i)] = 1;
          sample_count++;
        }
      }
    }
    i++;
  }
  galois::gPrint("[", myID, "] ", mask_type, "_mask range: [", begin, ", ", end,
                 ") Number of valid samples: ", sample_count, "(",
                 (float)sample_count / (float)n * (float)100, "\%)\n");
  in.close();
  return sample_count;
}

float_t* DistContext::get_in_ptr() { return &h_feats[0]; }

void DistContext::initializeSyncSubstrate() {
  DistContext::syncSubstrate = new galois::graphs::GluonSubstrate<DGraph>(
      *DistContext::partitionedGraph,
      galois::runtime::getSystemNetworkInterface().ID,
      galois::runtime::getSystemNetworkInterface().Num, false);
}

void DistContext::allocNormFactor() {
#ifdef USE_MKL
  this->normFactors.resize(partitionedGraph->sizeEdges());
#else
  this->normFactors.resize(partitionedGraph->size());
#endif
}

void DistContext::allocNormFactorSub(int subID) {
#ifdef USE_MKL
  this->normFactorsSub.resize(partitionedSubgraphs[subID]->sizeEdges());
#else
  this->normFactorsSub.resize(partitionedSubgraphs[subID]->size());
#endif
}

void DistContext::constructNormFactor(deepgalois::Context* globalContext) {
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
  galois::gPrint("[", myID, "] Norm factor construction\n");
  // using original graph to get ids
  Graph* wholeGraph = globalContext->getFullGraph();

  allocNormFactor();
  // this is for testing purposes
  // galois::do_all(galois::iterate((size_t)0, partitionedGraph->size()),
  //  [&] (unsigned i) {
  //    this->normFactors[i] = 0;
  //  }
  //);

#ifdef USE_MKL
  galois::do_all(
      galois::iterate((size_t)0, partitionedGraph->size()),
      [&](unsigned i) {
        float_t c_i = std::sqrt(
            float_t(wholeGraph->get_degree(partitionedGraph->getGID(i))));

        for (auto e = partitionedGraph->edge_begin(i);
             e != partitionedGraph->edge_end(i); e++) {
          const auto j = partitionedGraph->getEdgeDst(e);
          float_t c_j  = std::sqrt(
              float_t(wholeGraph->get_degree(partitionedGraph->getGID(j))));

          if (c_i == 0.0 || c_j == 0.0) {
            this->normFactors[*e] = 0.0;
          } else {
            this->normFactors[*e] = 1.0 / (c_i * c_j);
          }
        }
      },
      galois::loopname("NormCountingEdge"));
#else
  galois::do_all(
      galois::iterate((size_t)0, partitionedGraph->size()),
      [&](unsigned v) {
        auto degree  = wholeGraph->get_degree(partitionedGraph->getGID(v));
        float_t temp = std::sqrt(float_t(degree));
        if (temp == 0.0) {
          this->normFactors[v] = 0.0;
        } else {
          this->normFactors[v] = 1.0 / temp;
        }
      },
      galois::loopname("NormCountingNode"));
#endif
  galois::gPrint("[", myID, "] Norm factor construction done \n");
}

void DistContext::constructNormFactorSub(int subgraphID) {
  // galois::gPrint("Sub norm factor construction\n");
  // right now norm factor based on subgraph
  // TODO fix this for dist execution

  allocNormFactorSub(subgraphID);

  Graph& graphToUse = *partitionedSubgraphs[subgraphID];
  graphToUse.degree_counting();

  // TODO using partitioned subgraph rather than whoel graph; i.e. dist
  // setting wrong
#ifdef USE_MKL
  galois::do_all(
      galois::iterate((size_t)0, graphToUse.size()),
      [&](unsigned i) {
        // float_t c_i =
        // std::sqrt(float_t(wholeGraph->get_degree(partitionedGraph->getGID(i))));
        float_t c_i = std::sqrt(float_t(graphToUse.get_degree(i)));

        for (index_t e = graphToUse.edge_begin(i); e != graphToUse.edge_end(i);
             e++) {
          const auto j = graphToUse.getEdgeDst(e);
          float_t c_j  = std::sqrt(float_t(graphToUse.get_degree(j)));

          if (c_i == 0.0 || c_j == 0.0) {
            this->normFactorsSub[e] = 0.0;
          } else {
            this->normFactorsSub[e] = 1.0 / (c_i * c_j);
          }
        }
      },
      galois::loopname("NormCountingEdge"));
#else
  galois::do_all(
      galois::iterate((size_t)0, graphToUse.size()),
      [&](unsigned v) {
        // auto degree = wholeGraph->get_degree(partitionedGraph->getGID(v));
        auto degree  = graphToUse.get_degree(v);
        float_t temp = std::sqrt(float_t(degree));
        if (temp == 0.0) {
          this->normFactorsSub[v] = 0.0;
        } else {
          this->normFactorsSub[v] = 1.0 / temp;
        }
        // galois::gPrint(this->normFactorsSub[v], "\n");
      },
      galois::loopname("NormCountingNode"));
#endif
  // galois::gPrint("Sub norm factor construction done\n");
}
//! generate labels for the subgraph, m is subgraph size, mask
//! tells which vertices to use
void DistContext::constructSubgraphLabels(size_t m, const mask_t* masks) {
  if (DistContext::usingSingleClass) {
    DistContext::h_labels_subg.resize(m);
  } else {
    DistContext::h_labels_subg.resize(m * DistContext::num_classes);
  }
  size_t count = 0;
  // see which labels to copy over for this subgraph
  for (size_t i = 0; i < this->partitionedGraph->size(); i++) {
    if (masks[i] == 1) {
      if (DistContext::usingSingleClass) {
        DistContext::h_labels_subg[count] = h_labels[i];
      } else {
        std::copy(
            DistContext::h_labels + i * DistContext::num_classes,
            DistContext::h_labels + (i + 1) * DistContext::num_classes,
            &DistContext::h_labels_subg[count * DistContext::num_classes]);
      }
      // galois::gPrint("l ", (float)DistContext::h_labels_subg[count], "\n");
      count++;
    }
  }
  GALOIS_ASSERT(count == m);
}

//! generate input features for the subgraph, m is subgraph size,
//! masks tells which vertices to use
void DistContext::constructSubgraphFeatures(size_t m, const mask_t* masks) {
  size_t count = 0;
  DistContext::h_feats_subg.resize(m * feat_len);
  for (size_t i = 0; i < this->partitionedGraph->size(); i++) {
    if (masks[i] == 1) {
      std::copy(DistContext::h_feats + i * DistContext::feat_len,
                DistContext::h_feats + (i + 1) * DistContext::feat_len,
                &DistContext::h_feats_subg[count * DistContext::feat_len]);
      // for (unsigned a = 0; a < DistContext::feat_len; a++) {
      //  if (h_feats_subg[count * DistContext::feat_len + a] != 0) {
      //    galois::gPrint(h_feats_subg[count * DistContext::feat_len + a],
      //    " ");
      //  }
      //}
      // galois::gPrint("\n");
      count++;
    }
  }
  GALOIS_ASSERT(count == m);
}

galois::graphs::GluonSubstrate<DGraph>* DistContext::getSyncSubstrate() {
  return DistContext::syncSubstrate;
}

//! allocate memory for subgraphs (don't actually build them)
void DistContext::allocateSubgraphs(int num_subgraphs, unsigned max_size) {
  this->partitionedSubgraphs.resize(num_subgraphs);
  for (int i = 0; i < num_subgraphs; i++) {
    this->partitionedSubgraphs[i] = new Graph();
    this->partitionedSubgraphs[i]->set_max_size(max_size);
  }
}

bool DistContext::isOwned(unsigned gid) {
  return this->partitionedGraph->isOwned(gid);
}

bool DistContext::isLocal(unsigned gid) {
  return this->partitionedGraph->isLocal(gid);
}

unsigned DistContext::getGID(unsigned lid) {
  return this->partitionedGraph->getGID(lid);
}

unsigned DistContext::getLID(unsigned gid) {
  return this->partitionedGraph->getLID(gid);
}

} // namespace deepgalois
