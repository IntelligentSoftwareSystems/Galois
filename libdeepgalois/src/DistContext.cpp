#include "deepgalois/DistContext.h"
#include "deepgalois/utils.h"
#include "deepgalois/configs.h"

namespace deepgalois {
DistContext::DistContext() {}
DistContext::~DistContext() {}

size_t DistContext::read_labels(std::string dataset_str) {
  DGraph* dGraph = DistContext::partitionedGraph;
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
  galois::gPrint("[", myID, "] Reading labels from disk...\n");

  std::string filename = path + dataset_str + "-labels.txt";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  size_t m;
  // read file header
  in >> m >> num_classes >> std::ws;
  assert(m == dGraph->globalSize());
  // size of labels should be # local nodes
  h_labels = new label_t[dGraph->size()]; // single-class (one-hot) label for
                                          // each vertex: N x 1

  uint32_t foundVertices = 0;
  unsigned v             = 0;
  // each line contains a set of 0s and 1s
  while (std::getline(in, line)) {
    // only bother if local node
    if (dGraph->isLocal(v)) {
      std::istringstream label_stream(line);
      unsigned x;
      // for each class
      for (size_t idx = 0; idx < num_classes; ++idx) {
        // check if that class is labeled
        label_stream >> x;
        if (x != 0) {
          // set local id
          h_labels[dGraph->getLID(v)] = idx;
          foundVertices++;
          break;
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

size_t DistContext::read_features(std::string dataset_str) {
  DGraph* dGraph = DistContext::partitionedGraph;
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
  galois::gPrint("[", myID, "] Reading features from disk...\n");

  std::string filename = path + dataset_str + ".ft";
  std::ifstream in;
  std::string line;

  in.open(filename, std::ios::in);
  size_t m; // m = number of global vertices
  // header read
  in >> m >> feat_len >> std::ws;

//    std::string file_dims = path + dataset_str + "-dims.txt";
//    std::ifstream ifs;
//    ifs.open(file_dims, std::ios::in);
//    ifs >> m >> feat_len >> std::ws;
//    ifs.close();
//

  galois::gPrint("N x D: ", m, " x ", feat_len, "\n");
  // use local size, not global size
  h_feats = new float_t[dGraph->size() * feat_len];

  // loop through all features
  while (std::getline(in, line)) {
    std::istringstream edge_stream(line);
    unsigned u, v;
    float_t w;
    // vertex to set feature for
    edge_stream >> u;
    // only set if local
    if (dGraph->isLocal(u)) {
      // feature index
      edge_stream >> v;
      // actual feature
      edge_stream >> w;
      h_feats[dGraph->getLID(u) * feat_len + v] = w;
    }
    //galois::gPrint(u, "\n");
  }
  in.close();

  galois::gPrint("[", myID, "] Done with features, feature length: ", feat_len,
                 "\n");

  return feat_len;
}

size_t DistContext::read_masks(std::string dataset_str, std::string mask_type,
                               size_t n, size_t& begin, size_t& end,
                               mask_t* masks, DGraph* dGraph) {
  bool dataset_found = false;
  for (int i = 0; i < NUM_DATASETS; i++) {
    if (dataset_str == dataset_names[i]) {
      dataset_found = true;
      break;
    }
  }
  if (!dataset_found) {
    std::cout << "Dataset currently not supported\n";
    exit(1);
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
  std::cout << mask_type + "_mask range: [" << begin << ", " << end
            << ") Number of valid samples: " << sample_count << "("
            << (float)sample_count / (float)n * (float)100 << "\%)\n";
  in.close();
  return sample_count;
}

float_t* DistContext::get_in_ptr() { return &h_feats[0]; }

void DistContext::initializeSyncSubstrate() {
  DistContext::syncSubstrate = new galois::graphs::GluonSubstrate<DGraph>(
      *DistContext::partitionedGraph, galois::runtime::getSystemNetworkInterface().ID,
      galois::runtime::getSystemNetworkInterface().Num, false);
}

void DistContext::allocNormFactor() {
  if (!normFactors) {
#ifdef USE_MKL
    normFactors = new float_t[partitionedGraph->sizeEdges()];
#else
    normFactors = new float_t[partitionedGraph->size()];
#endif
  }
  if (!normFactors) {
    GALOIS_DIE("norm factors failed to be allocated");
  }
}

void DistContext::allocNormFactorSub(int subID) {
#ifdef USE_MKL
  normFactorsSub.resize(partitionedSubgraphs[subID]->sizeEdges());
#else
  normFactorsSub.resize(partitionedSubgraphs[subID]->size());
#endif
  // TODO clean out?
}


//void DistContext::allocSubNormFactor(int subID) {
//  if (!normFactors) {
//#ifdef USE_MKL
//    normFactors = new float_t[partitionedGraph->sizeEdges()];
//#else
//    normFactors = new float_t[partitionedGraph->size()];
//#endif
//  }
//  if (!normFactors) {
//    GALOIS_DIE("norm factors failed to be allocated");
//  }
//}

void DistContext::constructNormFactor(deepgalois::Context* globalContext) {
  // TODO IMPLEMENT THIS; get relevant info from the original context
  // sets current subgraph + gets degrees
  Graph* wholeGraph = globalContext->getCurrentGraph(false);

  allocNormFactor();

  // this is for testing purposes
  //galois::do_all(galois::iterate((size_t)0, partitionedGraph->size()),
  //  [&] (unsigned i) {
  //    this->normFactors[i] = 0;
  //  }
  //);

#ifdef USE_MKL
  galois::do_all(galois::iterate((size_t)0, partitionedGraph->size()),
    [&] (unsigned i) {
      float_t c_i = std::sqrt(float_t(wholeGraph->get_degree(partitionedGraph->getGID(i))));

      for (auto e = partitionedGraph->edge_begin(i); e != partitionedGraph->edge_end(i); e++) {
        const auto j = partitionedGraph->getEdgeDst(e);
        float_t c_j  = std::sqrt(float_t(wholeGraph->get_degree(partitionedGraph->getGID(j))));

        if (c_i == 0.0 || c_j == 0.0) {
          this->normFactors[e] = 0.0;
        } else {
          this->normFactors[e] = 1.0 / (c_i * c_j);
        }
    },
    galois::loopname("NormCountingEdge"));
  );
#else
  galois::do_all(galois::iterate((size_t)0, partitionedGraph->size()),
    [&] (unsigned v) {
      auto degree = wholeGraph->get_degree(partitionedGraph->getGID(v));
      float_t temp = std::sqrt(float_t(degree));
      if (temp == 0.0) {
        this->normFactors[v] = 0.0;
      } else {
        this->normFactors[v] = 1.0 / temp;
      }
    },
    galois::loopname("NormCountingNode"));
#endif
}

void DistContext::constructNormFactorSub(int subgraphID) {
  // right now norm factor based on subgraph
  // TODO fix this for dist execution

  allocNormFactorSub(subgraphID);

  Graph& graphToUse = *partitionedSubgraphs[subgraphID];
  graphToUse.degree_counting();

  // TODO using partitioned subgraph rather than whoel graph; i.e. dist setting wrong
#ifdef USE_MKL
  galois::do_all(galois::iterate((size_t)0, graphToUse->size()),
    [&] (unsigned i) {
      //float_t c_i = std::sqrt(float_t(wholeGraph->get_degree(partitionedGraph->getGID(i))));
      float_t c_i = std::sqrt(float_t(graphToUse.get_degree(i)));

      for (auto e = graphToUse->edge_begin(i); e != graphToUse->edge_end(i); e++) {
        const auto j = graphToUse->getEdgeDst(e);
        float_t c_j  = std::sqrt(float_t(graphToUse.get_degree(j)));

        if (c_i == 0.0 || c_j == 0.0) {
          this->normFactorsSub[e] = 0.0;
        } else {
          this->normFactorsSub[e] = 1.0 / (c_i * c_j);
        }
    },
    galois::loopname("NormCountingEdge"));
  );
#else
  galois::do_all(galois::iterate((size_t)0, graphToUse.size()),
    [&] (unsigned v) {
      //auto degree = wholeGraph->get_degree(partitionedGraph->getGID(v));
      auto degree = graphToUse.get_degree(v);
      float_t temp = std::sqrt(float_t(degree));
      if (temp == 0.0) {
        this->normFactorsSub[v] = 0.0;
      } else {
        this->normFactorsSub[v] = 1.0 / temp;
      }
      galois::gPrint(this->normFactorsSub[v], "\n");
    },
    galois::loopname("NormCountingNode"));
#endif
}
//! generate labels for the subgraph, m is subgraph size, mask
//! tells which vertices to use
void DistContext::constructSubgraphLabels(size_t m, const mask_t* masks) {
  // TODO multiclass

  // if (h_labels_subg == NULL) h_labels_subg = new label_t[m];
  //if (DistContext::is_single_class) {
  //} else {
  //  DistContext::h_labels_subg.resize(m * Context::num_classes);
  //}

  DistContext::h_labels_subg.resize(m);

  size_t count = 0;
  // see which labels to copy over for this subgraph
  for (size_t i = 0; i < this->partitionedGraph->size(); i++) {
    if (masks[i] == 1) {
      //if (Context::is_single_class) {
      //} else {
      //  std::copy(Context::h_labels + i * Context::num_classes,
      //            Context::h_labels + (i + 1) * Context::num_classes,
      //            &Context::h_labels_subg[count * Context::num_classes]);
      //}
      DistContext::h_labels_subg[count] = h_labels[i];
      //galois::gPrint("l ", (float)DistContext::h_labels_subg[count], "\n");
      count++;
    }
  }
  GALOIS_ASSERT(count == m);
}

//! generate input features for the subgraph, m is subgraph size,
//! masks tells which vertices to use
void DistContext::constructSubgraphFeatures(size_t m, const mask_t* masks) {
  size_t count = 0;
  // if (h_feats_subg == NULL) h_feats_subg = new float_t[m*feat_len];
  DistContext::h_feats_subg.resize(m * feat_len);
  for (size_t i = 0; i < this->partitionedGraph->size(); i++) {
    if (masks[i] == 1) {
      std::copy(DistContext::h_feats + i * DistContext::feat_len,
                DistContext::h_feats + (i + 1) * DistContext::feat_len,
                &DistContext::h_feats_subg[count * DistContext::feat_len]);
      //for (unsigned a = 0; a < DistContext::feat_len; a++) {
      //  if (h_feats_subg[count * DistContext::feat_len + a] != 0) {
      //    galois::gPrint(h_feats_subg[count * DistContext::feat_len + a], " ");
      //  }
      //}
      //galois::gPrint("\n");
      count++;
    }
  }
  GALOIS_ASSERT(count == m);
}


galois::graphs::GluonSubstrate<DGraph>* DistContext::getSyncSubstrate() {
  return DistContext::syncSubstrate;
};

void DistContext::allocateSubgraphs(int num_subgraphs) {
  partitionedSubgraphs.resize(num_subgraphs);
  for (int i = 0; i < num_subgraphs; i++) {
    partitionedSubgraphs[i] = new Graph();
  }
}

} // namespace deepgalois
