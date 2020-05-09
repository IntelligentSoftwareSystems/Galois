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

//void DistContext::constructNormFactorSub(deepgalois::Context* globalContext, bool isSubgraph,
//                         int subgraphID) {

galois::graphs::GluonSubstrate<DGraph>* DistContext::getSyncSubstrate() {
  return DistContext::syncSubstrate;
};

} // namespace deepgalois
