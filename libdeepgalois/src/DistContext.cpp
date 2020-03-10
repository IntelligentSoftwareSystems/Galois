#include "deepgalois/DistContext.h"

namespace deepgalois {
DistContext::DistContext() {}
DistContext::~DistContext() {}

void DistContext::saveGraph(Graph* dGraph) {
  graph_cpu = dGraph;
}
size_t DistContext::read_labels(std::string dataset_str) {
  Graph* dGraph = DistContext::graph_cpu;
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
  galois::gPrint("[", myID, "] Reading labels...\n");

  //std::string filename = path + dataset_str + "-labels.txt";
  //std::ifstream in;
  //std::string line;
  //in.open(filename, std::ios::in);
  //size_t m;
  //// read file header
  //an >> m >> num_classes >> std::ws;
  //assert(m == dGraph->globalSize());
  //// size of labels is only # local nodes
  //labels.resize(dGraph.size(), 0);

  //unsigned v = 0;
  //while (std::getline(in, line)) {
  //  std::istringstream label_stream(line);
  //  unsigned x;
  //  for (size_t idx = 0; idx < num_classes; ++idx) {
  //    label_stream >> x;
  //    if (x != 0) {
  //      labels[v] = idx;
  //      break;
  //    }
  //  }
  //  v++;
  //}
  //in.close();

  //// print the number of vertex classes
  //std::cout << "Done, unique label counts: " << num_classes
  //          << ", time: " << t_read.Millisecs() << " ms\n";
  //return num_classes;
}

size_t DistContext::read_features(std::string dataset_str) {
  // TODO
  return 0;
}

float_t* DistContext::get_in_ptr() {
  // TODO
  return nullptr;
}

void DistContext::norm_factor_counting() {
  // TODO
}

}  // deepgalois
