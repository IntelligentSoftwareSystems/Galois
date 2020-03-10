#include "deepgalois/DistContext.h"

namespace deepgalois {
DistContext::DistContext() {}
DistContext::~DistContext() {}

size_t DistContext::saveGraph(Graph* dGraph) {
  // TODO
  return 0;
}
size_t DistContext::read_labels(std::string dataset_str) {
  // TODO
  return 0;
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
