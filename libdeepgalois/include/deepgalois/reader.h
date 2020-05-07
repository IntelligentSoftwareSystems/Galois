#pragma once
#include "deepgalois/GraphTypes.h"

namespace deepgalois {

class Reader {
private:
  std::string dataset_str;
  void progressPrint(unsigned maxii, unsigned ii);

public:
  Reader() : dataset_str("") {}
  Reader(std::string dataset) : dataset_str(dataset) {}
  void init(std::string dataset) { dataset_str = dataset; }
  size_t read_labels(bool is_single_class, label_t*& labels);
  size_t read_features(float_t*& feats, std::string filetype = "bin");
  size_t read_masks(std::string mask_type, size_t n, size_t& begin, size_t& end,
                    mask_t* masks);
  void readGraphFromGRFile(Graph* g);
};

} // namespace deepgalois
