#pragma once
#include "galois/GNNTypes.h"

namespace galois {

//! Holds pointers for GPU memory for GCN layer
class GCNGPUAllocations {
public:
  // free memory
  ~GCNGPUAllocations();
  // allocate the 3 temp arrays
  void Allocate(size_t input_elements, size_t output_elements);
private:
  GNNFloat* in_temp_1_{nullptr};
  GNNFloat* in_temp_2_{nullptr};
  GNNFloat* out_temp_{nullptr};
};

}
