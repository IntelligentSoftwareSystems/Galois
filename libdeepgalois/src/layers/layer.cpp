#include "deepgalois/layers/layer.h"
#include "galois/Galois.h"

namespace deepgalois {

void layer::print_layer_info() {
  galois::gPrint("Layer", level_, " type: ", layer_type(), " input[",
                 input_dims[0], ",", input_dims[1], "] output[",
                 output_dims[0], ",", output_dims[1], "]\n");
}

}
