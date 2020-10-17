#include "galois/Logging.h"
#include "galois/GNNMath.h" // Please add GPU functions
#include "galois/layers/SoftmaxLayer.h"

// Allocate memory and initialize
void galois::SoftmaxLayer::Init() {
}

// Input: in_tensor
// Output: out_tensor
void galois::SoftmaxLayer::Forward(const galois::GNNFloat* in_tensor,
                                   galois::GNNFloat* out_tensor) {
} 

// Input: in_tensor
// Input: out_tensor
// Input: out_gradients
// Output: in_gradients
void galois::SoftmaxLayer::Backward(const galois::GNNFloat* in_tensor,
                                    const galois::GNNFloat* out_tensor,
                                    galois::GNNFloat* in_gradients,
                                    galois::GNNFloat* out_gradients) {
}

