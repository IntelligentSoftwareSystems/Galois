#pragma once
//! @file GNNTypes.h
//! Typedefs used by the Galois GNN code

#include <cstdint>
#include <cstddef>
#include <vector>

#ifdef GALOIS_ENABLE_GPU
enum class DevicePersonality { CPU, GPU_CUDA };
extern DevicePersonality device_personality;
#endif

namespace galois {
//! Floating point type to use throughout GNN compute; typedef'd so it's easier
//! to flip later
using GNNFloat = float;
//! Type of the labels for a vertex
using GNNLabel = uint8_t;
//! Type of a feature on vertices
using GNNFeature = float;
//! Type of node index on gpus
using GPUNodeIndex = uint32_t;
//! Type of edge index on gpus
using GPUEdgeIndex = uint64_t;

//! Phase of GNN computation
enum class GNNPhase { kTrain, kValidate, kTest };

//! Vector like wrapper over a pointer and size; exists solely to pass around
//! raw pointers with size (because vectors are a no-go due to the code
//! handling both CPU and GPU.)
template <typename PointerType>
class PointerWithSize {
public:
  //! Default is empty
  PointerWithSize() : ptr_{nullptr}, num_elements_{0} {}
  //! Generic constructor which takes 2 fields to initialize
  PointerWithSize(PointerType* ptr, size_t num_elements)
      : ptr_{ptr}, num_elements_{num_elements} {}
  //! Grab vector pointer + size
  PointerWithSize(std::vector<PointerType>& v)
      : ptr_{v.data()}, num_elements_{v.size()} {}
  //! Alias to return pointer data
  PointerType* data() { return ptr_; }
  //! Alias to return pointer data (const version)
  const PointerType* data() const { return ptr_; }
  //! # elements that pointer should contain
  size_t size() const { return num_elements_; }
  // accessors; one lets you mess with the array
  PointerType& operator[](size_t i) { return ptr_[i]; }
  const PointerType& operator[](size_t i) const { return ptr_[i]; }

private:
  //! Pointer to data
  PointerType* ptr_;
  //! # elements that I should be able to access from pointer
  size_t num_elements_;
};

} // end namespace galois
