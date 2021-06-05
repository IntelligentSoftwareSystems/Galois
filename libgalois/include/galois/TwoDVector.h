#pragma once

#include "gstl.h" 
#include "PODResizeableArray.h" 

namespace galois {

template <typename T>
class TwoDVector {
public:
  using value_type = T;

  void SetVecSize(size_t fixed_vector_size) {
    fixed_vector_size_ = fixed_vector_size;
  }

  //! Call this before using this else bad things will happen: initializes
  //! the memory + fixed size metadata
  void Create(size_t num_elements) {
    num_elements_ = num_elements;
    underlying_memory_.resize(num_elements_ * fixed_vector_size_);
  }
  void SetVector(size_t index, const galois::gstl::Vector<T>& to_copy) {
    // TODO(loc) for generality should work with any vector type, but for
    // now just use gstl
    assert(index < num_elements_);
    assert(to_copy == fixed_vector_size_);
    size_t array_index = index * fixed_vector_size_;
    std::memcpy((void*)(&(underlying_memory_[array_index])),
                (void*)to_copy.data(),
                sizeof(T) * fixed_vector_size_);
  }

  PODResizeableArray<T>& edit_data() { return underlying_memory_; }
  const PODResizeableArray<T>& data() { return underlying_memory_; }
  void resize(size_t s) { underlying_memory_.resize(s); }
  size_t size() const { return underlying_memory_.size(); }
private:
  size_t num_elements_{0};
  size_t fixed_vector_size_{0};
  PODResizeableArray<T> underlying_memory_;
};

}
