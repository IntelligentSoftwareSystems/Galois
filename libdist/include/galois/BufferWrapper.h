#ifndef GALOIS_BUFFER_WRAPPER
#define GALOIS_BUFFER_WRAPPER
#include "galois/gstl.h"
#include <cassert>

namespace galois {

//! Wraps a pointer representing an array with the number of elements the
//! array contains (or that we want to handle with this class)
//!
//! Used to avoid copying of memory into a vector for
//! serialization/deserialization purpose
//! @todo give this a better name
template <typename ElementType>
class BufferWrapper {
public:
  using size_type  = size_t;
  using value_type = ElementType;

private:
  //! This vector is allocated when creating a buffer wrapper from scratch
  //! (i.e. during deserialization into one)
  galois::gstl::Vector<ElementType> dummy;
  //! Raw memory kept by this class; either points to existing memory or to the
  //! vector memory held by this class
  ElementType* raw_memory;
  //! Number of elements that can be accessed from the raw_memory pointer
  size_type num_elements;

public:
  //! Default constructor 0s everything
  BufferWrapper() {
    this->raw_memory   = 0;
    this->num_elements = 0;
  }

  //! frees dummy vector
  ~BufferWrapper() {
    // explicit vector clear; regular destructor probably frees it, but
    // doing it for safetey
    if (dummy.size()) {
      dummy.clear();
    }
  }

  //! Save a pointer and the number of elements in that array that this can
  //! access
  BufferWrapper(ElementType* pointer, size_t num_elements_)
      : raw_memory(pointer), num_elements(num_elements_){};

  //! Returns element at some specified index of the array
  ElementType& operator[](size_t index) {
    assert(index < num_elements);
    return raw_memory[index];
  }

  //! Returns element at some specified index of the array; const i.e. not
  //! modifiable
  const ElementType& operator[](size_t index) const {
    assert(index < num_elements);
    return raw_memory[index];
  }

  //! Return number of elements in the array
  size_t size() const { return this->num_elements; }

  //! return unmodifiable pointer to raw_memory
  const ElementType* data() const { return raw_memory; }
  //! return pointer to raw_memory
  ElementType* data() { return raw_memory; }

  //! Allocates memory in the underlying vector; should only be used for
  //! deserialization into this class during communication
  void resize(size_t new_size) {
    if (!this->raw_memory) {
      this->dummy.resize(new_size);
      this->raw_memory   = this->dummy.data();
      this->num_elements = this->dummy.size();
    } else {
      GALOIS_DIE("calling resize when there is already raw memory "
                 "allocated");
    }
  }
};

} // namespace galois
#endif
