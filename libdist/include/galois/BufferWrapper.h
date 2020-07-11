#ifndef GALOIS_BUFFER_WRAPPER
#define GALOIS_BUFFER_WRAPPER
#include <cassert>

namespace galois {

//! Wraps a pointer representing an array with the number of elements the
//! array contains (or that we want to handle with this class)
//! Used to avoid copying of memory into a vector for
//! serialization/deserialization purpose
//! @todo give this a better name
template<typename ElementType>
class BufferWrapper {
  //! Raw memory kept by this class
  ElementType* raw_memory;
  //! Number of elements that can be accessed from the raw_memory pointer
  size_t num_elements;
public:
  //! Default constructor doesn't exist: must provide pointer and size
  BufferWrapper() = delete;
  //! Save a pointer and the number of elements in that array that this can access
  BufferWrapper(ElementType* pointer, size_t num_elements_) : raw_memory(pointer),
  num_elements(num_elements_) {};

  //! Returns element at some specified index of the array
  ElementType& operator[](size_t index) {
    assert(index < num_elements);
    return raw_memory[index];
  }

  //! Returns element at some specified index of the array; const i.e. not modifiable
  const ElementType& operator[](size_t index) const {
    assert(index < num_elements);
    return raw_memory[index];
  }

  //! Return number of elements in the array
  size_t size() const {
    return this->num_elements;
  }
};

} // end namespace
#endif
