#ifndef _SVECTOR_WRAPPER_H_
#define _SVECTOR_WRAPPER_H_

#include <vector>
#include "galois/runtime/Extra_dist_traits.h"

namespace galois {

/**
 * Wrapper around a vector that is meant to be serializable. As such, the
 * vector needs to have a fixed size, and it cannot change in size unless
 * resize is explicitly called. 
 *
 * @tparam T type that the wrapped vector will have elements of
 */
template<typename T>
class SerializableVector {
 public:
  using VecType = std::vector<T>;
  using iterator = typename VecType::iterator;
  using const_iterator = typename VecType::const_iterator;

 private:
  VecType wrappedVector;
  size_t vectorSize;

 public:
  /**
   * Initialize to empty vector.
   */
  SerializableVector() : vectorSize(0) { }

  /**
   * @returns iterator to beginning of vector
   */
  iterator begin() {
    return wrappedVector.begin();
  }

  /**
   * @returns constant iterator to beginning of vector
   */
  const_iterator begin() const {
    return wrappedVector.begin();
  }

  /**
   * @returns constant iterator to beginning of vector
   */
  const_iterator cbegin() const {
    return wrappedVector.cbegin();
  }

  /**
   * @returns iterator to end of vector
   */
  iterator end() {
    return wrappedVector.end();
  }
 
  /**
   * @returns constant iterator to end of vector
   */
  const_iterator end() const {
    return wrappedVector.end();
  }

  /**
   * @returns constant iterator to end of vector
   */
  const_iterator cend() const {
    return wrappedVector.cend();
  }

  /**
   * @returns Current size of the vector
   */
  size_t size() const {
    return vectorSize;
  }

  /**
   * Resizes wrapped vector
   *
   * @param newSize size to resize vector to
   */
  void resize(size_t newSize) {
    vectorSize = newSize;
    wrappedVector.resize(newSize);
  }

  /**
   * @param n Element to get
   * @returns reference to a particular element of wrapped vector.
   */
  T& operator[](size_t n) {
    assert(n < vectorSize);
    return wrappedVector[n];
  }

  /**
   * @param n Element to get
   * @returns copy of a particular element of wrapped vector.
   */
  T operator[](size_t n) const {
    assert(n < vectorSize);
    return wrappedVector[n];
  }

  /**
   * @returns reference to wrapped vector
   */
  VecType& getVector() {
    return wrappedVector;
  }

  /**
   * @returns constant reference to wrapped vector
   */
  const VecType& getVector() const {
    return wrappedVector;
  }

  // only make copyable if the inner type is copyable
  //using tt_is_copyable = 
  //  typename std::enable_if<galois::runtime::is_memory_copyable<T>::value, 
  //                                                              int>::type;
};

} // end galois namespace


// below goes in Serialize.h if you wanted to use the above
///**
// * Specialization for SerializableVector. Serializes the size (so other end 
// * knows how much to deserialize) and the data itself.
// *
// * @param buf Buffer to serialize data into
// * @param data Vector that needs to be serialized
// */
//template<typename T>
//inline void gSerializeObj(SerializeBuffer& buf, 
//                          const galois::SerializableVector<T>& data) {
//  gSerializeObj(buf, data.size());
//  gSerializeObj(buf, data.getVector());
//}
//
///**
// * Specialization for SerializableVector. Gets the size of the vector, then
// * loads that many elements from the buffer into the vector.
// */
//template <typename T>
//inline void gDeserializeObj(DeSerializeBuffer& buf, 
//                            galois::SerializableVector<T>& data) {
//  size_t size = 0;
//  gDeserializeObj(buf, size);
//  data.resize(size);
//  gDeserializeObj(buf, data.getVector());
//}

#endif
