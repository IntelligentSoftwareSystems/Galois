/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

/**
 * @file Serialize.h
 *
 * Contains functions that serialize/deserialize data, mainly for sending
 * out serialized data over the network and deserializing it on the other end.
 */

#ifndef GALOIS_RUNTIME_SERIALIZE_H
#define GALOIS_RUNTIME_SERIALIZE_H

#include <type_traits>
#include <ostream>
#include <vector>
#include <deque>
#include <string>
#include <cassert>
#include <tuple>

#include <boost/mpl/has_xxx.hpp>
#include "galois/runtime/Extra_dist_traits.h"

#include <galois/gdeque.h>
#include <galois/DynamicBitset.h>
#include <galois/AtomicWrapper.h>
#include <galois/PODResizeableArray.h>
#include "galois/CopyableTuple.h"
#include "galois/Bag.h"

namespace galois {
namespace runtime {

class DeSerializeBuffer; // forward declaration for friend declaration

/**
 * Buffer for serialization of data. Mainly used during network communication.
 */
class SerializeBuffer {
  //! Access to a deserialize buffer
  friend DeSerializeBuffer;

  //! type of data buffer
  //using vTy = std::vector<uint8_t>;
  using vTy = galois::PODResizeableArray<uint8_t>;
  //! the actual data stored in this buffer
  vTy bufdata;

public:
  //! default constructor
  SerializeBuffer() = default;
  //! disabled copy constructor
  SerializeBuffer(SerializeBuffer&& rhs) = default;
  //! Creates a buffer from another buffer
  //! @param d buffer to create from
  //! @param len amount of copy from buffer d
  SerializeBuffer(const char* d, unsigned len) : bufdata(d, d + len) {}

  //! Push a character onto the serialize buffer
  inline void push(const char c) { bufdata.push_back(c); }

  //! Insert characters from a buffer into the serialize buffer
  void insert(const uint8_t* c, size_t bytes) {
    bufdata.insert(bufdata.end(), c, c + bytes);
  }

  //! Insert characters from a buffer into the serialize buffer at a particular
  //! offset
  void insertAt(const uint8_t* c, size_t bytes, size_t offset) {
    std::copy_n(c, bytes, bufdata.begin() + offset);
  }

  /**
   * Reserve space at the end for inserting new data into the serialize
   * buffer
   *
   * @param bytes number of bytes to reserve at the end
   * @returns offset to the end of the buffer before new space was reserved
   */
  size_t encomber(size_t bytes) {
    size_t retval = bufdata.size();
    bufdata.resize(retval + bytes);
    return retval;
  }

  void resize(size_t bytes) {
    bufdata.resize(bytes);
  }

  /**
   * Reserve more space in the serialize buffer.
   *
   * @param s extra space to reserve
   */
  void reserve(size_t s) { bufdata.reserve(bufdata.size() + s); }

  //! Returns a pointer to the data stored in this serialize buffer
  const uint8_t* linearData() const { return bufdata.data(); }
  //! Returns vector of data stored in this serialize buffer
  vTy& getVec() { return bufdata; }

  //! Returns an iterator to the beginning of the data in this serialize buffer
  vTy::const_iterator begin() const { return bufdata.cbegin(); }
  //! Returns an iterator to the end of the data in this serialize buffer
  vTy::const_iterator end() const { return bufdata.cend(); }

  using size_type = vTy::size_type;

  //! Returns the size of the serialize buffer
  size_type size() const { return bufdata.size(); }

  //! Utility print function for the serialize buffer
  //! @param o stream to print to
  void print(std::ostream& o) const {
    o << "<{" << std::hex;
    for (auto& i : bufdata)
      o << (unsigned int)i << " ";
    o << std::dec << "}>";
  }

  //! Operator that calls the print function of the serialize buffer
  friend std::ostream& operator<<(std::ostream& os, const SerializeBuffer& b) {
    b.print(os);
    return os;
  }
};

/**
 * Buffer for deserialization of data. Mainly used during network
 * communication.
 */
class DeSerializeBuffer {
  //! Access to serialize buffer
  friend SerializeBuffer;
  //! type of data buffer
  //using vTy = std::vector<uint8_t>;
  using vTy = galois::PODResizeableArray<uint8_t>;
  //! the actual data stored in this buffer
  vTy bufdata;
  int offset;

public:
  //! Constructor initializes offset into buffer to 0
  DeSerializeBuffer() : offset(0) {}
  //! Disable copy constructor
  DeSerializeBuffer(DeSerializeBuffer&&) = default;
  //! Move constructor
  //! @param v vector to act as deserialize buffer
  //! @param start offset to start saving data into
  DeSerializeBuffer(vTy&& v, uint32_t start = 0)
      : bufdata(std::move(v)), offset(start) {}

  //! Constructor that takes an existing vector to use as the deserialize
  //! buffer
  explicit DeSerializeBuffer(vTy& data) {
    bufdata.swap(data);
    offset = 0;
  }

  /**
   * Initializes the deserialize buffer with a certain size
   * @param [in] count size to initialize buffer to
   */
  explicit DeSerializeBuffer(int count) : bufdata(count), offset(0) {}

  /**
   * Initializes the deserialize buffer using vector initialization from
   * 2 iterators.
   */
  template <typename Iter>
  DeSerializeBuffer(Iter b, Iter e) : bufdata(b, e), offset{0} {}

  /**
   * Initialize a deserialize buffer from a serialize buffer
   */
  explicit DeSerializeBuffer(SerializeBuffer&& buf) : offset(0) {
    bufdata.swap(buf.bufdata);
  }

  /**
   * Disable copy constructor
   */
  DeSerializeBuffer& operator=(DeSerializeBuffer&& buf) = default;

  /**
   * Reset deserialize buffer
   * @param count new size of buffer
   */
  void reset(int count) {
    offset = 0;
    bufdata.resize(count);
  }

  //! Gets the current offset into the deserialize buffer
  unsigned getOffset() const { return offset; }
  //! Sets the offset into the deserialize buffer
  void setOffset(unsigned off) {
    assert(off <= size());
    offset = off;
  }

  //! Gets the size of the deserialize buffer
  unsigned size() const { return bufdata.size(); }

  //! Returns true if the deserialize buffer is empty
  //! @returns true if the deserialize buffer is empty
  bool empty() const { return bufdata.empty(); }

  //! Get the next character in the deserialize buffer
  unsigned char pop() { return bufdata.at(offset++); }

  //! Clears the last x bytes of the deserialize buffer, resizing it as well
  //! @param x How many bytes from the end to clear
  void pop_back(unsigned x) { bufdata.resize(bufdata.size() - x); }

  /**
   * Extracts a certain amount of data from the deserialize buffer
   *
   * @param dst buffer to copy data from deserialize buffer into
   * @param num Amount of data to get from deserialize buffer
   */
  void extract(uint8_t* dst, size_t num) {
    memcpy(dst, &bufdata[offset], num);
    offset += num;
  }

  //! Get the underlying vector storing the data of the deserialize
  //! buffer
  vTy& getVec() { return bufdata; }

  //! Get a pointer to the underlying data of the deserialize buffer
  void* linearData() { return &bufdata[0]; }

  //! Get a pointer to the remaining data of the deserialize buffer
  //! (as determined by offset)
  const uint8_t* r_linearData() const { return &bufdata[offset]; }
  //! Get the remaining size of the deserialize buffer (as determined
  //! by offset)
  size_t r_size() const { return bufdata.size() - offset; }

  //! Checks if the current location in the deserialize buffer is aligned
  //! to some size a
  bool atAlignment(size_t a) { return (uintptr_t)r_linearData() % a == 0; }

  //! Utility print of deserialize buffer
  //! @param o stream to print to
  void print(std::ostream& o) const {
    o << "<{(" << offset << ") " << std::hex;
    for (auto ii = bufdata.begin(), ee = bufdata.end(); ii != ee; ++ii)
      o << (unsigned int)*ii << " ";
    o << std::dec << "}>";
  }

  //! Operator for printing deserialize buffer
  friend std::ostream& operator<<(std::ostream& os,
                                  const DeSerializeBuffer& buf) {
    buf.print(os);
    return os;
  }
};

namespace internal {

/**
 * Returns the size necessary for an object in a buffer.
 * This version runs if the data is memory copyable; uses sizeof.
 *
 * @tparam T type of datato get size of
 */
template <typename T>
__attribute__((always_inline)) constexpr size_t
gSizedObj(const T& data,
          typename std::enable_if<is_memory_copyable<T>::value>::type* = 0) {
  return sizeof(T);
}

/**
 * Returns the size necessary for an object in a buffer.
 * This version runs if the data is not memory copyable but is serializable.
 * It returns the size of a uintptr_t.
 *
 * @tparam T type of datato get size of
 * @returns size of uintptr_t
 */
template <typename T>
__attribute__((always_inline)) constexpr size_t
gSizedObj(const T& data,
          typename std::enable_if<!is_memory_copyable<T>::value>::type* = 0,
          typename std::enable_if<has_serialize<T>::value>::type*       = 0) {
  return sizeof(uintptr_t);
}

/**
 * Returns the size necessary for storing 2 elements of a pair into a
 * serialize buffer.
 *
 * @param data pair of 2 elements
 */
template <typename T1, typename T2>
inline size_t gSizedObj(const std::pair<T1, T2>& data) {
  return gSizedObj(data.first) + gSizedObj(data.second);
}

/**
 * Returns the size necessary to store a sequence in a serialize buffer.
 * This depends on if the sequence is memory copyable.
 */
template <typename Seq>
size_t gSizedSeq(const Seq& seq) {
  typename Seq::size_type size = seq.size();
  typedef typename Seq::value_type T;
  size_t tsize = std::conditional<
      is_memory_copyable<T>::value, std::integral_constant<size_t, sizeof(T)>,
      std::integral_constant<size_t, sizeof(uintptr_t)>>::type::value;
  return sizeof(size) + tsize * size;
}

/**
 * Returns the size needed to store the elements a vector in a serialize
 * buffer.
 *
 * @returns size needed to store a vector into a serialize buffer
 */
template <typename T, typename Alloc>
inline size_t gSizedObj(const std::vector<T, Alloc>& data) {
  return gSizedSeq(data);
}

/**
 * Returns the size needed to store the elements a PODResizeableArray in a serialize
 * buffer.
 *
 * @returns size needed to store a PODResizeableArray into a serialize buffer
 */
template <typename T>
inline size_t gSizedObj(const galois::PODResizeableArray<T>& data) {
  return gSizedSeq(data);
}

/**
 * Returns the size needed to store the elements a deque into a serialize
 * buffer.
 *
 * @returns size needed to store a deque into a serialize buffer
 */
template <typename T, typename Alloc>
inline size_t gSerializeObj(const std::deque<T, Alloc>& data) {
  return gSizedSeq(data);
}

/**
 * Returns the size needed to store the elements a Galois deque into a serialize
 * buffer.
 *
 * @returns size needed to store a Galois deque into a serialize buffer
 */
template <typename T, unsigned CS>
inline size_t gSizedObj(const galois::gdeque<T, CS>& data) {
  return gSizedSeq(data);
}

/**
 * Returns the size needed to store a string into a serialize
 * buffer.
 *
 * @returns size needed to store a string into a serialize buffer
 */
template <typename A>
inline size_t
gSizedObj(const std::basic_string<char, std::char_traits<char>, A>& data) {
  return data.length() + 1;
}

/**
 * Returns the size of the passed in serialize buffer
 *
 * @returns size of the serialize buffer passed into it
 */
inline size_t gSizedObj(const SerializeBuffer& data) { return data.size(); }

/**
 * Returns the size of the passed in deserialize buffer
 *
 * @returns size of the deserialize buffer passed into it
 */
inline size_t gSizedObj(const DeSerializeBuffer& rbuf) { return rbuf.r_size(); }

/**
 * Returns the size of the passed in insert bag.
 *
 * @returns size of the insert bag passed into it
 */
template <typename T>
inline size_t gSizedObj(const galois::InsertBag<T>& bag) {
  return bag.size();
}

/**
 * Returns 0.
 * @returns 0
 */
inline size_t adder() { return 0; }
/**
 * Returns the passed in argument.
 * @param a a number
 * @returns a
 */
inline size_t adder(size_t a) { return a; }
/**
 * Returns the sum of all passed in arguments.
 * @returns sum of all arguments
 */
template <typename... Args>
inline size_t adder(size_t a, size_t b, Args&&... args) {
  return a + b + adder(args...);
}

} // namespace internal

/**
 * Gets the total size necessary for storing all of the passed in arguments into
 * a serialize buffer.
 *
 * @returns size necessary for storing all arguments into a serialize buffer
 */
template <typename... Args>
static inline size_t gSized(Args&&... args) {
  return internal::adder(internal::gSizedObj(args)...);
}

////////////////////////////////////////////////////////////////////////////////
// Serialize support
////////////////////////////////////////////////////////////////////////////////

namespace internal {

/**
 * Serialize a memory copyable object into a serialize buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data Data to serialize
 */
template <typename T>
inline void gSerializeObj(
    SerializeBuffer& buf, const T& data,
    typename std::enable_if<is_memory_copyable<T>::value>::type* = 0) {
  uint8_t* pdata = (uint8_t*)&data;
  buf.insert(pdata, sizeof(T));
}

/**
 * Serialize a non-memory copyable but serializable object into a serialize
 * buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data Data to serialize
 */
template <typename T>
inline void
gSerializeObj(SerializeBuffer& buf, const T& data,
              typename std::enable_if<!is_memory_copyable<T>::value>::type* = 0,
              typename std::enable_if<has_serialize<T>::value>::type* = 0) {
  data.serialize(buf);
}

/**
 * Serialize a pair into a serialize buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data Pair to serialize
 */
template <typename T1, typename T2>
inline void gSerializeObj(SerializeBuffer& buf, const std::pair<T1, T2>& data) {
  gSerialize(buf, data.first, data.second);
}

/**
 * Serialize a tuple of 3. Either memcpys entire struct or serializes
 * each element individually.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data Tuple of 3 to serialize
 * @todo This specialization isn't being used as expected. Figure out why.
 */
template <typename T1, typename T2, typename T3>
inline void gSerializeObj(SerializeBuffer& buf,
                          const galois::TupleOfThree<T1, T2, T3>& data) {
  if (is_memory_copyable<T1>::value && is_memory_copyable<T2>::value &&
      is_memory_copyable<T3>::value) {
    // do memcpy
    buf.insert((uint8_t*)&data, sizeof(data));
  } else {
    // serialize each individually
    gSerialize(buf, data.first, data.second, data.third);
  }
}

/**
 * Serialize a copyable atomic: load atomic data as a plain old
 * datatype (POD) and mem copy it to the buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data copyable atomic to serialize
 */
template <typename T>
inline void gSerializeObj(SerializeBuffer& buf,
                          const galois::CopyableAtomic<T>& data) {
  T temp = data.load();
  buf.insert((uint8_t*)(&temp), sizeof(T));
}

/**
 * Serialize a string into a buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data String
 */
template <typename A>
inline void
gSerializeObj(SerializeBuffer& buf,
              const std::basic_string<char, std::char_traits<char>, A>& data) {
  buf.insert((uint8_t*)data.data(), data.length() + 1);
}

// Forward declaration of vector serialize
template <typename T, typename Alloc>
inline void gSerializeObj(SerializeBuffer& buf,
                          const std::vector<T, Alloc>& data);

/**
 * Serialize a sequence type into a buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] seq sequence to serialize
 * @todo specialize for Sequences with consecutive PODS
 */
template <typename Seq>
void gSerializeSeq(SerializeBuffer& buf, const Seq& seq) {
  typename Seq::size_type size = seq.size();
  gSerializeObj(buf, size);
  for (auto& o : seq)
    gSerializeObj(buf, o);
}

/**
 * Serialize a linear sequence type (i.e. memcopyable) into a buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] seq sequence to serialize
 */
template <typename Seq>
void gSerializeLinearSeq(SerializeBuffer& buf, const Seq& seq) {
  typename Seq::size_type size = seq.size();
  typedef typename Seq::value_type T;
  size_t tsize = sizeof(T);
  //  buf.reserve(size * tsize + sizeof(size));
  gSerializeObj(buf, size);
  buf.insert((uint8_t*)seq.data(), size * tsize);
}

/**
 * Serialize a vector into a buffer, choosing to do a memcopy or
 * to serialize each element individually depending on data.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data vector to serialize
 */
template <typename T, typename Alloc>
inline void gSerializeObj(SerializeBuffer& buf,
                          const std::vector<T, Alloc>& data) {
  if (is_memory_copyable<T>::value)
    gSerializeLinearSeq(buf, data);
  else
    gSerializeSeq(buf, data);
}

/**
 * Serialize a PODResizeableArray into a buffer, choosing to do a memcopy or
 * to serialize each element individually depending on data.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data PODResizeableArray to serialize
 */
template <typename T>
inline void gSerializeObj(SerializeBuffer& buf,
                          const galois::PODResizeableArray<T>& data) {
  gSerializeLinearSeq(buf, data);
}

/**
 * Serialize a deque into a buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data deque to serialize
 */
template <typename T, typename Alloc>
inline void gSerializeObj(SerializeBuffer& buf,
                          const std::deque<T, Alloc>& data) {
  gSerializeSeq(buf, data);
}

/**
 * Serialize a Galois deque into a buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data deque to serialize
 */
template <typename T, unsigned CS>
inline void gSerializeObj(SerializeBuffer& buf,
                          const galois::gdeque<T, CS>& data) {
  gSerializeSeq(buf, data);
}

/**
 * Serialize data in another serialize buffer into a buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data serialize buffer to get data from
 */
inline void gSerializeObj(SerializeBuffer& buf, const SerializeBuffer& data) {
  buf.insert(data.linearData(), data.size());
}

/**
 * Serialize data in a deserialize buffer into a buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] rbuf deserialize buffer to get data from
 */
inline void gSerializeObj(SerializeBuffer& buf, const DeSerializeBuffer& rbuf) {
  //  buf.reserve(rbuf.r_size());
  buf.insert(rbuf.r_linearData(), rbuf.r_size());
}

/**
 * Serialize a dynamic bitset into a buffer.
 *
 * @param [in,out] buf Serialize buffer to serialize into
 * @param [in] data dynamic bitset to serialize
 */
inline void gSerializeObj(SerializeBuffer& buf,
                          const galois::DynamicBitSet<>& data) {
  gSerializeObj(buf, data.size());
  gSerializeObj(buf, data.get_vec());
}

// we removed the functions in Bag.h that this function requires, so this
// won't work
#if 0
/**
 * For serializing insertBag.
 * Insert contigous memory chunks for each thread
 * and clear it.
 * Can not be const.
 * Implemention below makes sure that it can be deserialized
 * into a linear sequence like vector or deque.
 */
template<typename T>
inline void gSerializeObj(SerializeBuffer& buf, galois::InsertBag<T>& bag){
  gSerializeObj(buf, bag.size());
  auto headerVec = bag.getHeads();
  size_t totalSize = 0;
  for(auto h : headerVec){
    size_t localSize = (h->dend - h->dbegin);
    buf.insert((uint8_t*)h->dbegin, localSize*sizeof(T));
    totalSize += (h->dend - h->dbegin);
  }

  assert(totalSize == bag.size());
  bag.clear();
}
#endif
} // namespace internal

/**
 * LazyRef structure; used to store both a type and an offset to begin
 * saving data into
 */
template <typename T>
struct LazyRef {
  size_t off;
};

/**
 * Lazy serialize: doesn't actually serialize the data itself, but only
 * reserves space for it in the serialize buffer + serializes the
 * passed in num.
 */
template <typename Seq>
static inline LazyRef<typename Seq::value_type>
gSerializeLazySeq(SerializeBuffer& buf, unsigned num, Seq*) {
  static_assert(is_memory_copyable<typename Seq::value_type>::value,
                "Not POD Sequence");
  typename Seq::size_type size = num;
  internal::gSerializeObj(buf, size);
  size_t tsize = sizeof(typename Seq::value_type);
  return LazyRef<typename Seq::value_type>{buf.encomber(tsize * num)};
}

/**
 * Lazy serialize: given an offset and type through a LazyRef object,
 * serializes a certain amount from the passed in data array.
 *
 * @param buf Buffer to serialize into
 * @param r struct with info on where to start saving data and the type
 * of the data that needs to be saved
 * @param item Number of items that need to be serialized
 * @param data Data array containing data that needs to be serialized
 */
template <typename Ty>
static inline void gSerializeLazy(SerializeBuffer& buf, LazyRef<Ty> r,
                                  unsigned item, Ty&& data) {
  size_t off     = r.off + sizeof(Ty) * item;
  uint8_t* pdata = (uint8_t*)&data;
  buf.insertAt(pdata, sizeof(Ty), off);
}

/**
 * Serialize an entire series of datatypes into a provided serialize buffer
 */
template <typename T1, typename... Args>
static inline void gSerialize(SerializeBuffer& buf, T1&& t1, Args&&... args) {
  buf.reserve(gSized(t1, args...));
  internal::gSerializeObj(buf, std::forward<T1>(t1));
  gSerialize(buf, std::forward<Args>(args)...);
}

/**
 * No-op function. "Base case" for recursive gSerialize function.
 */
static inline void gSerialize(SerializeBuffer&) {}

////////////////////////////////////////////////////////////////////////////////
// Deserialize support
////////////////////////////////////////////////////////////////////////////////

namespace internal {

/**
 * Deserialize a memcopyable object from a buffer.
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] Data to deserialize into
 */
template <typename T>
void gDeserializeObj(
    DeSerializeBuffer& buf, T& data,
    typename std::enable_if<is_memory_copyable<T>::value>::type* = 0) {
  uint8_t* pdata = (uint8_t*)&data;
  buf.extract(pdata, sizeof(T));
}

/**
 * Deserialize a non-memcopyable but seralizable object from a buffer.
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] Data to deserialize into
 */
template <typename T>
void gDeserializeObj(
    DeSerializeBuffer& buf, T& data,
    typename std::enable_if<!is_memory_copyable<T>::value>::type* = 0,
    typename std::enable_if<has_serialize<T>::value>::type*       = 0) {
  data.deserialize(buf);
}

/**
 * Deserialize a pair from a buffer.
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] pair to deserialize into
 */
template <typename T1, typename T2>
void gDeserializeObj(DeSerializeBuffer& buf, std::pair<T1, T2>& data) {
  gDeserialize(buf, data.first, data.second);
}

/**
 * Deserialize into a tuple of 3. Either memcpys from buffer or deserializes
 * each element individually.
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] triple to deserialize into
 * @todo This specialization isn't being used as expected. Figure out why.
 */
template <typename T1, typename T2, typename T3>
inline void gDeserializeObj(DeSerializeBuffer& buf,
                            galois::TupleOfThree<T1, T2, T3>& data) {
  if (is_memory_copyable<T1>::value && is_memory_copyable<T2>::value &&
      is_memory_copyable<T3>::value) {
    // do memcpy straight to data
    buf.extract((uint8_t*)&data, sizeof(data));
  } else {
    // deserialize each individually
    gDeserialize(buf, data.first, data.second, data.third);
  }
}

/**
 * Deserialize into a CopyableAtomic. Loads the POD from the DeserializeBuffer
 * then stores it into the atomic.
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] copyable atomic to deserialize into
 */
template <typename T>
void gDeserializeObj(DeSerializeBuffer& buf, galois::CopyableAtomic<T>& data) {
  T tempData;
  uint8_t* pointerToTemp = (uint8_t*)&tempData;
  buf.extract(pointerToTemp, sizeof(T));
  data.store(tempData);
}

namespace {
template <int...>
struct seq {};
template <int N, int... S>
struct gens : gens<N - 1, N - 1, S...> {};
template <int... S>
struct gens<0, S...> {
  typedef seq<S...> type;
};
} // namespace

/**
 * Deserialize into a tuple.
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] tuple to serialize into
 */
template <typename... T, int... S>
void gDeserializeTuple(DeSerializeBuffer& buf, std::tuple<T...>& data,
                       seq<S...>) {
  gDeserialize(buf, std::get<S>(data)...);
}

/**
 * Wrapper for deserialization into a tuple.
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] tuple to serialize into
 */
template <typename... T>
void gDeserializeObj(DeSerializeBuffer& buf, std::tuple<T...>& data) {
  return gDeserializeTuple(buf, data, typename gens<sizeof...(T)>::type());
}

/**
 * Deserialize into a string.
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] string to serialize into
 */
template <typename A>
inline void
gDeserializeObj(DeSerializeBuffer& buf,
                std::basic_string<char, std::char_traits<char>, A>& data) {
  char c = buf.pop();
  while (c != '\0') {
    data.push_back(c);
    c = buf.pop();
  };
}

// Forward declaration of vector deserialize
template <typename T, typename Alloc>
void gDeserializeObj(DeSerializeBuffer& buf, std::vector<T, Alloc>& data);

/**
 * Deserialize into a sequence object
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param seq [in,out] sequence to deserialize into
 */
template <typename Seq>
void gDeserializeSeq(DeSerializeBuffer& buf, Seq& seq) {
  seq.clear();
  typename Seq::size_type size;
  gDeserializeObj(buf, size);
  while (size--) {
    typename Seq::value_type v;
    gDeserializeObj(buf, v);
    seq.push_back(v);
  }
}

/**
 * Deserialize into a linear sequence object (i.e. one that is mem-copyable)
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param seq [in,out] sequence to deserialize into
 */
template <typename Seq>
void gDeserializeLinearSeq(DeSerializeBuffer& buf, Seq& seq) {
  typedef typename Seq::value_type T;
  //  seq.clear();
  typename Seq::size_type size;
  gDeserializeObj(buf, size);
  // If the alignment is right, cast to a T array and insert
  if (buf.atAlignment(alignof(T))) {
    T* src = (T*)buf.r_linearData();
    seq.assign(src, &src[size]);
    buf.setOffset(buf.getOffset() + size * sizeof(T));
  } else {
    seq.resize(size);
    buf.extract((uint8_t*)seq.data(), size * sizeof(T));
  }
}

/**
 * Deserialize into a deque
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] deque to deserialize into
 */
template <typename T, typename Alloc>
void gDeserializeObj(DeSerializeBuffer& buf, std::deque<T, Alloc>& data) {
  gDeserializeSeq(buf, data);
}

/**
 * Deserialize into a vector; implementation depends on whether or not data in
 * vector is mem-copyable
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] vector to deserialize into
 */
template <typename T, typename Alloc>
void gDeserializeObj(DeSerializeBuffer& buf, std::vector<T, Alloc>& data) {
  if (is_memory_copyable<T>::value)
    gDeserializeLinearSeq(buf, data);
  else
    gDeserializeSeq(buf, data);
}

/**
 * Deserialize into a PODResizeableArray
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] PODResizeableArray to deserialize into
 */
template <typename T>
void gDeserializeObj(DeSerializeBuffer& buf, galois::PODResizeableArray<T>& data) {
  gDeserializeLinearSeq(buf, data);
}

/**
 * Deserialize into a galois deque
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] galois deque to deserialize into
 */
template <typename T, unsigned CS>
void gDeserializeObj(DeSerializeBuffer& buf, galois::gdeque<T, CS>& data) {
  gDeserializeSeq(buf, data);
}

/**
 * Deserialize into a dynamic bitset
 *
 * @param buf [in,out] Buffer to deserialize from
 * @param data [in,out] bitset to deserialize into
 */
inline void gDeserializeObj(DeSerializeBuffer& buf,
                            galois::DynamicBitSet<>& data) {
  size_t size = 0;
  gDeserializeObj(buf, size);
  data.resize(size);
  gDeserializeObj(buf, data.get_vec());
}

} // namespace internal

/**
 * Deserialize data in a buffer into a series of objects
 */
template <typename T1, typename... Args>
void gDeserialize(DeSerializeBuffer& buf, T1&& t1, Args&&... args) {
  internal::gDeserializeObj(buf, std::forward<T1>(t1));
  gDeserialize(buf, std::forward<Args>(args)...);
}

/**
 * Base case for regular gDeserialize recursive call.
 */
inline void gDeserialize(DeSerializeBuffer& buf) {}

/**
 * "Deserialize" data in an iterator type into a data object.
 *
 * @tparam Iter iterator type that has objects of type T
 * @tparam T type of data to deserialize into
 * @param iter Iterator containing data that we want to save into the passed in
 * data reference
 * @param data Object to save data in the iterator type into
 */
template <typename Iter, typename T>
auto gDeserializeRaw(Iter iter, T& data) -> decltype(
    std::declval<typename std::enable_if<is_memory_copyable<T>::value>::type>(),
    Iter()) {
  unsigned char* pdata = (unsigned char*)&data;
  for (size_t i = 0; i < sizeof(T); ++i)
    pdata[i] = *iter++;
  return iter;
}

} // namespace runtime
} // namespace galois

#endif // SERIALIZE DEF end
