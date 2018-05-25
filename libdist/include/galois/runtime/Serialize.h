/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
#include "galois/CopyableTuple.h"
#include "galois/Bag.h"

namespace galois {
namespace runtime {

//Objects with this tag have a member function which serializes them.
//Objects with this tag have a member function which replaces an
//already constructed object with the deserializes version (inplace
//deserialization with default constructor)
//We can also use this to update original objects during writeback

class DeSerializeBuffer; // forward declaration for friend declaration

class SerializeBuffer {
  friend DeSerializeBuffer;

  typedef std::vector<uint8_t> vTy;
  vTy bufdata;
public:
  SerializeBuffer() = default;
  SerializeBuffer(SerializeBuffer&& rhs) = default;  // disable copy constructor
  //inline explicit SerializeBuffer(DeSerializeBuffer&& buf);

  SerializeBuffer(const char* d, unsigned len) : bufdata(d, d+len) {}

  inline void push(const char c) {
    bufdata.push_back(c);
  }

  void insert(const uint8_t* c, size_t bytes) {
    bufdata.insert(bufdata.end(), c, c+bytes);
  }

  void insertAt(const uint8_t* c, size_t bytes, size_t offset) {
    std::copy_n(c, bytes, bufdata.begin() + offset);
  }
  
  /**
   * @returns offset to use for insertAt
   */
  size_t encomber(size_t bytes) {
    size_t retval = bufdata.size();
    bufdata.resize(retval+bytes);
    return retval;
  }

  void reserve(size_t s) {
    bufdata.reserve(bufdata.size() + s);
  }

  const uint8_t* linearData() const { return bufdata.data(); }
  std::vector<uint8_t>& getVec() { return bufdata; }

  vTy::const_iterator begin() const { return bufdata.cbegin(); }
  vTy::const_iterator end() const { return bufdata.cend(); }

  typedef vTy::size_type size_type;
  size_type size() const { return bufdata.size(); }

  // Utility
  void print(std::ostream& o) const {
    o << "<{" << std::hex;
    for (auto& i : bufdata)
      o << (unsigned int)i << " ";
    o << std::dec << "}>";
  }

  friend std::ostream& operator<<(std::ostream& os, const SerializeBuffer& b) {
    b.print(os);
    return os;
  }
};


class DeSerializeBuffer {
  friend SerializeBuffer;

  std::vector<uint8_t> bufdata;
  int offset;
public:
  DeSerializeBuffer() :offset(0) {}
  DeSerializeBuffer(DeSerializeBuffer&&) = default; // disable copy constructor
  DeSerializeBuffer(std::vector<uint8_t>&& v, uint32_t start = 0) 
    : bufdata(std::move(v)), offset(start) {}

  explicit DeSerializeBuffer(std::vector<uint8_t>& data) {
    bufdata.swap(data);
    offset = 0;
  }

  explicit DeSerializeBuffer(int count) :bufdata(count), offset(0) {}

  template<typename Iter>
  DeSerializeBuffer(Iter b, Iter e) : bufdata(b,e), offset{0} {}

  explicit DeSerializeBuffer(SerializeBuffer&& buf) :offset(0) {
    bufdata.swap(buf.bufdata);
  }

  DeSerializeBuffer& operator=(DeSerializeBuffer&& buf) = default;

  void reset(int count) {
    offset = 0;
    bufdata.resize(count);
  }

  unsigned getOffset() const { return offset; }
  void setOffset(unsigned off) { assert(off <= size()); offset = off; }

  unsigned size() const { return bufdata.size(); }

  bool empty() const {return bufdata.empty(); }

  unsigned char pop() {
    return bufdata.at(offset++);
  }

  void pop_back(unsigned x) { bufdata.resize(bufdata.size() - x); }

  void extract(uint8_t* dst, size_t num) {
    memcpy(dst, &bufdata[offset], num);
    offset += num;
  }

  std::vector<uint8_t>& getVec() { return bufdata; }

  void* linearData() { return &bufdata[0]; }

  const uint8_t* r_linearData() const { return &bufdata[offset]; }
  size_t r_size() const { return bufdata.size() - offset; }

  bool atAlignment(size_t a) { return (uintptr_t)r_linearData() % a == 0; }
    

  // Utility
  void print(std::ostream& o) const {
    o << "<{(" << offset << ") " << std::hex;
    for (auto ii = bufdata.begin(), ee = bufdata.end(); ii != ee; ++ii)
      o << (unsigned int)*ii << " ";
    o << std::dec << "}>";
  }

  friend std::ostream& operator<<(std::ostream& os, const DeSerializeBuffer& buf) {
    buf.print(os);
    return os;
  }
};

namespace internal {

template<typename T>
__attribute__((always_inline)) constexpr size_t gSizedObj(const T& data,
                   typename std::enable_if<is_memory_copyable<T>::value>::type* = 0)
{
  return sizeof(T);
}

template<typename T>
__attribute__((always_inline)) constexpr size_t gSizedObj(const T& data,
                   typename std::enable_if<!is_memory_copyable<T>::value>::type* = 0,
                   typename std::enable_if<has_serialize<T>::value>::type* = 0)
{
  return sizeof(uintptr_t);
}

template<typename T1, typename T2>
inline size_t gSizedObj(const std::pair<T1, T2>& data) {
  return gSizedObj(data.first) + gSizedObj(data.second);
}

template<typename Seq>
size_t gSizedSeq(const Seq& seq) {
  typename Seq::size_type size = seq.size();
  typedef typename Seq::value_type T;
  size_t tsize = std::conditional<is_memory_copyable<T>::value, 
                                  std::integral_constant<size_t, sizeof(T)>,
                                  std::integral_constant<size_t, sizeof(uintptr_t)>>::type::value;
  return sizeof(size) + tsize * size;
}

template<typename T, typename Alloc>
inline size_t gSizedObj(const std::vector<T, Alloc>& data) {
  return gSizedSeq(data);
}

template<typename T, typename Alloc>
inline size_t gSerializeObj(const std::deque<T, Alloc>& data) {
  return gSizedSeq(data);
}

template<typename T, unsigned CS>
inline size_t gSizedObj(const galois::gdeque<T,CS>& data) {
  return gSizedSeq(data);
}

template <typename A>
inline size_t gSizedObj(const std::basic_string<char, std::char_traits<char>, A>& data) {
  return data.length() + 1;
}


inline size_t gSizedObj(const SerializeBuffer& data) {
  return data.size();
}

inline size_t gSizedObj(const DeSerializeBuffer& rbuf) {
  return rbuf.r_size();
}

template<typename T>
inline size_t gSizedObj(const galois::InsertBag<T>& bag){
  return bag.size();
}

inline size_t adder() { return 0; }
inline size_t adder(size_t a) { return a; }
template<typename... Args>
inline size_t adder(size_t a, size_t b, Args&&... args) { return a + b + adder(args...); }

} // end internal namespace

template<typename... Args>
static inline size_t gSized(Args&&... args) {
  return internal::adder(internal::gSizedObj(args)...);
}

////////////////////////////////////////////////////////////////////////////////
// Serialize support
////////////////////////////////////////////////////////////////////////////////

namespace internal {

template<typename T>
inline void gSerializeObj(SerializeBuffer& buf, const T& data,
                   typename std::enable_if<is_memory_copyable<T>::value>::type* = 0)
{
  uint8_t* pdata = (uint8_t*)&data;
  buf.insert(pdata, sizeof(T));
}

template<typename T>
inline void gSerializeObj(SerializeBuffer& buf, const T& data,
                   typename std::enable_if<!is_memory_copyable<T>::value>::type* = 0,
                   typename std::enable_if<has_serialize<T>::value>::type* = 0)
{
  data.serialize(buf);
}

template<typename T1, typename T2>
inline void gSerializeObj(SerializeBuffer& buf, const std::pair<T1, T2>& data) {
  gSerialize(buf, data.first, data.second);
}

/**
 * Specialization for tuple of 3. Either memcpys entire struct or serializes
 * each element individually.
 */
template<typename T1, typename T2, typename T3>
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
 * Specialization for copyable atomic: load atomic data as a plain old 
 * datatype (POD) and mem copy it to the buffer.
 */
template<typename T>
inline void gSerializeObj(SerializeBuffer& buf, 
                          const galois::CopyableAtomic<T>& data){
  T temp = data.load();
  buf.insert((uint8_t*)(&temp), sizeof(T));
}

template <typename A>
inline void gSerializeObj(SerializeBuffer& buf, const std::basic_string<char, std::char_traits<char>, A>& data) {
  buf.insert((uint8_t*)data.data(), data.length()+1);
}

// Forward declaration of vector serialize
template<typename T, typename Alloc>
inline void gSerializeObj(SerializeBuffer& buf, const std::vector<T, Alloc>& data); 

// Fixme: specialize for Sequences with consecutive PODS
template<typename Seq>
void gSerializeSeq(SerializeBuffer& buf, const Seq& seq) {
  typename Seq::size_type size = seq.size();
  //  typedef decltype(*seq.begin()) T;

  // size_t tsize = std::conditional<is_memory_copyable<T>::value, 
  //   std::integral_constant<size_t, sizeof(T)>,
  //   std::integral_constant<size_t, 1>>::type::value;
  //  buf.reserve(size * tsize + sizeof(size));
  gSerializeObj(buf, size);
  for (auto& o : seq)
    gSerializeObj(buf, o);
}

template<typename Seq>
void gSerializeLinearSeq(SerializeBuffer& buf, const Seq& seq) {
  typename Seq::size_type size = seq.size();
  typedef typename Seq::value_type T;
  size_t tsize = sizeof(T);
  //  buf.reserve(size * tsize + sizeof(size));
  gSerializeObj(buf, size);
  buf.insert((uint8_t*)seq.data(), size*tsize);
}

template<typename T, typename Alloc>
inline void gSerializeObj(SerializeBuffer& buf, const std::vector<T, Alloc>& data) {
  if (is_memory_copyable<T>::value) 
    gSerializeLinearSeq(buf, data);
  else
    gSerializeSeq(buf, data);
}

template<typename T, typename Alloc>
inline void gSerializeObj(SerializeBuffer& buf, const std::deque<T, Alloc>& data) {
  gSerializeSeq(buf, data);
}

template<typename T, unsigned CS>
inline void gSerializeObj(SerializeBuffer& buf, const galois::gdeque<T,CS>& data) {
  gSerializeSeq(buf,data);
}

inline void gSerializeObj(SerializeBuffer& buf, const SerializeBuffer& data) {
  //  buf.reserve(data.size());
  buf.insert(data.linearData(), data.size());
}

inline void gSerializeObj(SerializeBuffer& buf, const DeSerializeBuffer& rbuf) {
  //  buf.reserve(rbuf.r_size());
  buf.insert(rbuf.r_linearData(), rbuf.r_size());
}

//template<typename T>
//inline void gSerializeObj(SerializeBuffer& buf, const std::vector<galois::CopyableAtomic<T>>& data){
  //gSerializeSeq(buf, data);
//}

inline void gSerializeObj(SerializeBuffer& buf, 
                          const galois::DynamicBitSet& data) {
  gSerializeObj(buf, data.size());
  gSerializeObj(buf, data.get_vec());
}

// TODO removed the functions in Bag.h that this function requires, so this 
// won't work
///**
// * For serializing insertBag.
// * Insert contigous memory chunks for each thread
// * and clear it.
// * Can not be const.
// * Implemention below makes sure that it can be deserialized
// * into a linear sequence like vector or deque.
// */
//template<typename T>
//inline void gSerializeObj(SerializeBuffer& buf, galois::InsertBag<T>& bag){
//  gSerializeObj(buf, bag.size());
//  auto headerVec = bag.getHeads();
//  size_t totalSize = 0;
//  for(auto h : headerVec){
//    size_t localSize = (h->dend - h->dbegin);
//    buf.insert((uint8_t*)h->dbegin, localSize*sizeof(T));
//    totalSize += (h->dend - h->dbegin);
//  }
//
//  assert(totalSize == bag.size());
//  bag.clear();
//}

} // end internal namespace

template<typename T>
struct LazyRef { size_t off; };

template<typename Seq>
static inline LazyRef<typename Seq::value_type> gSerializeLazySeq(SerializeBuffer& buf, unsigned num, Seq*) {
  static_assert(is_memory_copyable<typename Seq::value_type>::value, "Not POD Sequence");
  typename Seq::size_type size = num;
  internal::gSerializeObj(buf, size);
  size_t tsize = sizeof(typename Seq::value_type);
  return LazyRef<typename Seq::value_type>{buf.encomber(tsize*num)};
}

template<typename Ty>
static inline void gSerializeLazy(SerializeBuffer& buf, LazyRef<Ty> r, unsigned item, Ty&& data) {
  size_t off = r.off + sizeof(Ty) * item;
  uint8_t* pdata = (uint8_t*)&data;
  buf.insertAt(pdata, sizeof(Ty), off);
}

template<typename T1, typename... Args>
static inline void gSerialize(SerializeBuffer& buf, T1&& t1, Args&&... args) {
  buf.reserve(gSized(t1, args...));
  internal::gSerializeObj(buf, std::forward<T1>(t1));
  gSerialize(buf, std::forward<Args>(args)...);
}

static inline void gSerialize(SerializeBuffer&) {}

//template<typename... Args>
//void gSerialize(SerializeBuffer& buf, Args&&... args) {
//  int dummy[sizeof...(Args)] = { (internal::gSerializeObj(buf, std::forward<Args>(args)), 0)...};
//}


////////////////////////////////////////////////////////////////////////////////
// Deserialize support
////////////////////////////////////////////////////////////////////////////////

namespace internal {

template<typename T>
void gDeserializeObj(DeSerializeBuffer& buf, T& data,
                     typename std::enable_if<is_memory_copyable<T>::value>::type* = 0) 
{
  uint8_t* pdata = (uint8_t*)&data;
  buf.extract(pdata, sizeof(T));
}

template<typename T>
void gDeserializeObj(DeSerializeBuffer& buf, T& data,
		     typename std::enable_if<!is_memory_copyable<T>::value>::type* = 0,
                     typename std::enable_if<has_serialize<T>::value>::type* = 0) 
{
  data.deserialize(buf);
}

template<typename T1, typename T2>
void gDeserializeObj(DeSerializeBuffer& buf, std::pair<T1, T2>& data) {
  gDeserialize(buf, data.first, data.second);
}

/**
 * Specialization for tuple of 3. Either memcpys from buffer or deserializes
 * each element individually.
 */
template<typename T1, typename T2, typename T3>
inline void gSerializeObj(DeSerializeBuffer& buf, 
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
 * Specialization for CopyableAtomic. Loads the POD from the DeserializeBuffer
 * then stores it into the atomic.
 */
template<typename T>
void gDeserializeObj(DeSerializeBuffer& buf, galois::CopyableAtomic<T>& data){
  T tempData;
  uint8_t* pointerToTemp = (uint8_t*)&tempData;
  buf.extract(pointerToTemp, sizeof(T));
  data.store(tempData); 
}

namespace {
template<int ...> struct seq {};
template<int N, int ...S> struct gens : gens<N-1, N-1, S...> {};
template<int ...S> struct gens<0, S...>{ typedef seq<S...> type; };
}

template<typename... T, int... S>
void gDeserializeTuple(DeSerializeBuffer& buf, std::tuple<T...>& data, seq<S...>) {
  gDeserialize(buf, std::get<S>(data)...);
}

template<typename... T>
void gDeserializeObj(DeSerializeBuffer& buf, std::tuple<T...>& data) {
  return gDeserializeTuple(buf, data, typename gens<sizeof...(T)>::type());
}

template <typename A>
inline void gDeserializeObj(DeSerializeBuffer& buf, std::basic_string<char, std::char_traits<char>, A>& data) {
  char c = buf.pop();
  while(c != '\0') {
    data.push_back(c);
    c = buf.pop();
  };
}

// Forward declaration of vector deserialize
template<typename T, typename Alloc>
void gDeserializeObj(DeSerializeBuffer& buf, std::vector<T, Alloc>& data);

template<typename Seq>
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

template<typename Seq>
void gDeserializeLinearSeq(DeSerializeBuffer& buf, Seq& seq) {
  typedef typename Seq::value_type T;
  //  seq.clear();
  typename Seq::size_type size;
  gDeserializeObj(buf, size);
  //If the alignment is right, cast to a T array and insert
  if (buf.atAlignment(alignof(T))) {
    T* src = (T*)buf.r_linearData();
    seq.assign(src, &src[size]);
    buf.setOffset(buf.getOffset() + size * sizeof(T));
  } else {
    seq.resize(size);
    buf.extract((uint8_t*)seq.data(), size * sizeof(T));
  }
}


template<typename T, typename Alloc>
void gDeserializeObj(DeSerializeBuffer& buf, std::deque<T, Alloc>& data) {
  gDeserializeSeq(buf, data);
}

template<typename T, typename Alloc>
void gDeserializeObj(DeSerializeBuffer& buf, std::vector<T, Alloc>& data) {
  if (is_memory_copyable<T>::value)
    gDeserializeLinearSeq(buf, data);
  else
    gDeserializeSeq(buf, data);
}

template<typename T, unsigned CS>
void gDeserializeObj(DeSerializeBuffer& buf, galois::gdeque<T,CS>& data) {
  gDeserializeSeq(buf, data);
}

inline void gDeserializeObj(DeSerializeBuffer& buf, galois::DynamicBitSet& data) {
  size_t size = 0;
  gDeserializeObj(buf, size);
  data.resize(size);
  gDeserializeObj(buf, data.get_vec());
}

} //namespace internal

//SerializeBuffer::SerializeBuffer(DeSerializeBuffer&& buf) {
//  bufdata.swap(buf.bufdata);
//}

template<typename T1, typename... Args>
void gDeserialize(DeSerializeBuffer& buf, T1&& t1, Args&&... args) {
  internal::gDeserializeObj(buf, std::forward<T1>(t1));
  gDeserialize(buf, std::forward<Args>(args)...);
}

inline void gDeserialize(DeSerializeBuffer& buf) { }

template<typename Iter, typename T>
auto gDeserializeRaw(Iter iter, T& data) ->
  decltype(std::declval<typename std::enable_if<is_memory_copyable<T>::value>::type>(), Iter())
{
  unsigned char* pdata = (unsigned char*)&data;
  for (size_t i = 0; i < sizeof(T); ++i)
    pdata[i] = *iter++;
  return iter;
}

} // end runtime namespace
} // end galois namespace

#endif // SERIALIZE DEF end
