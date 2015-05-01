/** Galois serialization support -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
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

#include <Galois/gdeque.h>

//from libc++, clang specific
namespace std {
#ifdef __clang__
template <class T> struct is_trivially_copyable;
template <class _Tp> struct is_trivially_copyable
  : public std::integral_constant<bool, __is_trivially_copyable(_Tp)>
{};
#else
template<class T>
using is_trivially_copyable = is_trivial<T>;
#endif
}

namespace Galois {
namespace Runtime {

//Objects with this tag have a member function which serializes them.
//Objects with this tag have a member function which replaces an
//already constructed object with the deserializes version (inplace
//deserialization with default constructor)
//We can also use this to update original objects during writeback
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_has_serialize)
template<typename T>
struct has_serialize : public has_tt_has_serialize<T> {};

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_is_copyable)
//! User assertion that class is trivially copyable
template<typename T>
struct is_copyable :  public has_tt_is_copyable<T> {};

template<typename T>
struct is_serializable {
  static const bool value = has_serialize<T>::value || is_copyable<T>::value || std::is_trivially_copyable<T>::value;
};

template<typename T>
struct is_memory_copyable {
  static const bool value = is_copyable<T>::value || std::is_trivially_copyable<T>::value;
};

class DeSerializeBuffer;

class SerializeBuffer {
  friend DeSerializeBuffer;
  typedef std::vector<char> vTy;
  vTy bufdata;
  unsigned start;
public:

  SerializeBuffer() {
    //reserve a header
    bufdata.resize(3*sizeof(void*));
    start = 3*sizeof(void*);
  }

  SerializeBuffer(SerializeBuffer&& rhs) = default;  //disable copy constructor

  inline explicit SerializeBuffer(DeSerializeBuffer&& buf);

  SerializeBuffer(const char* d, unsigned len) : bufdata(d, d+len), start(0) {}

  inline void push(const char c) {
    bufdata.push_back(c);
  }

  void insert(const char* c, size_t bytes) {
    bufdata.insert(bufdata.end(), c, c+bytes);
  }

  void reserve(size_t s) {
    bufdata.reserve(bufdata.size() + s);
  }

  void serialize_header(void* data) {
    assert(start != 0);
    unsigned char* pdata = (unsigned char*)&data;
    start -= sizeof(void*);
    for (size_t i = 0; i < sizeof(void*); ++i)
      bufdata[start + i] = pdata[i];
  }

  const char* linearData() const { return &bufdata[start]; }

  vTy::const_iterator begin() const { return bufdata.cbegin() + start; }
  vTy::const_iterator end() const { return bufdata.cend(); }

  typedef vTy::size_type size_type;
  size_type size() const { return bufdata.size() - start; }

  //Utility

  void print(std::ostream& o) const {
    o << "<{" << std::hex;
    for (auto ii = bufdata.begin() + start, ee = bufdata.end(); ii != ee; ++ii)
      o << (unsigned int)*ii << " ";
    o << std::dec << "}>";
  }

  friend std::ostream& operator<<(std::ostream& os, const SerializeBuffer& b) {
    b.print(os);
    return os;
  }
};


class DeSerializeBuffer {
  friend SerializeBuffer;

  std::vector<char> bufdata;
  int offset;
public:

  DeSerializeBuffer() :offset(0) {}
  DeSerializeBuffer(DeSerializeBuffer&&) = default; //disable copy constructor

  explicit DeSerializeBuffer(int count) {
    offset = 0;
    bufdata.resize(count);
  }

  template<typename Iter>
  DeSerializeBuffer(Iter b, Iter e) : bufdata(b,e), offset{0} {}

  explicit DeSerializeBuffer(SerializeBuffer&& buf) {
    bufdata.swap(buf.bufdata);
    offset = buf.start;
  }

  DeSerializeBuffer& operator=(DeSerializeBuffer&& buf) = default;

  void reset(int count) {
    offset = 0;
    bufdata.resize(count);
  }

  unsigned size() const { return bufdata.size(); }

  unsigned char pop() {
    return bufdata.at(offset++);
  }

  void extract(char* dst, size_t num) {
    for (size_t i = 0; i < num; ++i)
      dst[i] = bufdata[offset++];
  }

  void* linearData() { return &bufdata[0]; }

  const char* r_linearData() const { return &bufdata[offset]; }
  size_t r_size() const { return bufdata.size() - offset; }

  //Utility

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




namespace detail {

template<typename T>
__attribute__((always_inline)) void gSerializeObj(SerializeBuffer& buf, const T& data,
                   typename std::enable_if<is_memory_copyable<T>::value>::type* = 0)
{
  char* pdata = (char*)&data;
  buf.insert(pdata, sizeof(T));
}

template<typename T>
__attribute__((always_inline)) void gSerializeObj(SerializeBuffer& buf, const T& data,
                   typename std::enable_if<!is_memory_copyable<T>::value>::type* = 0,
                   typename std::enable_if<has_serialize<T>::value>::type* = 0)
{
  data.serialize(buf);
}

template<typename T1, typename T2>
void gSerializeObj(SerializeBuffer& buf, const std::pair<T1, T2>& data) {
  gSerialize(buf, data.first, data.second);
}

template<typename Seq>
void gSerializeSeq(SerializeBuffer& buf, const Seq& seq) {
  typename Seq::size_type size = seq.size();
  typedef decltype(*seq.begin()) T;

  size_t tsize = std::conditional<is_memory_copyable<T>::value, 
    std::integral_constant<size_t, sizeof(T)>,
    std::integral_constant<size_t, 1>>::type::value;
  buf.reserve(size * tsize);
  gSerializeObj(buf, size);
  for (auto ii = seq.begin(), ee = seq.end(); ii != ee; ++ii)
    gSerializeObj(buf, *ii);
}

template<typename T, typename Alloc>
void gSerializeObj(SerializeBuffer& buf, const std::vector<T, Alloc>& data) {
  gSerializeSeq(buf, data);
}

template<typename T, typename Alloc>
void gSerializeObj(SerializeBuffer& buf, const std::deque<T, Alloc>& data) {
  gSerializeSeq(buf, data);
}

template<typename T, unsigned CS>
void gSerializeObj(SerializeBuffer& buf, const Galois::gdeque<T,CS>& data) {
  gSerializeSeq(buf,data);
}

inline void gSerializeObj(SerializeBuffer& buf, const std::string& data) {
  buf.insert(data.data(), data.length()+1);
}

inline void gSerializeObj(SerializeBuffer& buf, const SerializeBuffer& data) {
  buf.insert(data.linearData(), data.size());
}

inline void gSerializeObj(SerializeBuffer& buf, const DeSerializeBuffer& rbuf) {
  buf.insert(rbuf.r_linearData(), rbuf.r_size());
}

} //detail

template<typename T1, typename... Args>
static inline void gSerialize(SerializeBuffer& buf, T1&& t1, Args&&... args) {
  detail::gSerializeObj(buf, std::forward<T1>(t1));
  gSerialize(buf, std::forward<Args>(args)...);
}

static inline void gSerialize(SerializeBuffer&) {}

//template<typename... Args>
//void gSerialize(SerializeBuffer& buf, Args&&... args) {
//  int dummy[sizeof...(Args)] = { (detail::gSerializeObj(buf, std::forward<Args>(args)), 0)...};
//}


////////////////////////////////////////////////////////////////////////////////
//Deserialize support
////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
void gDeserializeObj(DeSerializeBuffer& buf, T& data,
                     typename std::enable_if<is_memory_copyable<T>::value>::type* = 0) 
{
  char* pdata = (char*)&data;
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

template<typename Seq>
void gDeserializeSeq(DeSerializeBuffer& buf, Seq& seq) {
  seq.clear();
  typename Seq::size_type size, sorig;
  gDeserializeObj(buf, size);
  sorig = size;
  while (size--) {
    typename Seq::value_type v;
    gDeserializeObj(buf, v);
    seq.push_back(v);
  }
}

inline void gDeserializeObj(DeSerializeBuffer& buf, std::string& data) {
  char c = buf.pop();
  while(c != '\0') {
    data.push_back(c);
  };
}

template<typename T, typename Alloc>
void gDeserializeObj(DeSerializeBuffer& buf, std::deque<T, Alloc>& data) {
  gDeserializeSeq(buf, data);
}

template<typename T, typename Alloc>
void gDeserializeObj(DeSerializeBuffer& buf, std::vector<T, Alloc>& data) {
  gDeserializeSeq(buf, data);
}

template<typename T, unsigned CS>
void gDeserializeObj(DeSerializeBuffer& buf, Galois::gdeque<T,CS>& data) {
  gDeserializeSeq(buf, data);
}

} //namespace detail


SerializeBuffer::SerializeBuffer(DeSerializeBuffer&& buf) {
  bufdata.swap(buf.bufdata);
  start = buf.offset;
}


template<typename T1, typename... Args>
void gDeserialize(DeSerializeBuffer& buf, T1&& t1, Args&&... args) {
  detail::gDeserializeObj(buf, std::forward<T1>(t1));
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

} //Runtime
} //Galois
#endif //SERIALIZE
