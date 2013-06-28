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
template <class T> struct is_trivially_copyable;
template <class _Tp> struct is_trivially_copyable
  : public std::integral_constant<bool, __is_trivially_copyable(_Tp)>
{};
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

template<typename T>
struct is_serializable {
  static const bool value = has_serialize<T>::value || std::is_trivially_copyable<T>::value;
};

class DeSerializeBuffer;

class SerializeBuffer {
  friend DeSerializeBuffer;
  std::vector<unsigned char> bufdata;
  unsigned start;
public:

  SerializeBuffer() {
    //reserve a header
    bufdata.resize(sizeof(void*));
    start = sizeof(void*);
  }

  inline void push(const char c) {
    bufdata.push_back(c);
  }

  void serialize_header(void* data) {
    assert(bufdata.size() >= sizeof(uintptr_t));
    unsigned char* pdata = (unsigned char*)&data;
    for (size_t i = 0; i < sizeof(void*); ++i)
      bufdata[i] = pdata[i];
    start = 0;
  }

  void* linearData() { return &bufdata[start]; }

  size_t size() const { return bufdata.size() - start; }

  //Utility

  void print(std::ostream& o) {
    o << "<{";
    for (auto ii = bufdata.begin(), ee = bufdata.end(); ii != ee; ++ii)
      o << (unsigned int)*ii << " ";
    o << "}>";
  }
};

template<typename T1, typename T2>
bool gSerialize(SerializeBuffer& buf, const std::pair<T1, T2>& data) {
  gSerialize(buf, data.first, data.second);
  return true;
}

template<typename... Args>
bool gSerialize(SerializeBuffer& buf, const Args&... args) {
  //  std::cerr << "\n" << sizeof...(args) << "\n";
  __attribute__((unused)) bool tmp[] = {gSerialize(buf, args)...};
  return true;
}


inline bool gSerialize(SerializeBuffer& buf, const std::string& data) {
  typename std::string::size_type size;
  size = data.size();
  gSerialize(buf, size);
  for (auto x = 0 ; x < size; ++x)
    gSerialize(buf, data.at(x));
  return true;
}

template<typename T, typename Alloc>
bool gSerialize(SerializeBuffer& buf, const std::vector<T, Alloc>& data) {
  typename std::vector<T, Alloc>::size_type size;
  size = data.size();
  gSerialize(buf, size);
  for (auto ii = data.begin(), ee = data.end(); ii != ee; ++ii)
    gSerialize(buf,*ii);
  return true;
}

template<typename T, typename Alloc>
bool gSerialize(SerializeBuffer& buf, const std::deque<T, Alloc>& data) {
  typename std::deque<T, Alloc>::size_type size;
  size = data.size();
  gSerialize(buf,size);
  for (auto ii = data.begin(), ee = data.end(); ii != ee; ++ii)
    gSerialize(buf,*ii);
  return true;
}

template<typename T, unsigned CS>
bool gSerialize(SerializeBuffer& buf, const Galois::gdeque<T,CS>& data) {
  typename gdeque<T,CS>::size_type size;
  size = data.size();
  gSerialize(buf,size);
  for (auto ii = data.begin(), ee = data.end(); ii != ee; ++ii)
    gSerialize(buf,*ii);
  return true;
}

extern uint32_t networkHostID;

template<typename T>
bool gSerialize(SerializeBuffer& buf, const T& data, typename std::enable_if<std::is_trivially_copyable<T>::value>::type* = 0) {
  //  std::cerr << networkHostID <<  " sesize " << sizeof(T) << " of " << typeid(T).name() << " " << "\n";
  unsigned char* pdata = (unsigned char*)&data;
  for (size_t i = 0; i < sizeof(T); ++i)
    buf.push(pdata[i]);
  return true;
}

template<typename T>
bool gSerialize(SerializeBuffer& buf, const T& data, typename std::enable_if<has_serialize<T>::value>::type* = 0) {
  data.serialize(buf);
  return true;
}


class DeSerializeBuffer {
  std::vector<unsigned char> bufdata;
  int offset;
public:

  explicit DeSerializeBuffer(int count) {
    offset = 0;
    bufdata.resize(count);
  }

  explicit DeSerializeBuffer(SerializeBuffer&& buf) {
    bufdata.swap(buf.bufdata);
    offset = buf.start;
  }

  unsigned char pop() {
    return bufdata.at(offset++);
  }

  void* linearData() { return &bufdata[0]; }

  const unsigned char* r_linearData() const { return &bufdata[offset]; }
  size_t r_size() const { return bufdata.size() - offset; }

  //Utility

  void print(std::ostream& o) {
    o << "<{(" << offset << ") ";
    for (auto ii = bufdata.begin(), ee = bufdata.end(); ii != ee; ++ii)
      o << (unsigned int)*ii << " ";
    o << "}>";
  }
};


  //Deserialize support

template<typename T>
bool gDeserialize(DeSerializeBuffer& buf, T& data, typename std::enable_if<std::is_trivially_copyable<T>::value>::type* = 0) {
  //  std::cerr << networkHostID <<  " desize " << sizeof(T) << " of " << typeid(T).name() << "\n";
  unsigned char* pdata = (unsigned char*)&data;
  for (size_t i = 0; i < sizeof(T); ++i)
    pdata[i] = buf.pop();
  return true;
}

template<typename T>
bool gDeserialize(DeSerializeBuffer& buf, T& data, typename std::enable_if<has_serialize<T>::value>::type* = 0) {
  data.deserialize(buf);
  return true;
}

inline bool gDeserialize(DeSerializeBuffer& buf, std::string& data) {
  typedef typename std::string::size_type lsty;
  lsty size;
  gDeserialize(buf, size);
  data.resize(size);
  for (lsty x = 0; x < size; ++x)
    gDeserialize(buf, data.at(x));
  return true;
}

template<typename T, typename Alloc>
bool gDeserialize(DeSerializeBuffer& buf, std::deque<T, Alloc>& data) {
  typedef typename std::deque<T, Alloc>::size_type lsty;
  lsty size;
  gDeserialize(buf,size);
  data.resize(size);
  for (lsty x = 0; x < size; ++x)
    gDeserialize(buf,data[x]);
  return true;
}

template<typename T, typename Alloc>
bool gDeserialize(DeSerializeBuffer& buf, std::vector<T, Alloc>& data) {
  typedef typename std::vector<T, Alloc>::size_type lsty;
  lsty size;
  gDeserialize(buf,size);
  data.resize(size);
  for (lsty x = 0; x < size; ++x)
    gDeserialize(buf,data[x]);
  return true;
}

template<typename T, unsigned CS>
bool gDeserialize(DeSerializeBuffer& buf, Galois::gdeque<T,CS>& data) {
  typename gdeque<T,CS>::size_type size;
  gDeserialize(buf,size);
  data.clear();
  for (unsigned x = 0; x < size; ++x) {
    T t;
    gDeserialize(buf,t);
    data.push_back(std::move(t));
  }
  return true;
}

template<typename T1, typename T2>
bool gDeserialize(DeSerializeBuffer& buf, std::pair<T1, T2>& data) {
  gDeserialize(buf,data.first,data.second);
  return true;
}

namespace {
template<int ...> struct seq {};
template<int N, int ...S> struct gens : gens<N-1, N-1, S...> {};
template<int ...S> struct gens<0, S...>{ typedef seq<S...> type; };
}
template<typename... T, int... S>
bool gDeserialize(DeSerializeBuffer& buf, std::tuple<T...>& data, seq<S...>) {
  gDeserialize(buf, std::get<S>(data) ...);
  return true;
}

template<typename... T>
bool gDeserialize(DeSerializeBuffer& buf, std::tuple<T...>& data) {
  gDeserialize(buf, data, typename gens<sizeof...(T)>::type());
  return true;
}

template<typename T1, typename T2, typename... U>
bool gDeserialize1(DeSerializeBuffer& buf, T1& a1, T2& a2, U&... an) {
  //  std::cerr << "\n";
  gDeserialize(buf,a1);
  gDeserialize(buf,a2);
  gDeserialize(buf,an...);
}

template<typename... Args>
bool gDeserialize(DeSerializeBuffer& buf, Args&... args) {
  //  std::cerr << "\n" << sizeof...(args) << "\n";
  __attribute__((unused)) bool tmp[] = {gDeserialize(buf, args)...};
  return true;
}

inline bool gSerialize(SerializeBuffer& buf, const DeSerializeBuffer& rbuf) {
  for (unsigned x = 0; x < rbuf.r_size(); ++x)
    buf.push(rbuf.r_linearData()[x]);
  return true;
}


} //Runtime
} //Galois
#endif //SERIALIZE
