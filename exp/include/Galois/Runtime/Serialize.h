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

#include <boost/mpl/has_xxx.hpp>

namespace Galois {
namespace Runtime {
namespace Distributed {

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
  static const bool value = has_serialize<T>::value || std::is_pod<T>::value;
};

class DeSerializeBuffer;

class SerializeBuffer {
  friend DeSerializeBuffer;
  std::vector<unsigned char> bufdata;
  unsigned start;
public:

  SerializeBuffer() {
    //reserve a header
    bufdata.resize(sizeof(uintptr_t));
    start = sizeof(uintptr_t);
  }

  inline void push(const char c) {
    bufdata.push_back(c);
  }

  void serialize_header(uintptr_t data) {
    unsigned char* pdata = (unsigned char*)&data;
    for (size_t i = 0; i < sizeof(data); ++i)
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

inline void gSerialize(const SerializeBuffer&) {}

template<typename T1, typename T2>
void gSerialize(SerializeBuffer& buf, const std::pair<T1, T2>& data) {
  gSerialize(buf, data.first, data.second);
}

template<typename T1, typename T2, typename... U>
void gSerialize(SerializeBuffer& buf, const T1& a1, const T2& a2, const U&... an) {
  gSerialize(buf,a1);
  gSerialize(buf,a2);
  gSerialize(buf, an...);
}

template<typename T>
void gSerialize(SerializeBuffer& buf, const std::basic_string<T>& data) {
  typename std::basic_string<T>::size_type size;
  size = data.size();
  gSerialize(buf, size);
  for (auto ii = data.begin(), ee = data.end(); ii != ee; ++ii)
    gSerialize(buf, *ii);
}

template<typename T, typename Alloc>
void gSerialize(SerializeBuffer& buf, const std::vector<T, Alloc>& data) {
  typename std::vector<T, Alloc>::size_type size;
  size = data.size();
  gSerialize(buf, size);
  for (auto ii = data.begin(), ee = data.end(); ii != ee; ++ii)
    gSerialize(buf,*ii);
}

template<typename T, typename Alloc>
void gSerialize(SerializeBuffer& buf, const std::deque<T, Alloc>& data) {
  typename std::deque<T, Alloc>::size_type size;
  size = data.size();
  gSerialize(buf,size);
  for (auto ii = data.begin(), ee = data.end(); ii != ee; ++ii)
    gSerialize(buf,*ii);
}

template<typename T>
void gSerialize(SerializeBuffer& buf, const T& data, typename std::enable_if<std::is_pod<T>::value>::type* = 0) {
  unsigned char* pdata = (unsigned char*)&data;
  for (size_t i = 0; i < sizeof(data); ++i)
    buf.push(pdata[i]);
}

template<typename T>
void gSerialize(SerializeBuffer& buf, const T& data, typename std::enable_if<has_serialize<T>::value>::type* = 0) {
  data.serialize(buf);
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
    offset = 0;
    bufdata.swap(buf.bufdata);
  }

  unsigned char pop() {
    return bufdata[offset++];
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

inline void gDeserialize(const DeSerializeBuffer&) {}

template<typename T>
void gDeserialize(DeSerializeBuffer& buf, T& data, typename std::enable_if<std::is_pod<T>::value>::type* = 0) {
  unsigned char* pdata = (unsigned char*)&data;
  for (size_t i = 0; i < sizeof(data); ++i)
    pdata[i] = buf.pop();
}

template<typename T>
void gDeserialize(DeSerializeBuffer& buf, T& data, typename std::enable_if<has_serialize<T>::value>::type* = 0) {
  data.deserialize(buf);
}

template<typename T>
void gDeserialize(DeSerializeBuffer& buf, std::basic_string<T>& data) {
  typedef typename std::basic_string<T>::size_type lsty;
  lsty size;
  gDeserialize(buf, size);
  data.resize(size);
  for (lsty x = 0; x < size; ++x)
    gDeserialize(buf, data[x]);
}

template<typename T, typename Alloc>
void gDeserialize(DeSerializeBuffer& buf, std::deque<T, Alloc>& data) {
  typedef typename std::deque<T, Alloc>::size_type lsty;
  lsty size;
  gDeserialize(buf,size);
  data.resize(size);
  for (lsty x = 0; x < size; ++x)
    gDeserialize(buf,data[x]);
}

template<typename T, typename Alloc>
void gDeserialize(DeSerializeBuffer& buf, std::vector<T, Alloc>& data) {
  typedef typename std::vector<T, Alloc>::size_type lsty;
  lsty size;
  gDeserialize(buf,size);
  data.resize(size);
  for (lsty x = 0; x < size; ++x)
    gDeserialize(buf,data[x]);
}

template<typename T1, typename T2>
void gDeserialize(DeSerializeBuffer& buf, std::pair<T1, T2>& data) {
  gDeserialize(buf,data.first,data.second);
}

template<typename T1, typename T2, typename... U>
void gDeserialize(DeSerializeBuffer& buf, T1& a1, T2& a2, U&... an) {
  gDeserialize(buf,a1);
  gDeserialize(buf,a2);
  gDeserialize(buf,an...);
}

inline void gSerialize(SerializeBuffer& buf, const DeSerializeBuffer& rbuf) {
  for (unsigned x = 0; x < rbuf.r_size(); ++x)
    buf.push(rbuf.r_linearData()[x]);
}


} //Distributed
} //Runtime
} //Galois
#endif //SERIALIZE
