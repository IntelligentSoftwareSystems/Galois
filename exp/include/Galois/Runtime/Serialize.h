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

#include <boost/mpl/has_xxx.hpp>

namespace Galois {
namespace Runtime {
namespace Distributed {

//Objects with this tag have a member function which serializes them.
//Objects with this tag have a member function which replaces an
//already constructed object with the deserializes version (inplace
//deserialization with default constructor)
//We can also use this to update original objects durring writeback
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_has_serialize)
template<typename T>
struct has_serialize : public has_tt_has_serialize<T> {};

class SerializeBuffer {
  std::vector<unsigned char> bufdata;
public:

  SerializeBuffer() {
    //reserve a header
    for (size_t i = 0; i < sizeof(uintptr_t); ++i)
      bufdata.push_back(0);
  }

  //Serialize support
  inline void serialize() {}

  template<typename T>
  inline void serialize(const T& data, typename std::enable_if<std::is_pod<T>::value>::type* = 0) {
    unsigned char* pdata = (unsigned char*)&data;
    for (size_t i = 0; i < sizeof(data); ++i)
      bufdata.push_back(pdata[i]);
  }

  template<typename T>
  inline void serialize(const T& data, typename std::enable_if<has_serialize<T>::value>::type* = 0) {
    data.serialize(*this);
  }

  template<typename T, typename Alloc>
  inline void serialize(const std::deque<T, Alloc>& data) {
    typename std::deque<T, Alloc>::size_type size;
    size = data.size();
    serialize(size);
    for (auto ii = data.begin(), ee = data.end(); ii != ee; ++ii)
      serialize(*ii);
  }

  template<typename T1, typename T2>
  inline void serialize(const std::pair<T1, T2>& data) {
    serialize(data.first);
    serialize(data.second);
   }

  template<typename T1, typename T2, typename... U>
  void serialize(const T1& a1, const T2& a2, U... an) {
    serialize(a1);
    serialize(a2);
    serialize(an...);
  }

  void serialize_header(uintptr_t data) {
    unsigned char* pdata = (unsigned char*)&data;
    for (size_t i = 0; i < sizeof(data); ++i)
      bufdata[i] = pdata[i];
  }

  void* linearData() { return &bufdata[0]; }

  size_t size() const { return bufdata.size(); }

  //Utility

  void print(std::ostream& o) {
    o << "<{";
    for (auto ii = bufdata.begin(), ee = bufdata.end(); ii != ee; ++ii)
      o << (unsigned int)*ii << " ";
    o << "}>";
  }
};

class DeSerializeBuffer {
  std::vector<unsigned char> bufdata;
  int offset;
public:

  explicit DeSerializeBuffer(int count) {
    offset = 0;
    bufdata.resize(count);
  }

  //Deserialize support

  inline void deserialize() {}

  template<typename T>
  inline void deserialize(T& data, typename std::enable_if<std::is_pod<T>::value>::type* = 0) {
    unsigned char* pdata = (unsigned char*)&data;
    for (size_t i = 0; i < sizeof(data); ++i) {
      pdata[i] = bufdata[offset];
      ++offset;
    }
  }

  template<typename T>
  inline void deserialize(T& data, typename std::enable_if<has_serialize<T>::value>::type* = 0) {
    data.deserialize(*this);
  }

  template<typename T, typename Alloc>
  inline void deserialize(std::deque<T, Alloc>& data) {
    typedef typename std::deque<T, Alloc>::size_type lsty;
    lsty size;
    deserialize(size);
    data.resize(size);
    for (lsty x = 0; x < size; ++x)
      deserialize(data[x]);
  }

  template<typename T1, typename T2>
  inline void deserialize(std::pair<T1, T2>& data) {
    deserialize(data.first);
    deserialize(data.second);
   }

  template<typename T1, typename T2, typename... U>
  void deserialize(T1& a1, T2& a2, U... an) {
    deserialize(a1);
    deserialize(a2);
    deserialize(an...);
  }

  void* linearData() { return &bufdata[0]; }
  size_t size() const { return bufdata.size() - offset; }

  //Utility

  void print(std::ostream& o) {
    o << "<{(" << offset << ") ";
    for (auto ii = bufdata.begin(), ee = bufdata.end(); ii != ee; ++ii)
      o << (unsigned int)*ii << " ";
    o << "}>";
  }
};

/*
template<typename T>
void serialize(std::ostream& os, const T& data, typename std::enable_if<std::is_pod<T>::value>::type* = 0) {
  os.write((char*)&data, sizeof(T));
}

template<typename T>
void serialize(std::ostream& os, const T& data, typename std::enable_if<has_serialize<T>::value>::type* = 0) {
  data.serialize(os);
}

template<typename T>
T* deserialize(std::istream& is, typename std::enable_if<std::is_pod<T>::value>::type* = 0) {
  T* retval = new T();
  is.read((char*)retval, sizeof(T));
  return retval;
}

template<typename T>
T* deserialize(std::istream& is, typename std::enable_if<has_deserialize<T>::value>::type* = 0) {
  T* retval = new T();
  retval->deserialize(is);
  return retval;
}
*/

} //Distributed
} //Runtime
} //Galois
#endif //SERIALIZE
