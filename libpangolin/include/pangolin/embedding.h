#ifndef EMBEDDING_HPP_
#define EMBEDDING_HPP_

// bliss headers
//#include "bliss/defs.hh"
//#include "bliss/utils.hh"
//#include "bliss/bignum.hh"

#include "pangolin/element.h"

template <typename ElementTy>
class Embedding {
  // using iterator = typename std::vector<ElementTy>::iterator;
  using iterator = typename galois::gstl::Vector<ElementTy>::iterator;

public:
  Embedding() {}
  Embedding(size_t n) { elements.resize(n); }
  Embedding(const Embedding& emb) { elements = emb.elements; }
  ~Embedding() { elements.clear(); }
  VertexId get_vertex(unsigned i) const { return elements[i].get_vid(); }
  BYTE get_history(unsigned i) const { return elements[i].get_his(); }
  BYTE get_label(unsigned i) const { return elements[i].get_vlabel(); }
  BYTE get_key(unsigned i) const { return elements[i].get_key(); }
  bool empty() const { return elements.empty(); }
  iterator begin() { return elements.begin(); }
  iterator end() { return elements.end(); }
  iterator insert(iterator pos, const ElementTy& value) {
    return elements.insert(pos, value);
  }
  void push_back(ElementTy ele) { elements.push_back(ele); }
  void pop_back() { elements.pop_back(); }
  ElementTy& back() { return elements.back(); }
  const ElementTy& back() const { return elements.back(); }
  size_t size() const { return elements.size(); }
  void resize(size_t n) { elements.resize(n); }
  ElementTy* data() { return elements.data(); }
  const ElementTy* data() const { return elements.data(); }
  ElementTy get_element(unsigned i) const { return elements[i]; }
  void set_element(unsigned i, ElementTy& ele) { elements[i] = ele; }
  void set_vertex(unsigned i, VertexId vid) { elements[i].set_vertex_id(vid); }
  // std::vector<ElementTy> get_elements() const { return elements; }
  galois::gstl::Vector<ElementTy> get_elements() const { return elements; }
  void clean() { elements.clear(); }

protected:
  // std::vector<ElementTy> elements;
  galois::gstl::Vector<ElementTy> elements;
};

#endif // EMBEDDING_HPP_
