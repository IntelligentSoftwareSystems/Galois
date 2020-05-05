#pragma once
#include "pangolin/embedding.h"
#include "bliss/uintseqhash.hh"

// Basic Vertex-induced embedding
class BaseEmbedding : public Embedding<SimpleElement> {
  friend std::ostream& operator<<(std::ostream& strm, const BaseEmbedding& emb);

public:
  BaseEmbedding() {}
  BaseEmbedding(size_t n) : Embedding(n) {}
  ~BaseEmbedding() {}
  inline unsigned get_hash() const {
    bliss::UintSeqHash h;
    for (unsigned i = 0; i < size(); ++i)
      h.update(elements[i].get_vid());
    return h.get_value();
  }
  BaseEmbedding& operator=(const BaseEmbedding& other) {
    if (this == &other)
      return *this;
    elements = other.get_elements();
    return *this;
  }
  inline unsigned get_pid() const { return 0; } // not used
  inline void set_pid(unsigned) {}              // not used
  friend bool operator==(const BaseEmbedding& e1, const BaseEmbedding& e2) {
    return e1.elements == e2.elements;
  }
};

namespace std {
template <>
struct hash<BaseEmbedding> {
  std::size_t operator()(const BaseEmbedding& emb) const {
    return std::hash<int>()(emb.get_hash());
  }
};
} // namespace std
