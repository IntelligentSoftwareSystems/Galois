#pragma once
#include "pangolin/embedding.h"

// Edge induced embedding
template <typename ElementTy>
class EdgeInducedEmbedding;
template <typename ElementTy>
std::ostream& operator<<(std::ostream& strm,
                         const EdgeInducedEmbedding<ElementTy>& emb);

template <typename ElementTy>
class EdgeInducedEmbedding : public Embedding<ElementTy> {
  friend std::ostream& operator<<<>(std::ostream& strm,
                                    const EdgeInducedEmbedding<ElementTy>& emb);

public:
  EdgeInducedEmbedding() { qp_id = 0xFFFFFFFF; }
  EdgeInducedEmbedding(size_t n) : Embedding<ElementTy>(n) {}
  ~EdgeInducedEmbedding() {}
  void set_qpid(unsigned i) { qp_id = i; }    // set the quick pattern id
  unsigned get_qpid() const { return qp_id; } // get the quick pattern id
private:
  unsigned qp_id; // quick pattern id
};

template <typename ElementTy>
std::ostream& operator<<(std::ostream& strm,
                         const EdgeInducedEmbedding<ElementTy>& emb) {
  if (emb.empty()) {
    strm << "(empty)";
    return strm;
  }
  strm << "(";
  for (unsigned index = 0; index < emb.size() - 1; ++index)
    std::cout << emb.get_element(index) << ", ";
  std::cout << emb.get_element(emb.size() - 1);
  strm << ")";
  return strm;
}

typedef EdgeInducedEmbedding<LabeledElement> EdgeEmbedding;
