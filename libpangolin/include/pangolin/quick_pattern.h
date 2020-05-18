#ifndef QUICK_PATTERN_HPP_
#define QUICK_PATTERN_HPP_
/**
 * Code from on below link. Modified under Galois.
 *
 * https://github.com/rstream-system/RStream/
 *
 * Copyright (c) 2018, Kai Wang and the respective contributors
 * All rights reserved.
 * Reused/revised under 3-BSD
 */

#include "pangolin/embedding.h"
#include "pangolin/equivalence.h"
#include "bliss/uintseqhash.hh"

template <typename EmbTy, typename EleTy>
class QuickPattern;
template <typename EmbTy, typename EleTy>
std::ostream& operator<<(std::ostream& strm,
                         const QuickPattern<EmbTy, EleTy>& qp);

template <typename EmbTy, typename EleTy>
class QuickPattern {
  friend std::ostream& operator<<<>(std::ostream& strm,
                                    const QuickPattern<EmbTy, EleTy>& qp);

public:
  QuickPattern() {}
  QuickPattern(unsigned subgraph_size);
  QuickPattern(const EmbTy& emb);
  QuickPattern(EmbTy& emb, bool need_permute);
  QuickPattern(unsigned n, std::vector<bool> connected);
  ~QuickPattern() {}
  void get_equivalences(VertexPositionEquivalences& equ) {
    equ.set_size(size);
    for (unsigned i = 0; i < size; ++i)
      equ.add_equivalence(i, i);
    findAutomorphisms(equ);
  }
  // operator for map
  bool operator==(const QuickPattern& other) const {
    // compare edges
    assert(size == other.size);
    for (unsigned i = 0; i < size; ++i) {
      const EleTy& t1 = elements[i];
      const EleTy& t2 = other.elements[i];
      int cmp_element = t1.cmp(t2);
      if (cmp_element != 0) {
        return false;
      }
    }
    return true;
  }
  operator size_t() const {
    size_t a = 0;
    for (unsigned i = 0; i < size; ++i) {
      auto element = elements[i];
      a += element.get_vid();
    }
    return a;
  }
  inline unsigned get_hash() const { return hash_value; }
  inline void set_hash() {
    bliss::UintSeqHash h;
    h.update(size);
    // hash vertex labels and edges
    for (unsigned i = 0; i < size; ++i) {
      auto element = elements[i];
      h.update(element.get_vid());
      if (std::is_same<EleTy, LabeledElement>::value)
        h.update(element.get_vlabel());
      if (element.has_history())
        h.update(element.get_his());
    }
    hash_value = h.get_value();
    // return h.get_value();
  }
  EleTy& at(unsigned index) const { return elements[index]; }
  inline unsigned get_size() const { return size; }
  inline void clean() { delete[] elements; }
  inline unsigned get_id() const { return hash_value; }
  inline unsigned get_cgid() const { return cg_id; }
  void set_cgid(unsigned i) { cg_id = i; }

private:
  unsigned size;
  EleTy* elements;
  unsigned hash_value; // quick pattern ID
  unsigned
      cg_id; // ID of the canonical pattern that this quick pattern belongs to
  void findAutomorphisms(VertexPositionEquivalences& eq_sets);
};

namespace std {
template <typename EmbTy, typename EleTy>
struct hash<QuickPattern<EmbTy, EleTy>> {
  std::size_t operator()(const QuickPattern<EmbTy, EleTy>& qp) const {
    return std::hash<int>()(qp.get_hash());
  }
};
} // namespace std
#endif // QUICK_PATTERN_HPP_
