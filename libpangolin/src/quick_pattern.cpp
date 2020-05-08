#include "pangolin/quick_pattern.h"
#include "pangolin/vertex_embedding.h"
#include "pangolin/edge_embedding.h"

template <typename EmbTy, typename EleTy>
QuickPattern<EmbTy, EleTy>::QuickPattern(unsigned subgraph_size) {
  hash_value = 0;
  cg_id      = 0;
  size       = subgraph_size / sizeof(EleTy);
  elements   = new EleTy[size];
}

template <typename EmbTy, typename EleTy>
QuickPattern<EmbTy, EleTy>::QuickPattern(const EmbTy& emb) {
  cg_id          = 0;
  size           = emb.size();
  unsigned bytes = size * sizeof(EleTy);
  elements       = new EleTy[size];
  std::memcpy(elements, emb.data(), bytes);
  VertexId new_id = 1;
  std::unordered_map<VertexId, VertexId> map;
  for (unsigned i = 0; i < size; i++) {
    auto& element   = elements[i];
    VertexId old_id = element.get_vid();
    auto iterator   = map.find(old_id);
    if (iterator == map.end()) {
      element.set_vertex_id(new_id);
      map[old_id] = new_id++;
    } else
      element.set_vertex_id(iterator->second);
  }
  set_hash();
}

template <typename EmbTy, typename EleTy>
QuickPattern<EmbTy, EleTy>::QuickPattern(EmbTy& emb, bool) {
  cg_id          = 0;
  size           = emb.size();
  unsigned bytes = size * sizeof(EleTy);
  elements       = new EleTy[size];
  std::memcpy(elements, emb.data(), bytes);
  VertexId new_id = 1;
  if (std::is_same<EleTy, LabeledElement>::value) {
    if (size == 3) {
      BYTE l1 = emb.get_label(1);
      BYTE l2 = emb.get_label(2);
      BYTE h2 = emb.get_history(2);
      elements[0].set_vertex_id(1);
      elements[1].set_vertex_id(2);
      elements[2].set_vertex_id(3);
      if (h2 == 0) {
        if (l1 < l2) {
          elements[1].set_vertex_label(l2);
          elements[2].set_vertex_label(l1);
          VertexId v1 = emb.get_vertex(1);
          VertexId v2 = emb.get_vertex(2);
          emb.set_vertex(1, v2);
          emb.set_vertex(2, v1);
        }
      } else {
        assert(h2 == 1);
        elements[0].set_vertex_label(l1);
        elements[2].set_history_info(0);
        BYTE l0     = emb.get_label(0);
        VertexId v0 = emb.get_vertex(0);
        VertexId v1 = emb.get_vertex(1);
        VertexId v2 = emb.get_vertex(2);
        if (l0 < l2) {
          elements[1].set_vertex_label(l2);
          elements[2].set_vertex_label(l0);
          emb.set_vertex(1, v2);
          emb.set_vertex(2, v0);
        } else {
          elements[1].set_vertex_label(l0);
          emb.set_vertex(1, v0);
        }
        emb.set_vertex(0, v1);
      }
    } else { // size > 3
      std::unordered_map<VertexId, VertexId> map;
      for (unsigned i = 0; i < size; i++) {
        auto& element   = elements[i];
        VertexId old_id = element.get_vid();
        auto iterator   = map.find(old_id);
        if (iterator == map.end()) {
          element.set_vertex_id(new_id);
          map[old_id] = new_id++;
        } else
          element.set_vertex_id(iterator->second);
      }
    }
  } else { // non-label
    std::unordered_map<VertexId, VertexId> map;
    for (unsigned i = 0; i < size; i++) {
      auto& element   = elements[i];
      VertexId old_id = element.get_vid();
      auto iterator   = map.find(old_id);
      if (iterator == map.end()) {
        element.set_vertex_id(new_id);
        map[old_id] = new_id++;
      } else
        element.set_vertex_id(iterator->second);
    }
  }
  set_hash();
}

template <typename EmbTy, typename EleTy>
QuickPattern<EmbTy, EleTy>::QuickPattern(unsigned n,
                                         std::vector<bool> connected) {
  cg_id = 0;
  size  = std::count(connected.begin(), connected.end(), true) +
         1; // number of edges + 1
  elements = new EleTy[size];
  std::vector<unsigned> pos(n, 0);
  pos[1] = 1;
  pos[2] = 2;
  elements[0].set_vertex_id(1);
  elements[0].set_history_info(0);
  elements[1].set_vertex_id(2);
  elements[1].set_history_info(0);
  int count = 2;
  int l     = 1;
  for (unsigned i = 2; i < n; i++) {
    if (i < n - 2)
      pos[i + 1] = pos[i];
    for (unsigned j = 0; j < i; j++) {
      if (connected[l++]) {
        if (i < n - 2)
          pos[i + 1]++;
        elements[count].set_vertex_id(i + 1);
        elements[count++].set_history_info(pos[j]);
      }
    }
  }
  set_hash();
}

template <typename EmbTy, typename EleTy>
void QuickPattern<EmbTy, EleTy>::findAutomorphisms(
    VertexPositionEquivalences& eq_sets) {
  if (size == 2) { // single-edge
    if (at(0).get_vlabel() == at(1).get_vlabel()) {
      eq_sets.add_equivalence(0, 1);
      eq_sets.add_equivalence(1, 0);
    }
  } else if (size == 3) { // two-edge chain
    if (at(2).get_his() == 0) {
      if (at(1).get_vlabel() == at(2).get_vlabel()) {
        eq_sets.add_equivalence(1, 2);
        eq_sets.add_equivalence(2, 1);
      }
    } else if (at(2).get_his() == 1) {
      if (at(0).get_vlabel() == at(2).get_vlabel()) {
        eq_sets.add_equivalence(0, 2);
        eq_sets.add_equivalence(2, 0);
      }
    } else
      std::cout << "Error\n";
  } else if (size == 4) { // three-edge chain or star
    if (at(2).get_his() == 0) {
      if (at(3).get_his() == 0) {
        if (at(1).get_vlabel() == at(2).get_vlabel()) {
          eq_sets.add_equivalence(1, 2);
          eq_sets.add_equivalence(2, 1);
        }
        if (at(1).get_vlabel() == at(3).get_vlabel()) {
          eq_sets.add_equivalence(1, 3);
          eq_sets.add_equivalence(3, 1);
        }
        if (at(2).get_vlabel() == at(3).get_vlabel()) {
          eq_sets.add_equivalence(2, 3);
          eq_sets.add_equivalence(3, 2);
        }
      } else if (at(3).get_his() == 1) {
        if (at(2).get_vlabel() == at(3).get_vlabel()) {
          eq_sets.add_equivalence(2, 3);
          eq_sets.add_equivalence(3, 2);
        }
        if (at(0).get_vlabel() == at(1).get_vlabel()) {
          eq_sets.add_equivalence(0, 1);
          eq_sets.add_equivalence(1, 0);
        }
      } else if (at(3).get_his() == 2) {
        if (at(1).get_vlabel() == at(3).get_vlabel()) {
          eq_sets.add_equivalence(1, 3);
          eq_sets.add_equivalence(3, 1);
        }
        if (at(0).get_vlabel() == at(2).get_vlabel()) {
          eq_sets.add_equivalence(0, 2);
          eq_sets.add_equivalence(2, 0);
        }
      } else
        std::cout << "Error\n";
    } else if (at(2).get_his() == 1) {
      if (at(3).get_his() == 0) {
        if (at(2).get_vlabel() == at(3).get_vlabel()) {
          eq_sets.add_equivalence(2, 3);
          eq_sets.add_equivalence(3, 2);
        }
        if (at(0).get_vlabel() == at(1).get_vlabel()) {
          eq_sets.add_equivalence(0, 1);
          eq_sets.add_equivalence(1, 0);
        }
      } else if (at(3).get_his() == 1) {
        if (at(0).get_vlabel() == at(2).get_vlabel()) {
          eq_sets.add_equivalence(0, 2);
          eq_sets.add_equivalence(2, 0);
        }
        if (at(0).get_vlabel() == at(3).get_vlabel()) {
          eq_sets.add_equivalence(0, 3);
          eq_sets.add_equivalence(3, 0);
        }
        if (at(2).get_vlabel() == at(3).get_vlabel()) {
          eq_sets.add_equivalence(2, 3);
          eq_sets.add_equivalence(3, 2);
        }
      } else if (at(3).get_his() == 2) {
        if (at(0).get_vlabel() == at(3).get_vlabel()) {
          eq_sets.add_equivalence(0, 3);
          eq_sets.add_equivalence(3, 0);
        }
        if (at(1).get_vlabel() == at(2).get_vlabel()) {
          eq_sets.add_equivalence(1, 2);
          eq_sets.add_equivalence(2, 1);
        }
      } else
        std::cout << "Error\n";
    } else
      std::cout << "Error\n";
  } else { // four-edge and beyond
    std::cout << "Currently not supported\n";
  }
}

template <typename EmbTy, typename EleTy>
std::ostream& operator<<(std::ostream& strm,
                         const QuickPattern<EmbTy, EleTy>& qp) {
  if (qp.get_size() == 0) {
    strm << "(empty)";
    return strm;
  }
  strm << "(";
  for (unsigned index = 0; index < qp.get_size() - 1; ++index)
    strm << qp.elements[index] << ", ";
  strm << qp.elements[qp.get_size() - 1];
  strm << ")";
  return strm;
}

template class QuickPattern<VertexEmbedding, SimpleElement>; // Motif
template class QuickPattern<EdgeInducedEmbedding<StructuralElement>,
                            StructuralElement>; // Motif
template class QuickPattern<EdgeInducedEmbedding<LabeledElement>,
                            LabeledElement>; // FSM
