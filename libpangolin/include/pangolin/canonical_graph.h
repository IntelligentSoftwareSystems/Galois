#ifndef CANONICAL_GRAPH_HPP_
#define CANONICAL_GRAPH_HPP_
/**
 * Code from on below link. Modified under Galois.
 *
 * https://github.com/rstream-system/RStream/
 *
 * Copyright (c) 2018, Kai Wang and the respective contributors
 * All rights reserved.
 * Reused/revised under 3-BSD
 */

#define USE_DOMAIN // use domain support

GALOIS_IGNORE_EXTERNAL_UNUSED_PARAMETERS
#include "bliss/graph.hh"
GALOIS_END_IGNORE_EXTERNAL_UNUSED_PARAMETERS

#include "pangolin/embedding.h"
#include "pangolin/edge_type.h"

typedef std::unordered_map<VertexId, BYTE> VertexMap;
typedef std::vector<bliss::Graph::Vertex> BlissVertexList;

template <typename EmbeddingTy, typename ElementTy>
class CanonicalGraph;
template <typename EmbeddingTy, typename ElementTy>
std::ostream& operator<<(std::ostream& strm,
                         const CanonicalGraph<EmbeddingTy, ElementTy>& cg);

template <typename EmbeddingTy, typename ElementTy>
class CanonicalGraph {
  friend std::ostream&
  operator<<<>(std::ostream& strm,
               const CanonicalGraph<EmbeddingTy, ElementTy>& cg);

public:
  CanonicalGraph() : number_of_vertices(0), hash_value(0) {}
  CanonicalGraph(bliss::AbstractGraph* ag,
                 bool GALOIS_USED_ONLY_IN_DEBUG(is_directed) = false) {
    assert(!is_directed);
    construct_cg(ag);
  }
  CanonicalGraph(const QuickPattern<EmbeddingTy, ElementTy>& qp,
                 bool GALOIS_USED_ONLY_IN_DEBUG(is_directed) = false) {
    assert(!is_directed);
    bliss::AbstractGraph* ag = turn_abstract(qp);
    construct_cg(ag);
  }
  ~CanonicalGraph() {}
  int cmp(const CanonicalGraph& other_cg) const {
    // compare the numbers of vertices
    if (get_num_vertices() < other_cg.get_num_vertices())
      return -1;
    if (get_num_vertices() > other_cg.get_num_vertices())
      return 1;
    // compare hash value
    if (get_hash() < other_cg.get_hash())
      return -1;
    if (get_hash() > other_cg.get_hash())
      return 1;
    // compare edges
    assert(embedding.size() == other_cg.embedding.size());
    for (unsigned i = 0; i < embedding.size(); ++i) {
      const auto& t1  = embedding.get_element(i);
      const auto& t2  = other_cg.embedding.get_element(i);
      int cmp_element = t1.cmp(t2);
      if (cmp_element != 0)
        return cmp_element;
    }
    return 0;
  }
  inline unsigned get_hash() const { return hash_value; }
  inline int get_num_vertices() const { return number_of_vertices; }
  // operator for map
  inline bool operator==(const CanonicalGraph& other) const {
    return cmp(other) == 0;
  }
  // inline EmbeddingTy& get_embedding() { return embedding; }
  inline EmbeddingTy get_embedding() const { return embedding; }
  inline void set_number_vertices(int num_vertices) {
    number_of_vertices = num_vertices;
  }
  inline void set_hash_value(unsigned int hash) { hash_value = hash; }
  inline unsigned get_quick_pattern_index(unsigned i) { return qp_idx[i]; }
  inline unsigned get_id() const { return hash_value; }
  inline void clean() { embedding.clean(); }

private:
  EmbeddingTy embedding;
  std::vector<int> qp_idx; // TODO: try gstl::Vector
  int number_of_vertices;
  unsigned hash_value;
  unsigned support;
  void construct_cg(bliss::AbstractGraph* ag) {
    number_of_vertices = ag->get_nof_vertices();
    hash_value         = ag->get_hash();
    transform_to_embedding(ag);
  }
  void transform_to_embedding(bliss::AbstractGraph* ag) {
    bliss::Graph* graph = (bliss::Graph*)ag;
    VertexSet set;
    VertexMap map;
    EdgeHeap min_heap;
    BlissVertexList vertices = graph->get_vertices_rstream();
    VertexId first_src       = init_heapAndset(vertices, min_heap, set);
    assert(first_src != (VertexId)-1);
    push_first_element(first_src, map, vertices);
#ifdef USE_DOMAIN
    bool is_first_edge = true;
#endif
    while (!min_heap.empty()) {
      Edge edge = min_heap.top();
#ifdef USE_DOMAIN
      if (is_first_edge) {
        qp_idx.push_back(edge.src_domain);
        is_first_edge = false;
      }
#endif
      push_element(edge, map, vertices);
      min_heap.pop();
      add_neighbours(edge, min_heap, vertices, set);
    }
  }
  VertexId init_heapAndset(BlissVertexList& vertices, EdgeHeap& min_heap,
                           VertexSet& set) {
    for (unsigned i = 0; i < vertices.size(); ++i) {
      if (!vertices[i].edges.empty()) {
        for (auto v : vertices[i].edges) {
#ifdef USE_DOMAIN
          min_heap.push(Edge(i, v.first, v.second.first, v.second.second));
#else
          min_heap.push(Edge(i, v));
#endif
        }
        set.insert(i);
        return i;
      }
    }
    return -1;
  }
  void push_first_element(VertexId first, VertexMap& map,
                          BlissVertexList& vertices) {
    map[first] = 0;
    embedding.push_back(
        ElementTy(first + 1, (BYTE)0, (BYTE)vertices[first].color, (BYTE)0));
  }
  void push_element(Edge& edge, VertexMap& map, BlissVertexList& vertices) {
    assert(edge.src < edge.dst);
    if (map.find(edge.src) != map.end()) {
      embedding.push_back(ElementTy(edge.dst + 1, (BYTE)0,
                                    (BYTE)vertices[edge.dst].color,
                                    (BYTE)map[edge.src]));
#ifdef USE_DOMAIN
      qp_idx.push_back(edge.dst_domain);
#endif
      if (map.find(edge.dst) == map.end()) {
        int s         = embedding.size() - 1;
        map[edge.dst] = s;
      }
    } else if (map.find(edge.dst) != map.end()) {
      embedding.push_back(ElementTy(edge.src + 1, (BYTE)0,
                                    (BYTE)vertices[edge.src].color,
                                    (BYTE)map[edge.dst]));
#ifdef USE_DOMAIN
      qp_idx.push_back(edge.src_domain);
#endif
      if (map.find(edge.src) == map.end()) {
        int s         = embedding.size() - 1;
        map[edge.src] = s;
      }
    } else {
      // wrong case
      std::cout << "wrong case!!!" << std::endl;
      throw std::exception();
    }
  }
  void add_neighbours(Edge& edge, EdgeHeap& min_heap, BlissVertexList& vertices,
                      VertexSet& set) {
    add_neighbours(edge.src, min_heap, vertices, set);
    add_neighbours(edge.dst, min_heap, vertices, set);
  }
  void add_neighbours(VertexId srcId, EdgeHeap& min_heap,
                      BlissVertexList& vertices, VertexSet& set) {
    if (set.find(srcId) == set.end()) {
      for (auto v : vertices[srcId].edges) {
#ifdef USE_DOMAIN
        VertexId dst = v.first;
#else
        VertexId dst = v;
#endif
        if (set.find(dst) == set.end()) {
#ifdef USE_DOMAIN
          Edge edge(srcId, dst, v.second.first, v.second.second);
#else
          Edge edge(srcId, dst);
#endif
          edge.swap();
          min_heap.push(edge);
        }
      }
      set.insert(srcId);
    }
  }
  static void report_aut(void* GALOIS_USED_ONLY_IN_DEBUG(param),
                         const unsigned GALOIS_UNUSED(n),
                         const unsigned* GALOIS_UNUSED(aut)) {
    assert(param);
    // fprintf((FILE*) param, "Generator: ");
    // bliss::print_permutation((FILE*) param, n, aut, 1);
    // fprintf((FILE*) param, "\n");
  }
  bliss::AbstractGraph*
  turn_abstract(const QuickPattern<EmbeddingTy, ElementTy>& qp) {
    bliss::AbstractGraph* ag = 0;
    // get the number of vertices
    std::unordered_map<VertexId, BYTE> vertices;
    for (unsigned index = 0; index < qp.get_size(); ++index) {
      auto element = qp.at(index);
      if (std::is_same<ElementTy, LabeledElement>::value)
        vertices[element.get_vid()] = element.get_vlabel();
      else
        vertices[element.get_vid()] = 0;
    }
    // construct bliss graph
    const unsigned number_vertices = vertices.size();
    ag                             = new bliss::Graph(vertices.size());
    // set vertices
    for (unsigned i = 0; i < number_vertices; ++i)
      ag->change_color(i, (unsigned)vertices[i + 1]);
    // read edges
    assert(qp.get_size() > 1);
    for (unsigned index = 1; index < qp.get_size(); ++index) {
      auto element = qp.at(index);
      VertexId from, to;
      from = qp.at(element.get_his()).get_vid();
      to   = element.get_vid();
      // std::cout << "Adding edge: " << from << " --> " << to << "\n";
      ag->add_edge(from - 1, to - 1,
                   std::make_pair((unsigned)element.get_his(), index));
    }
    bliss::Stats stats;
    const unsigned* cl = ag->canonical_form(
        stats, &report_aut, stdout); // canonical labeling. This is expensive.
    bliss::AbstractGraph* cf = ag->permute(cl); // permute to canonical form
    delete ag;
    return cf;
  }
};

template <typename EmbeddingTy, typename ElementTy>
std::ostream& operator<<(std::ostream& strm,
                         const CanonicalGraph<EmbeddingTy, ElementTy>& cg) {
  strm << "{" << cg.embedding << "; " << cg.get_num_vertices() << "}";
  return strm;
}

namespace std {
// template<>
template <typename EmbeddingTy, typename ElementTy>
struct hash<CanonicalGraph<EmbeddingTy, ElementTy>> {
  std::size_t
  operator()(const CanonicalGraph<EmbeddingTy, ElementTy>& cg) const {
    return std::hash<int>()(cg.get_hash());
  }
};
} // namespace std
#endif // CANONICAL_GRAPH_HPP_
