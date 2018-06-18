#ifndef GALOIS_UNION_FIND
#define GALOIS_UNION_FIND

#include <cstddef>

template <typename ElTy, ElTy initializer>
struct UnionFind {

  ElTy* parents;
  const size_t size;

  explicit UnionFind(size_t sz) : size(sz) {

    parents = new ElTy[size];
    for (size_t s = 0; s < sz; s++)
      parents[s] = initializer;
  }

  ElTy uf_find(ElTy e) {
    if (parents[e] == initializer)
      return e;
    ElTy tmp = e;
    ElTy rep = initializer;
    while (parents[tmp] != initializer)
      tmp = parents[tmp];
    rep = tmp;
    tmp = e;
    while (parents[tmp] != initializer) {
      parents[tmp] = rep;
      tmp          = parents[tmp];
    }
    return rep;
  }

  void uf_union(ElTy e1, ElTy e2) { parents[e1] = e2; }

  ~UnionFind() { delete parents; }
};

void test_uf() { UnionFind<int, -1> sample(10000); }
#endif // def GALOIS_UNION_FIND
