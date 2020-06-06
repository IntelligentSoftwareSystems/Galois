/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

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
