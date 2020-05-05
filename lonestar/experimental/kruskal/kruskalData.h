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

#ifndef _KRUSKAL_DATA_H_
#define _KRUSKAL_DATA_H_

#include <string>
#include <list>
#include <algorithm>
#include <utility>

#include <cstdio>
#include <cassert>

#include "galois/Atomic.h"

template <typename KNode_tp>
struct KEdge;

struct KNode {
  unsigned id;
  unsigned rank;
  KNode* rep;

public:
  KNode(unsigned id) : id(id), rank(0), rep(this) {}

  const std::string str() const {
    char s[256];
    sprintf(s, "(id=%d,rank=%d,rep=%p)", id, rank, rep);
    return s;
  }

  KNode* getRep() const { return rep; }

public:
  struct IDcomparator {
    static int compare(const KNode& left, const KNode& right) {
      return (left.id - right.id);
    }

    // true if left < right
    bool operator()(const KNode& left, const KNode& right) {
      return (compare(left, right) < 0);
    }
  };

  //! a way to order nodes by their id
  //! required for constructing non-repeating edges
  bool operator<(const KNode& that) const {
    return (IDcomparator::compare(*this, that) < 0);
  }

  bool operator==(const KNode& that) const {
    return (IDcomparator::compare(*this, that) == 0);
  }

  bool operator!=(const KNode& that) const {
    return (IDcomparator::compare(*this, that) != 0);
  }
};

struct KNodeAdj : public KNode {
  typedef std::list<KEdge<KNodeAdj>*> EdgeListTy;
  EdgeListTy edges;

public:
  KNodeAdj(unsigned id) : KNode(id){};

  void addEdge(KEdge<KNodeAdj>* const that) {
    assert(that != NULL);
    // assumes there are no duplicates
    assert(std::find(edges.begin(), edges.end(), that) == edges.end() &&
           "Duplicate edge???");

    edges.push_back(that);
  }

  KNodeAdj* getRep() const { return static_cast<KNodeAdj*>(rep); }
};

struct KNodeMin : public KNode {
  galois::GAtomic<const KEdge<KNodeMin>*> minEdge;

public:
  KNodeMin(unsigned id) : KNode(id), minEdge(NULL){};

  bool claimAsMin(const KEdge<KNodeMin>* const edge);

  KNodeMin* getRep() const { return static_cast<KNodeMin*>(rep); }
};

template <typename KNode_tp>
struct KEdge {
  KNode_tp* src;
  KNode_tp* dst;
  unsigned weight;
  bool inMST;

public:
  KEdge(KNode_tp* const src, KNode_tp* const dst, unsigned weight)
      : src(src), dst(dst), weight(weight), inMST(false) {
    assert(src != NULL);
    assert(dst != NULL);

    if (src == dst || (*src) == (*dst)) {
      fprintf(stderr, "Self edges not allowed\n");
      abort();
    }

    // order nodes by id inorder to compare edges of equal weight
    if ((*this->dst) < (*this->src)) {
      std::swap(this->src, this->dst);
    }
  }

  const std::string str() const {
    char s[256];
    sprintf(s, "(%p,%p,%d)", src, dst, weight);
    return s;
  }

  struct NodeIDcomparator {
    static int compare(const KEdge& left, const KEdge& right) {
      int cmp = KNode::IDcomparator::compare(*(left.src), *(right.src));

      if (cmp == 0) {
        cmp = KNode::IDcomparator::compare(*(left.dst), *(right.dst));
      }

      return cmp;
    }

    bool operator()(const KEdge& left, const KEdge& right) const {
      return (compare(left, right) < 0);
    }
  };

  struct PtrComparator {

    static int compare(const KEdge* left, const KEdge* right) {
      assert(left != NULL);
      assert(right != NULL);

      int cmp = (left->weight - right->weight);

      if (cmp == 0) {
        cmp = NodeIDcomparator::compare(*left, *right);
      }

      assert(((left->src == right->src && left->dst == right->dst) ? (cmp == 0)
                                                                   : true) &&
             "duplicate edge with different weights?");

      assert(((*(left->src) == *(right->src) && *(left->dst) == *(right->dst))
                  ? (cmp == 0)
                  : true) &&
             "duplicate edge with different weights");

      return cmp;
    }

    bool operator()(const KEdge* left, const KEdge* right) const {
      return (compare(left, right) < 0);
    }
  };

  bool operator<(const KEdge& that) const {
    return (PtrComparator::compare(this, &that) < 0);
  }

  bool operator==(const KEdge& that) const {
    return (PtrComparator::compare(this, &that) == 0);
  }
};

bool KNodeMin::claimAsMin(const KEdge<KNodeMin>* const edge) {
  assert(edge != NULL);

  bool succ = false;

  // if it's NULL try to set it
  if (minEdge == NULL) {
    succ = minEdge.cas(NULL, edge);
  }

  // by now this thread or some other has set it to non-null value
  assert(minEdge != NULL);

  // keep trying until min achieved
  for (const KEdge<KNodeMin>* curr = minEdge;
       KEdge<KNodeMin>::PtrComparator::compare(curr, edge) > 0;
       curr = minEdge) {

    succ = minEdge.cas(curr, edge);
  }

  return succ;
}

#endif // _KRUSKAL_DATA_H_
