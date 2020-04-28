/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_GRAPHS_SPATIALTREE_H
#define GALOIS_GRAPHS_SPATIALTREE_H

namespace galois {
namespace graphs {

//! Stores sets of objects at specific spatial coordinates in a quad tree.
//! Lookup returns an approximation of the closest item
template <typename T>
class SpatialTree2d {
  struct Box2d {
    double xmin;
    double ymin;
    double xmax;
    double ymax;

    double xmid() const { return (xmin + xmax) / 2.0; }
    double ymid() const { return (ymin + ymax) / 2.0; }

    void decimate(int quad, double midx, double midy) {
      if (quad & 1)
        xmin = midx;
      else
        xmax = midx;
      if (quad & 2)
        ymin = midy;
      else
        ymax = midy;
    }
  };
  struct Node {
    // existing item
    T val;
    double x, y;

    // center
    double midx, midy;

    Node* children[4];
    // needs c++11: Node(const T& v) :val(v), children({0,0,0,0}) {}
    Node(const T& v, double _x, double _y) : val(v), x(_x), y(_y) {
      children[0] = children[1] = children[2] = children[3] = 0;
    }

    void setCenter(double cx, double cy) {
      midx = cx;
      midy = cy;
    }

    int getQuad(double _x, double _y) {
      int retval = 0;
      if (_x > midx)
        retval += 1;
      if (_y > midy)
        retval += 2;
      return retval;
    }
  };

  galois::runtime::FixedSizeAllocator<Node> nodeAlloc;

  Node* root;
  Box2d bounds;

  // true if x,y is closer to testx, testy than oldx, oldy
  bool closer(double x, double y, double testx, double testy, double oldx,
              double oldy) const {
    double doldx  = x - oldx;
    double doldy  = y - oldy;
    double dtestx = x - testx;
    double dtesty = y - testy;
    doldx *= doldx;
    doldy *= doldy;
    dtestx *= dtestx;
    dtesty *= dtesty;
    return (dtestx + dtesty) < (doldx + doldy);
  }

  /*
  T* recfind(Node* n, T* best, double bestx, double besty, double x, double y,
  Box2d b) { if (!n) return best; if (!best) { // || closer(x, y, n->x, n->y,
  bestx, besty)) { best = &n->val; bestx = n->x; besty = n->y;
    }
    int quad = b.getQuad(x,y);
    b.decimate(quad);
    return recfind(n->children[quad], best, bestx, besty, x, y, b);
  }
  */

  T* recfind(Node* n, double x, double y) {
    Node* best = 0;
    while (n) {
      if (!best || closer(x, y, n->x, n->y, best->x, best->y))
        best = n;
      //      best = &n->val;
      int quad = n->getQuad(x, y);
      n        = n->children[quad];
    }
    return &best->val;
  }

  void recinsert(Node** pos, Box2d b, Node* node) {
    if (!*pos) {
      // only do an atomic if it looks empty
      node->setCenter(b.xmid(), b.ymid());
      if (__sync_bool_compare_and_swap(pos, 0, node))
        return; // worked!
    }
    // We should recurse
    int quad = (*pos)->getQuad(node->x, node->y);
    b.decimate(quad, (*pos)->midx, (*pos)->midy);
    recinsert(&(*pos)->children[quad], b, node);
  }

  Node* mkNode(const T& v, double x, double y) {
    Node* n = nodeAlloc.allocate(1);
    nodeAlloc.construct(n, Node(v, x, y));
    return n;
    // return new Node(v,x,y);
  }

  void delNode(Node* n) {
    nodeAlloc.destroy(n);
    nodeAlloc.deallocate(n, 1);
    // delete n;
  }

  void freeTree(Node* n) {
    if (!n)
      return;
    for (int x = 0; x < 4; ++x)
      freeTree(n->children[x]);
    delNode(n);
  }

public:
  SpatialTree2d(double xmin = 0.0, double ymin = 0.0, double xmax = 0.0,
                double ymax = 0.0)
      : root(0) {
    init(xmin, ymin, xmax, ymax);
  }

  ~SpatialTree2d() {
    freeTree(root);
    root = 0;
  }

  void init(double xmin, double ymin, double xmax, double ymax) {
    bounds.xmin = xmin;
    bounds.ymin = ymin;
    bounds.xmax = xmax;
    bounds.ymax = ymax;
  }

  //! Returns null if tree is empty
  T* find(double x, double y) {
    assert(root);
    return recfind(root, x, y);
  }

  //! Insert an element. Will always insert and never roll back and thus must
  //! be used after failsafe point.
  void insert(double x, double y, const T& v) {
    recinsert(&root, bounds, mkNode(v, x, y));
  }
};

} // namespace graphs
} // namespace galois

#endif
