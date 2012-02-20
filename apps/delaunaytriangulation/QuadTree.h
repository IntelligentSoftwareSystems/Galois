/** A quad-tree -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef QUADTREE_H
#define QUADTREE_H

#include "Point.h"

#include <boost/iterator/transform_iterator.hpp>

#include <limits>

inline int getIndex(const Tuple& a, const Tuple& b) {
  int index = 0;
  for (int i = 0; i < 2; ++i) {
    if (a[i] < b[i]) {
      index += 1 << i;
    }
  }
  return index;
}

inline void makeNewCenter(int index, const Tuple& center, double radius, Tuple& newCenter) {
  newCenter = center;
  for (int i = 0; i < 2; ++i) {
    newCenter[i] += (index & (1 << i)) > 0 ? radius : -radius;
  }
}


static const int maxLeafSize = 128;

/**
 * Finds points nearby a given point.
 */
class QuadTree {
  struct FindResult {
    Point* p;
    double best;
  };

  struct DerefPointer: public std::unary_function<Point*,Point> {
    Point operator()(Point* p) const { return *p; }
  };

  struct Node {
    Node* child[4];
    Point** points;
    int size;

    bool isLeaf() const {
      return points != NULL;
    }

    void makeInternal(const Tuple& center, double radius) {
      memset(child, 0, sizeof(*child) * 4);
      Point** begin = points;
      points = NULL;

      for (Point **p = begin, **end = begin + size; p != end; ++p) {
        add(*p, center, radius);
      } 
      delete [] begin;
    }

    void add(Point* p, const Tuple& center, double radius) {
      if (isLeaf()) {
        if (size < maxLeafSize) {
          points[size] = p;
          ++size;
        } else {
          makeInternal(center, radius);
          add(p, center, radius);
        }
        return;
      }

      int index = getIndex(center, p->t());
      Node*& kid = child[index];
      if (kid == NULL) {
        kid = new Node();
        kid->points = new Point*[maxLeafSize];
        kid->points[0] = p;
        kid->size = 1;
      } else {
        radius *= 0.5;
        assert(radius != 0.0);
        Tuple newCenter;
        makeNewCenter(index, center, radius, newCenter);
        kid->add(p, newCenter, radius);
      }
    }

    bool couldBeCloser(const Point* p, const Tuple& center, double radius, FindResult& result) {
      if (result.p == NULL)
        return true;

      const Tuple& t = p->t();
      double d = 0;
      for (int i = 0; i < t.dim(); ++i) {
        double min = center[i] - radius - t[i];
        double max = center[i] + radius - t[i];
        d += std::min(min*min, max*max);
      }
      return d < result.best;
    }

    void find(const Point* p, const Tuple& center, double radius, FindResult& result) {
      if (isLeaf()) {
        const Tuple& t0 = p->t();
        for (int i = 0; i < size; ++i) {
          double d = 0;
          const Tuple& t1 = points[i]->t();
          for (int j = 0; j < t0.dim(); ++j) {
            double v = t0[j] - t1[j];
            d += v * v;
          }
          if (result.p == NULL || d < result.best) {
            result.p = points[i];
            result.best = d;
          }
        }
        return;
      }

      // Search, starting at closest quadrant to p
      radius *= 0.5;
      int start = getIndex(center, p->t());
      for (int i = 0; i < 4; ++i) {
        int index = (start + i) % 4;
        Node* kid = child[index];
        if (kid != NULL) {
          Tuple newCenter;
          makeNewCenter(index, center, radius, newCenter);
          if (kid->couldBeCloser(p, newCenter, radius, result))
            kid->find(p, newCenter, radius, result);
        }
      }
    }

    template<typename OutputTy>
    void output(OutputTy out) {
      if (isLeaf()) {
        std::copy(
            boost::make_transform_iterator(points, DerefPointer()),
            boost::make_transform_iterator(points + size, DerefPointer()),
            out);
      } else {
        for (int i = 0; i < 4; ++i) {
          Node* kid = child[i];
          if (kid != NULL)
            kid->output(out);
        }
      }
    }
  };

  void deleteNode(Node*& n) {
    if (n == NULL)
      return;
    if (n->isLeaf()) {
      delete [] n->points;
      n->points = NULL;
    } else {
      for (int i = 0; i < 4; ++i) {
        deleteNode(n->child[i]);
      }
    }

    delete n;
    n = NULL;
  }
  
  template<typename Begin,typename End>
  void computeBox(Begin begin, End end, Tuple& least, Tuple& most) {
    least.x() = least.y() = std::numeric_limits<double>::max();
    most.x() = most.y() = std::numeric_limits<double>::min();

    for (; begin != end; ++begin) {
      const Tuple& p = (*begin)->t();
      for (int i = 0; i < 2; ++i) {
        if (p[i] < least[i]) 
          least[i] = p[i];
        
        if (p[i] > most[i]) 
          most[i] = p[i];
      }
    }
  }

  template<typename Begin,typename End>
  void init(Begin begin, End end) {
    Tuple least, most;
    computeBox(begin, end, least, most);

    radius = std::max(most.x() - least.x(), most.y() - least.y()) / 2.0;
    center = least;
    center.x() += radius;
    center.y() += radius;
  }

  void add(Point* p) {
    if (root == NULL) {
      root = new Node();
      root->points = NULL;
      memset(root->child, 0, sizeof(*root->child) * 4);
    }
    root->add(p, center, radius);
  }

  Tuple center;
  double radius;
  Node* root;

public:
  template<typename Begin,typename End>
  QuadTree(Begin begin, End end): root(NULL) { 
    init(begin, end);
    for (; begin != end; ++begin)
      add(*begin);
  }

  ~QuadTree() {
    deleteNode(root);
  }

  //! Find point nearby to p
  bool find(const Point* p, Point*& result) {
    FindResult r;
    r.p = NULL;
    root->find(p, center, radius, r);
    if (r.p != NULL) {
      result = r.p;
      return true;
    }
    return false;
  }

  template<typename OutputTy>
  void output(OutputTy out) {
    if (root != NULL) {
      root->output(out);
    }
  }
};

#endif
