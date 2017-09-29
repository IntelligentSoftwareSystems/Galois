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
#include "galois/Galois.h"
#include "galois/Accumulator.h"
#include <boost/iterator/transform_iterator.hpp>
#include <boost/array.hpp>

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


static const int maxLeafSize = 16;

/**
 * Finds points nearby a given point.
 */
class PQuadTree {
  struct FindResult {
    Point* p;
    double best;
  };

  struct DerefPointer: public std::unary_function<Point*,Point> {
    Point operator()(Point* p) const { return *p; }
  };

  struct Node {
    typedef boost::array<Point*,maxLeafSize> PointsTy;
    Node* child[4];
    PointsTy* points;
    int size;

    //! Make internal node
    explicit Node() {
      memset(child, 0, sizeof(*child) * 4);
      points = NULL;
    }

    //! Make leaf node
    Node(Point* p, PointsTy* ps) {
      memset(child, 0, sizeof(*child) * 4);
      points = ps;
      points->at(0) = p;
      size = 1;
    }
  
    bool isLeaf() const {
      return points != NULL;
    }
  };

  void deleteNode(Node* root) {
    if (root->isLeaf()) {
      pointsAlloc.destroy(root->points);
      pointsAlloc.deallocate(root->points, 1);
    } else {
      for (int i = 0; i < 4; ++i) {
        if (root->child[i])
          deleteNode(root->child[i]);
      }
    }
    nodeAlloc.destroy(root);
    nodeAlloc.deallocate(root, 1);
  }

  Node* newNode() {
    Node* n = nodeAlloc.allocate(1);
    nodeAlloc.construct(n, Node());
    return n;
  }

  Node* newNode(Point *p) {
    Node* n = nodeAlloc.allocate(1);
    Node::PointsTy* ps = pointsAlloc.allocate(1);
    pointsAlloc.construct(ps, Node::PointsTy());
    nodeAlloc.construct(n, Node(p, ps));
    return n;
  }


  //! Provide appropriate initial values for reduction
  template<bool isMax>
  struct MTuple: public Tuple {
    MTuple(): Tuple(isMax ? std::numeric_limits<TupleDataTy>::min() : std::numeric_limits<TupleDataTy>::max()) { }
    MTuple(const Tuple& t): Tuple(t) { }
  };

  template<bool isMax>
  struct MTupleReducer {
    void operator()(MTuple<isMax>& lhs, const MTuple<isMax>& rhs) const {
      for (int i = 0; i < 2; ++i)
        lhs[i] = isMax ? std::max(lhs[i], rhs[i]) : std::min(lhs[i], rhs[i]);
    }
  };

  struct MinBox: public galois::GReducible<MTuple<false>, MTupleReducer<false> > { 
    MinBox() { }
  };

  struct MaxBox: public galois::GReducible<MTuple<true>, MTupleReducer<true> > { 
    MaxBox() { }
  };

  struct ComputeBox {
    MinBox& least;
    MaxBox& most;
    ComputeBox(MinBox& l, MaxBox& m): least(l), most(m) { }
    void operator()(const Point* p) const {
      least.update(p->t());
      most.update(p->t());
    }
  };

  template<typename IterTy>
  struct WorkItem {
    IterTy begin;
    IterTy end;
    Tuple center;
    double radius;
    Node* root;
    PQuadTree* self;

    WorkItem(PQuadTree* s, IterTy b, IterTy e, Node* n, Tuple c, double r): 
      begin(b), end(e), center(c), radius(r), root(n), self(s) { }
    
    void operator()() {
      for (; begin != end; ++begin) {
        self->add(root, *begin, center, radius);
      }
    }
  };

  template<typename IterTy>
  struct PAdd {
    void operator()(WorkItem<IterTy>& w) {
      w();
    }
    void operator()(WorkItem<IterTy>& w, galois::UserContext<WorkItem<IterTy> >&) {
      w();
    }
  };

  struct Split: public std::unary_function<Point*,bool> {
    int index;
    TupleDataTy pivot;
    Split(int i, TupleDataTy p): index(i), pivot(p) { }
    bool operator()(Point* p) {
      return p->t()[index] < pivot;
    }
  };

  Tuple m_center;
  double m_radius;
  Node* m_root;

  galois::FixedSizeAllocator<Node> nodeAlloc;
  galois::FixedSizeAllocator<Node::PointsTy> pointsAlloc;

  template<typename IterTy>
  void init(IterTy begin, IterTy end) {
    MinBox least;
    MaxBox most;
    galois::do_all(galois::iterate(begin, end), ComputeBox(least, most));
    //std::for_each(begin, end, ComputeBox(least, most));

    MTuple<true> mmost = most.reduce();
    MTuple<false> lleast = least.reduce();
    
    m_radius = std::max(mmost.x() - lleast.x(), mmost.y() - lleast.y()) / 2.0;
    
    m_center = lleast;
    m_center.x() += m_radius;
    m_center.y() += m_radius;
  }

  template<typename IterTy,typename OutIterTy>
  void divideWork(IterTy begin, IterTy end, Node* root, Tuple center, double radius, OutIterTy& out, int depth) {
    if (depth == 0 || std::distance(begin, end) <= 16) {
      *out++ = WorkItem<IterTy>(this, begin, end, root, center, radius);
      return;
    }
    
    IterTy its[5];
    its[0] = begin;
    its[4] = end;

    its[2] = std::partition(its[0], its[4], Split(1, center[1]));
    its[1] = std::partition(its[0], its[2], Split(0, center[0]));
    its[3] = std::partition(its[2], its[4], Split(0, center[0]));
    
    radius *= 0.5;
    --depth;

    for (int i = 0; i < 4; ++i) {
      Tuple newC;
      root->child[i] = newNode();
      makeNewCenter(i, center, radius, newC);
      divideWork(its[i], its[i+1], root->child[i], newC, radius, out, depth);
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

  bool find(Node* root, const Point* p, const Tuple& center, double radius, FindResult& result) {
    if (root->isLeaf()) {
      bool retval = false;
      const Tuple& t0 = p->t();
      for (int i = 0; i < root->size; ++i) {
        const Point* o = root->points->at(i);
        if (!o->inMesh())
          continue;

        double d = 0;
        const Tuple& t1 = o->t();
        for (int j = 0; j < t0.dim(); ++j) {
          double v = t0[j] - t1[j];
          d += v * v;
        }
        if (result.p == NULL || d < result.best) {
          result.p = root->points->at(i);
          result.best = d;
          retval = true;
        }
      }
      return retval;
    }

    // Search, starting at closest quadrant to p
    radius *= 0.5;
    int start = getIndex(center, p->t());
    for (int i = 0; i < 4; ++i) {
      int index = (start + i) % 4;
      Node* kid = root->child[index];
      if (kid != NULL) {
        Tuple newCenter;
        makeNewCenter(index, center, radius, newCenter);
        if (couldBeCloser(p, newCenter, radius, result)) {
          if (false) {
            // exhaustive
            find(kid, p, newCenter, radius, result);
          } else {
            // return only first
            if (find(kid, p, newCenter, radius, result))
              return true;
          }
        }
      }
    }
    return false;
  }

  void makeInternal(Node* root, const Tuple& center, double radius) {
    assert(root->isLeaf());

    Node::PointsTy* points = root->points;
    root->points = NULL;

    for (Node::PointsTy::iterator ii = points->begin(), ei = points->begin() + root->size;
        ii != ei; ++ii) {
      add(root, *ii, center, radius);
    } 
    pointsAlloc.destroy(points);
    pointsAlloc.deallocate(points, 1);
  }
  
  void add(Node* root, Point* p, const Tuple& center, double radius) {
    if (root->isLeaf()) {
      if (root->size < maxLeafSize) {
        root->points->at(root->size++) = p;
      } else {
        makeInternal(root, center, radius);
        add(root, p, center, radius);
      }
      return;
    }

    int index = getIndex(center, p->t());
    Node*& kid = root->child[index];
    if (kid == NULL) {
      kid = newNode(p);
    } else {
      radius *= 0.5;
      assert(radius != 0.0);
      Tuple newCenter;
      makeNewCenter(index, center, radius, newCenter);
      add(kid, p, newCenter, radius);
    }
  }

  template<typename OutputTy>
  void output(Node* root, OutputTy out) {
    if (root->isLeaf()) {
      std::copy(
          boost::make_transform_iterator(root->points->begin(), DerefPointer()),
          boost::make_transform_iterator(root->points->begin() + root->size, DerefPointer()),
          out);
    } else {
      for (int i = 0; i < 4; ++i) {
        Node* kid = root->child[i];
        if (kid != NULL)
          output(kid, out);
      }
    }
  }

public:
  template<typename IterTy>
  PQuadTree(IterTy begin, IterTy end) { 
    m_root = newNode();

    init(begin, end);

    typedef std::vector<Point*> PointsBufTy;
    typedef WorkItem<PointsBufTy::iterator> WIT;
    typedef std::vector<WIT> WorkTy;
    typedef galois::worklists::dChunkedLIFO<1> WL;
    PointsBufTy points;
    std::copy(begin, end, std::back_inserter(points));

    WorkTy work;
    std::back_insert_iterator<WorkTy> it(work);
    divideWork(points.begin(), points.end(), m_root, m_center, m_radius, it, 4);
    galois::for_each(galois::iterate(work), PAdd<PointsBufTy::iterator>(), galois::wl<WL>());
  }

  ~PQuadTree() {
    deleteNode(m_root);
  }

  template<typename OutputTy>
  void output(OutputTy out) {
    if (m_root != NULL) {
      output(m_root, out);
    }
  }

  //! Find point nearby to p
  bool find(const Point* p, Point*& result) {
    FindResult r;
    r.p = NULL;
    if (m_root) {
      find(m_root, p, m_center, m_radius, r);
      if (r.p != NULL) {
        result = r.p;
        return true;
      }
    }
    return false;
  }
};

/**
 * Finds points nearby a given point.
 */
class SQuadTree {
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
          const Point* o = points[i];
          if (!o->inMesh())
            continue;
          const Tuple& t1 = o->t();
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
  SQuadTree(Begin begin, End end): root(NULL) { 
    init(begin, end);
    for (; begin != end; ++begin)
      add(*begin);
  }

  ~SQuadTree() {
    deleteNode(root);
  }

  //! Find point nearby to p
  bool find(const Point* p, Point*& result) {
    FindResult r;
    r.p = NULL;
    if (root) {
      root->find(p, center, radius, r);
      if (r.p != NULL) {
        result = r.p;
        return true;
      }
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

typedef PQuadTree QuadTree;

#endif
