#ifndef BH_OCTREE_H
#define BH_OCTREE_H

#include "Point.h"
#include "BoundingBox.h"

namespace bh {



struct SerialNodeBase {
protected:
  void setChild (unsigned, SerialNodeBase*, SerialNodeBase*) {}
};

// Note: during tree construction, setChild should be called at most twice;
// once when child is changed from null to some leaf node
// second when child is changed from leaf node to internal node (in order to
// expand) 
// We want to record the 2nd change. Reason being that the active elements in
// summarization are only the internal nodes and not the leaves. 
// 

struct SpecNodeBase: public Galois::Runtime::Lockable {
  unsigned level;

  SpecNodeBase (): level (0) {}
  
protected:
  void setChild (unsigned index, SpecNodeBase* c, SpecNodeBase* prev) {
    c->level = this->level + 1;
  }
};

struct LevelNodeBase {
  unsigned level;

  LevelNodeBase (): level (0) {}

protected:
  void setChild (unsigned index, LevelNodeBase* c, LevelNodeBase* prev) {
    c->level = this->level + 1;
  }
};

template <typename T>
struct LevelComparator {

  bool operator () (const T* left, const T* right) const {
    if (left->level == right->level) {
      return (left > right); // pointer comparison, assumes all pointers on heap
    }

    return (left->level > right->level);

  }
};

struct KDGNodeBase {

  // typedef Galois::GAtomic<unsigned> UnsignedAtomic;

  // KDGNodeBase* parent;
  // GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE std::atomic<unsigned> numChild;
  std::atomic<unsigned> numChild;
  KDGNodeBase* parent;
  
  KDGNodeBase (): numChild (0), parent (nullptr) {}

protected:
  void setChild (unsigned index, KDGNodeBase* c, KDGNodeBase* prev) {
    ++numChild;
    c->parent = this;
  }

};


/**
 * A node in an octree is either an internal node or a body (leaf).
 */

template <typename B>
struct Octree: public B {
  Point pos;
  double mass;

  Octree (): B (), pos (), mass (0.0) {}

  explicit Octree (Point _pos): B (), pos (_pos), mass (0.0) {}

  // virtual ~Octree() { }
  virtual bool isLeaf() const = 0;

  virtual ~Octree (void) {}
};

template <typename B>
class OctreeInternal: public Octree<B> {
  
  Octree<B>* child[8];

public:
  OctreeInternal (Point _pos) : Octree<B> (_pos) {
    bzero(child, sizeof(*child) * 8);
  }

  void setChild (unsigned index, Octree<B>* c) {
    assert (index < 8);

    Octree<B>* prev = child[index];
    child[index] = c;

    if (!c->isLeaf ()) {
      B::setChild (index, c, prev);
    }
  }

  bool casChild (unsigned index, Octree<B>* oldVal, Octree<B>* newVal) { 
    assert (index < 8);
    if( __sync_bool_compare_and_swap (&child[index], oldVal, newVal)) {
      if (!newVal->isLeaf ()) {
        B::setChild (index, newVal, oldVal);
      }
      return true;
    }
    return false;
  }

  Octree<B>* getChild (unsigned index) {
    assert (index < 8);
    return child[index];
  }

  const Octree<B>* getChild (unsigned index) const {
    assert (index < 8);
    return child[index];
  }

  // Reorganize leaves to be denser up front 
  // must be invoked after complete tree has been created
  void compactChildren () {
    unsigned index = 0;

    for (unsigned j = 0; j < 8; ++j) {
      if (child[j] == NULL) {
        continue;
      }

      if (index != j) {
        child[index] = child[j];
        child[j] = NULL;
      }

      ++index;
    }

    // alt impl.
    // if (child[j] != NULL) {
      // std::swap (child[j], child[index]);
      // ++index;
    // }
  }

  // virtual ~OctreeInternal() {
    // for (int i = 0; i < 8; i++) {
      // if (child[i] != NULL && !child[i]->isLeaf()) {
        // delete child[i];
      // }
    // }
  // }
  virtual bool isLeaf() const {
    return false;
  }
};

template <typename B>
struct Body: public Octree<B>  {
  Point vel;
  Point acc;

  Body(): Octree<B> () {}

  virtual bool isLeaf() const {
    return true;
  }

  friend std::ostream& operator<<(std::ostream& os, const Body<B>& b) {
    os << "(pos:" << b.pos
       << " vel:" << b.vel
       << " acc:" << b.acc
       << " mass:" << b.mass << ")";
    return os;
  }
};


inline unsigned  getIndex(const Point& a, const Point& b) {
  unsigned  index = 0;
  if (a.x < b.x)
    index += 1;
  if (a.y < b.y)
    index += 2;
  if (a.z < b.z)
    index += 4;
  return index;
}

inline void updateCenter(Point& p, int index, double radius) {
  for (int i = 0; i < 3; i++) {
    double v = (index & (1 << i)) > 0 ? radius : -radius;
    p[i] += v;
  }
}

template<typename B>
GALOIS_ATTRIBUTE_PROF_NOINLINE static void summarizeNode (OctreeInternal<B>* node) {

  assert ((node != NULL) && (!node->isLeaf ()));

  double massSum = 0.0;
  Point accum;

  node->compactChildren ();

  for (unsigned i = 0; i < 8; ++i) {
    Octree<B>* child = node->getChild (i);

    if (child == NULL) {
      break;
    }

    massSum += child->mass;

    for (unsigned j = 0; j < 3; ++j) {
      accum [j] += child->mass * child->pos[j];
    }

  } // end for child

  node->mass = massSum;

  if (massSum > 0.0) {
    double invSum = 1.0 / massSum;

    for (unsigned j = 0; j < 3; ++j) {
      node->pos [j] = accum [j] * invSum;
    }
  }

}


template <typename B1, typename B2>
void compareTrees (const OctreeInternal<B1>* ref, const OctreeInternal<B2>* obs) {
  assert (!ref->isLeaf ());
  assert (!obs->isLeaf ());

  if (!checkRelativeError (ref->pos, obs->pos)) {
    GALOIS_DIE ("different value for position ", ref->pos, " ", obs->pos);
  }

  if (!checkRelativeError (ref->mass, obs->mass)) {
    GALOIS_DIE ("different value for mass ", ref->mass, " ", obs->mass);
  }

  for (unsigned i = 0; i < 8; ++i) {
    if (ref->getChild (i) != nullptr) {
      if (obs->getChild (i) == nullptr) { 
        GALOIS_DIE ("child mismatch");
      }

      if (!ref->getChild (i)->isLeaf ()) {
        if (obs->getChild (i)->isLeaf ()) {
          GALOIS_DIE ("child mismatch");
        }

        compareTrees (
            static_cast<const OctreeInternal<B1>*> (ref->getChild(i)), 
            static_cast<const OctreeInternal<B2>*> (obs->getChild(i)));
      }
    }
  }
}

#if 0 // disabling for now
template <typename B>
struct BuildOctreeTopDown {

  typedef Galois::gdeque<B*,4096> PartList;
  typedef Galois::Runtime::MM::FSBGaloisAllocator<PartList> PartListAlloc;

  template <typename I>
  struct WorkItem {
    OctreeInternal<B>* node;
    BoundingBox box;
    I beg;
    I end;
  };

  BoundingBox box;

  BuildOctreeTopDown (const BoundingBox& _box): box (_box) {
  }

  template <typename WL>
  void operator () (WorkItem& r, WL& wl) {

    WorkItem child[8];

    I next = r.beg;
    ++next;

    assert (r.beg != r.end);
    assert (next != r.end);

    Point center = r.box.center ();

    partitionMultiPass (r, center, child);

    for (unsigned i = 0; i < 8; ++i) {
      if (distOne (child[i].beg, child[i].end)) {

        r.node->setChild (i, *(child[i].beg));

      } else {
        if (child[i].beg != child[i].end) {
          wl.push (child[i]);
        }
      }
    }
  }

  void partitionSinglePass (const WorkItem& r, const Point& center, WorkItem* child) {

    for (unsigned i = 0; i < 8; ++i) {
      PartList* list = partListAlloc.allocate (1);
      new (list) PartList ();
      child[i].partList = list;
    }

    for (I i = r.beg; i != r.end; ++i) {
      unsigned index = getIndex (*i, center);
      child[index].partList->push_back (*i);
    }

    for (unsigned i = 0; i < 8; ++i) {
      child[i].beg = child[i].partList->begin ();
      child[i].end = child[i].partList->end ();
    }

    // delete the old list
    partListAlloc.destroy (r.partList);
    partList.deallocate (r.partList, 1);

  }

  template <typename F0, typename F1, typename F2>
  void partitionMultiPassImpl (WorkItem& r, const Point& center, WorkItem* child, const F0& f0, const F1& f1, const F2& f2) {

    typedef std::pair<typename WorkItem::I> Range;

    Range x[2];
    auto m_x = std::paritition (r.beg, r.end, f0);

    x[0] = Range (r.beg, m_x);
    x[1] = Range (m_x, r.end);

    Range y[4];

    Range z[8];

    for (unsigned i = 0; i < 2; ++i) {
      auto m_y = std::paritition (x[i].first, x[i].second, f1);

      y[i*2 + 0] = Range (x[i].first, m_y);
      y[i*2 + 1] = Range (m_y, x[i].second);

      for (unsigned j = 0; j < 2; ++j) {

        Range curr_y = y[i*2 + j];
        auto m_z = std::paritition (curr_y.first, curr_y.second, f2);

        z[i*4 + j*2 + 0] = Range (curr_y.first, m_z);
        z[i*4 + j*2 + 1] = Range (m_z, curr_y.second);
      }
    }


    for (unsigned c = 0; c < 8; ++c) {
      child[c]  = WorkItem (z[i].first, z[i].second);
    }

  }

};

#endif

// template <typename B>
// void copyToVecInterNodes (OctreeInternal<B>* root, std::vector<OctreeInternal<B>*>& vec) {
  // vec.clear ();
// 
  // vec.push_back (root);
// 
  // for (size_t i = 0; i < vec.size (); ++i) {
// 
    // if (!vec[i]->isLeaf ()) {
      // OctreeInternal<B>* node = static_cast<OctreeInternal<B>*> (vec[i]);
// 
      // for (unsigned i = 0; i < 8; ++i) {
        // Octree<B>* c = node->getChild (i);
        // if (c != NULL && !c->isLeaf ()) {
          // vec.push_back (static_cast<OctreeInternal<B>*> (c));
        // }
      // }
    // }
  // }
// }

} // end namespace bh

#endif //  BH_OCTREE_H
