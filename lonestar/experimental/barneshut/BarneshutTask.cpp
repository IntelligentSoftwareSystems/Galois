/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/gdeque.h"
#include "galois/Bag.h" // XXX
#include "galois/runtime/TaskWork.h"
#include "galois/worklists/WorkListAlt.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/math/constants/constants.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <limits>
#include <iostream>
#include <array>
#include <deque>
#include <list>
#include <strings.h>

const char* name = "Barnshut N-Body Simulator";
const char* desc =
  "Simulation of the gravitational forces in a galactic cluster using the "
  "Barnes-Hut n-body algorithm\n";
const char* url = "barneshut";

enum Algo {
  algoBase,
  algoBlocked,
  algoBlockedC,
  algoOrig,
  algoIndirect,
  algoUnroll,
  algoTree,
  algoTaskOrig,
  algoTaskBlocked
};

namespace cll = llvm::cl;

static cll::opt<int> nbodies("n", cll::desc("Number of bodies"), cll::init(10000));
static cll::opt<int> ntimesteps("steps", cll::desc("Number of steps"), cll::init(1));
static cll::opt<int> seed("seed", cll::desc("Random seed"), cll::init(7));
static cll::opt<Algo> algo(cll::desc("Algorithm:"),
    cll::values(
      clEnumVal(algoBase, "AlgoBase"),
      clEnumVal(algoOrig, "AlgoOrig"),
      clEnumVal(algoIndirect, "AlgoIndirect"),
      clEnumVal(algoUnroll, "AlgoUnroll"),
      clEnumVal(algoTree, "AlgoTree"),
      clEnumVal(algoBlocked, "AlgoBlocked"),
      clEnumVal(algoBlockedC, "AlgoBlockedContinuation"),
      clEnumVal(algoTaskOrig, "AlgoTaskOrig"),
      clEnumVal(algoTaskBlocked, "AlgoTaskBlocked"),
      clEnumValEnd), cll::init(algoOrig));
static cll::opt<int> blockSizeX("blockSizeX", cll::desc("Block size for Blocked algos"), cll::init(8));
static cll::opt<int> blockSizeY("blockSizeY", cll::desc("Block size for Blocked algos"), cll::init(8));
static cll::opt<bool> useTaskRefine("useTaskRefine", cll::desc("Refine task schedule"), cll::init(false));
static cll::opt<bool> blockFIFO("blockFIFO", cll::desc("Use FIFO for Blocked algos"), cll::init(false));
static cll::opt<bool> allocateFIFO("allocateFIFO", cll::desc("Use FIFO allocation of tree"), cll::init(false));

//XXX(ddn): Change to tbb
#ifdef GALOIS_USE_CILK
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_scheduler_init.h"
static cll::opt<bool> useCilkTreeBuild("useTBBTreeBuild", cll::desc("Use Cilk scheduler for tree build"), cll::init(false));
struct FixThreads: public tbb::task_scheduler_observer {
  FixThreads() {
    observe(true);
  }
  virtual void on_scheduler_entry(bool is_worker) {
    galois::runtime::LL::initTID();
    unsigned id = galois::runtime::LL::getTID();
    galois::runtime::initPTS();
    galois::runtime::LL::bindThreadToProcessor(id);
  }
};
#else
static bool useCilkTreeBuild;
struct FixThreads { };
#define cilk_spawn
#define cilk_sync
#endif

static const int MaxBlockSizeX = 32;

struct Point {
  double x, y, z;
  Point() : x(0.0), y(0.0), z(0.0) { }
  Point(double _x, double _y, double _z) : x(_x), y(_y), z(_z) { }
  explicit Point(double v) : x(v), y(v), z(v) { }

  double operator[](const int index) const {
    switch (index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
    }
    assert(false && "index out of bounds");
    abort();
  }

  double& operator[](const int index) {
    switch (index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
    }
    assert(false && "index out of bounds");
    abort();
  }

  bool operator==(const Point& other) {
    if (x == other.x && y == other.y && z == other.z)
      return true;
    return false;
  }

  bool operator!=(const Point& other) {
    return !operator==(other);
  }

  Point& operator+=(const Point& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  Point& operator*=(double value) {
    x *= value;
    y *= value;
    z *= value;
    return *this;
  }

  double dist2() {
    return x * x + y * y + z * z;
  }

};

std::ostream& operator<<(std::ostream& os, const Point& p) {
  os << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
  return os;
}

/**
 * A node in an octree is either an internal node or a body (leaf).
 */
struct Octree {
  Point pos;
  double mass;
  Octree* parent;

  Octree(const Point& p, double m, Octree* parent): pos(p), mass(m), parent(parent) { }
  Octree() { }

  virtual ~Octree() { }
  virtual bool isLeaf() const = 0;
};

struct OctreeInternal: public Octree {
  // Extra space for end of array element needed by 
  // ComputeForceBlockedContinuation
  Octree* child[8+1];

  OctreeInternal(const Point& p, Octree* parent): Octree(p, 0.0, parent) {
    bzero(child, sizeof(*child) * (8+1));
  }
  
//  virtual ~OctreeInternal() {
//    for (int i = 0; i < 8; i++) {
//      if (child[i] && !child[i]->isLeaf()) {
//        delete child[i]; // XXX not needed with heap
//      }
//    }
//  }
  
  virtual bool isLeaf() const {
    return false;
  }
};

struct Body: public Octree, public galois::runtime::Lockable {
  Point vel;
  Point acc;
  Point oldacc;

  Body() { }

  virtual bool isLeaf() const { return true; }
};

std::ostream& operator<<(std::ostream& os, const Body& b) {
  os << "(pos:" << b.pos
     << " vel:" << b.vel
     << " acc:" << b.acc
     << " mass:" << b.mass << ")";
  return os;
}

struct BoundingBox {
  Point min;
  Point max;
  explicit BoundingBox(const Point& p) : min(p), max(p) { }
  BoundingBox() :
    min(std::numeric_limits<double>::max()),
    max(std::numeric_limits<double>::min()) { }

  void merge(const BoundingBox& other) {
    for (int i = 0; i < 3; i++) {
      if (other.min[i] < min[i])
        min[i] = other.min[i];
    }
    for (int i = 0; i < 3; i++) {
      if (other.max[i] > max[i])
        max[i] = other.max[i];
    }
  }

  void merge(const Point& other) {
    for (int i = 0; i < 3; i++) {
      if (other[i] < min[i])
        min[i] = other[i];
    }
    for (int i = 0; i < 3; i++) {
      if (other[i] > max[i])
        max[i] = other[i];
    }
  }

  double diameter() const {
    double diameter = max.x - min.x;
    for (int i = 1; i < 3; i++) {
      double t = max[i] - min[i];
      if (diameter < t)
        diameter = t;
    }
    return diameter;
  }

  double radius() const {
    return diameter() / 2;
  }

  Point center() const {
    return Point(
        (max.x + min.x) * 0.5,
        (max.y + min.y) * 0.5,
        (max.z + min.z) * 0.5);
  }
};

std::ostream& operator<<(std::ostream& os, const BoundingBox& b) {
  os << "(min:" << b.min << " max:" << b.max << ")";
  return os;
}

struct Config {
  const double dtime; // length of one time step
  const double eps; // potential softening parameter
  const double tol; // tolerance for stopping recursion, <0.57 to bound error
  const double dthf, epssq, itolsq;

  Config() :
    dtime(0.5),
    eps(0.05),
    tol(0.025),
    dthf(dtime * 0.5),
    epssq(eps * eps),
    itolsq(1.0 / (tol * tol))  { }
};

Config config;

inline int getIndex(const Point& a, const Point& b) {
  int index = 0;
  if (a.x < b.x) index += 1;
  if (a.y < b.y) index += 2;
  if (a.z < b.z) index += 4;
  return index;
}

inline void updateCenter(Point& p, int index, double radius) {
  for (int i = 0; i < 3; i++) {
    double v = (index & (1 << i)) > 0 ? radius : -radius;
    p[i] += v;
  }
}

// XXX numa vector
typedef galois::gdeque<Body> Bodies;
typedef galois::gdeque<Body*> BodyPtrs;

struct BuildOctreeCilk {
  typedef galois::FixedSizeAllocator<BodyPtrs> BodyPtrsAlloc;
  typedef galois::FixedSizeAllocator<OctreeInternal> OctreeInternalAlloc;

  BodyPtrsAlloc& bodyPtrsAlloc;
  OctreeInternalAlloc& octreeInternalAlloc;

  BuildOctreeCilk(BodyPtrsAlloc& a1, OctreeInternalAlloc& a2):
    bodyPtrsAlloc(a1),
    octreeInternalAlloc(a2) { }

  void computeCenterOfMass(OctreeInternal* node) {
    Point accum;
    double mass = 0.0;

    for (int i = 0; i < 8; i++) {
      Octree* child = node->child[i];
      
      if (child == NULL)
        break;

      double m;
      const Point* p;
      if (child->isLeaf()) {
        Body* n = static_cast<Body*>(child);
        m = n->mass;
        p = &n->pos;
      } else {
        OctreeInternal* n = static_cast<OctreeInternal*>(child);
        m = n->mass;
        p = &n->pos;
      }

      mass += m;
      for (int j = 0; j < 3; j++) 
        accum[j] += (*p)[j] * m;
    }

    node->mass = mass;
    
    if (mass > 0.0) {
      double inv_mass = 1.0 / mass;
      for (int j = 0; j < 3; j++)
        node->pos[j] = accum[j] * inv_mass;
    }
  }

  void buildOctree(BodyPtrs::iterator bb, BodyPtrs::iterator eb, 
      OctreeInternal* node, double radius) {

    typedef std::array<int,8> Counts;
    typedef std::array<BodyPtrs*,8> Bags;

    Bags bags;
    Counts counts;

    for (int i = 0; i < 8; ++i) {
      bags[i] = bodyPtrsAlloc.allocate(1);
      bodyPtrsAlloc.construct(bags[i]);
      //bags[i] = new BodyPtrs(); // XXX
      counts[i] = 0;
    }

    for (; bb != eb; ++bb) {
      int i = getIndex(node->pos, (*bb)->pos);
      bags[i]->push_back(*bb);
      ++counts[i];
    }

    radius *= 0.5;

    for (int i = 0; i < 8; ++i) {
      if (counts[i] == 0)
        continue;
      if (counts[i] == 1) {
        node->child[i] = *bags[i]->begin();
        node->child[i]->parent = node;
        continue;
      }

      Point new_pos(node->pos);
      updateCenter(new_pos, i, radius);
      OctreeInternal* new_node = octreeInternalAlloc.allocate(1);
      octreeInternalAlloc.construct(new_node, new_pos, node);
      //OctreeInternal* new_node = new OctreeInternal(new_pos, node); // XXX
      node->child[i] = new_node;

      cilk_spawn buildOctree(bags[i]->begin(), bags[i]->end(), new_node, radius);
    }
    cilk_sync;

    // Reorganize leaves to be denser up front 
    int index = 0;
    for (int i = 0; i < 8; i++) {
      Octree* child = node->child[i];
      if (child == NULL)
        continue;

      if (index != i) {
        node->child[index] = child;
        node->child[i] = NULL;
      }
      index++;
    }

    computeCenterOfMass(node);
  }
};



struct ComputeCenterOfMass {
  OctreeInternal* node;

  ComputeCenterOfMass(OctreeInternal* n): node(n) { }

// void inspect() { } // XXX

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>&) {
    Point accum;
    double mass = 0.0;

    for (int i = 0; i < 8; i++) {
      Octree* child = node->child[i];
      
      if (child == NULL)
        break;

      double m;
      const Point* p;
      if (child->isLeaf()) {
        Body* n = static_cast<Body*>(child);
        m = n->mass;
        p = &n->pos;
      } else {
        OctreeInternal* n = static_cast<OctreeInternal*>(child);
        m = n->mass;
        p = &n->pos;
      }

      mass += m;
      for (int j = 0; j < 3; j++) 
        accum[j] += (*p)[j] * m;
    }

    node->mass = mass;
    
    if (mass > 0.0) {
      double inv_mass = 1.0 / mass;
      for (int j = 0; j < 3; j++)
        node->pos[j] = accum[j] * inv_mass;
    }
  }
};

struct BuildOctree {
  typedef galois::FixedSizeAllocator<BodyPtrs> BodyPtrsAlloc;
  typedef galois::FixedSizeAllocator<OctreeInternal> OctreeInternalAlloc;

  BodyPtrsAlloc& bodyPtrsAlloc;
  OctreeInternalAlloc& octreeInternalAlloc;
  BodyPtrs::iterator bb;
  BodyPtrs::iterator eb;
  OctreeInternal* node;
  double radius;
  galois::Task parent;

  BuildOctree(BodyPtrsAlloc& a1, OctreeInternalAlloc& a2,
      BodyPtrs::iterator _bb, BodyPtrs::iterator _eb, 
      OctreeInternal* n, double r,
      galois::Task p):
    bodyPtrsAlloc(a1),
    octreeInternalAlloc(a2),
    bb(_bb), eb(_eb),
    node(n), radius(r), parent(p) { }

//  void inspect() { } // XXX

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    typedef std::array<int,8> Counts;
    typedef std::array<BodyPtrs*,8> Bags;

    Bags bags;
    Counts counts;

    for (int i = 0; i < 8; ++i) {
      bags[i] = bodyPtrsAlloc.allocate(1);
      bodyPtrsAlloc.construct(bags[i]);
      counts[i] = 0;
    }

    for (; bb != eb; ++bb) {
      int i = getIndex(node->pos, (*bb)->pos);
      bags[i]->push_back(*bb);
      ++counts[i];
    }

    radius *= 0.5;

    galois::Task after = ctx.addTask2(ComputeCenterOfMass(node));
    if (parent)
      ctx.addDependence(after, parent);

    for (int i = 0; i < 8; ++i) {
      if (counts[i] == 0)
        continue;
      if (counts[i] == 1) {
        node->child[i] = *bags[i]->begin();
        node->child[i]->parent = node;
        continue;
      }

      Point new_pos(node->pos);
      updateCenter(new_pos, i, radius);
      OctreeInternal* new_node = octreeInternalAlloc.allocate(1);
      octreeInternalAlloc.construct(new_node, new_pos, node);
      node->child[i] = new_node;

      galois::Task t = ctx.addTask1(BuildOctree(bodyPtrsAlloc, octreeInternalAlloc,
            bags[i]->begin(), bags[i]->end(), new_node, radius, after));
      ctx.addDependence(t, after);
      ctx.markDeferred(after);
    }

    // Reorganize leaves to be denser up front 
    int index = 0;
    for (int i = 0; i < 8; i++) {
      Octree* child = node->child[i];
      if (child == NULL)
        continue;

      if (index != i) {
        node->child[index] = child;
        node->child[i] = NULL;
      }
      index++;
    }
  }
};

template<typename OutTy>
struct Sort {
  OctreeInternal* node;
  OutTy& out;

  Sort(OctreeInternal* n, OutTy& o): node(n), out(o) { }

//  void inspect() { } // XXX

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    for (int i = 0; i < 8; i++) {
      Octree* child = node->child[i];
      if (child == NULL)
        break;

      if (child->isLeaf()) {
        out.push_back(*static_cast<Body*>(child));
        continue;
      }
      ctx.addTask1(Sort<OutTy>(static_cast<OctreeInternal*>(child), out));
    }
  }

  void operator()(OctreeInternal* n) {
    for (int i = 0; i < 8; i++) {
      Octree* child = n->child[i];
      if (child == NULL)
        break;

      if (child->isLeaf()) {
        out.push_back(*static_cast<Body*>(child));
        continue;
      }
      (*this)(static_cast<OctreeInternal*>(child));
    }
  }
};

void updateForce(Point& acc, const Point& delta, double psq, double mass) {
  // Computing force += delta * mass * (|delta|^2 + eps^2)^{-3/2}
  psq += config.epssq;
  double idr = 1 / sqrt((float) psq);
  double scale = mass * idr * idr * idr;

  for (int i = 0; i < 3; i++)
    acc[i] += delta[i] * scale;
}

void computeDelta(Point& p, const Body* body, const Octree* b) {
  for (int i = 0; i < 3; i++)
    p[i] = b->pos[i] - body->pos[i];
}

struct ComputeForceIndirect {
  Body* body;
  OctreeInternal* node;
  double dsq;

  ComputeForceIndirect(Body* b, OctreeInternal* n, double d): body(b), node(n), dsq(d) { }
  
//  void inspect() { } // XXX

  struct Frame {
    double dsq;
    OctreeInternal* node;
    Frame(OctreeInternal* _node, double _dsq) : dsq(_dsq), node(_node) { }
  };

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    std::deque<Frame*> stack;
    std::deque<Frame> backing;

    backing.push_back(Frame(node, dsq));
    stack.push_back(&backing.back());

    Point delta;
    while (!stack.empty()) {
      Frame f = *stack.back();
      stack.pop_back();

      computeDelta(delta, body, f.node);

      double psq = delta.dist2();
      // Node is far enough away, summarize contribution
      if (psq >= f.dsq) {
        updateForce(body->acc, delta, psq, f.node->mass);
        continue;
      }

      double new_dsq = f.dsq * 0.25;
      
      for (int i = 0; i < 8; i++) {
        Octree *next = f.node->child[i];
        if (next == NULL)
          break;
        if (body == next)
          continue;
        if (next->isLeaf()) {
          Point delta;
          computeDelta(delta, body, next);
          // body->mass is fine because every body has the same mass
          updateForce(body->acc, delta, delta.dist2(), body->mass);
        } else {
          backing.push_back(Frame(static_cast<OctreeInternal*>(next), new_dsq));
          stack.push_back(&backing.back());
        }
      }
    }
  }
};

struct ComputeForceBlockedContinuation {
  Body* body;
  OctreeInternal* node;
  Octree** next;
  double dsq;

  ComputeForceBlockedContinuation(Body* b, OctreeInternal* n, double d):
    body(b), node(n), next(0), dsq(d) { 
    down(node);
  }
  
  void down(OctreeInternal* n) {
    node = n;

    Point delta;
    computeDelta(delta, body, node);
    double psq = delta.dist2();

    if (psq >= dsq) {
      updateForce(body->acc, delta, psq, node->mass);
      up();
      return;
    }

    next = node->child;
    dsq = dsq * 0.25;
  }

  void up() {
    while (true) {
      OctreeInternal* oldNode = node;

      node = static_cast<OctreeInternal*>(node->parent);
      if (!node)
        break;

      dsq = dsq * 4.0;

      for (int i = 0; i < 8; ++i) {
        assert(node->child[i]);
        if (node->child[i] == oldNode) {
          // always valid due to end of array element
          next = &node->child[i+1];
          if (!next)
            break;
          return;
        }
      }
    }
  }

 // void inspect() { } // XXX

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    Point delta;
    for (int i = 0; i < blockSizeY && node != NULL; ++i) {
      Octree* n = *next;
      if (!n) {
        up();
      } else if (body == n) {
        ++next;
        --i; // don't include empty work in block factor
      } else if (n->isLeaf()) {
        Point delta;
        computeDelta(delta, body, n);
        // body->mass is fine because every body has the same mass
        updateForce(body->acc, delta, delta.dist2(), body->mass);
        ++next;
      } else {
        down(static_cast<OctreeInternal*>(n));
      }
    }

    if (node) {
      ctx.addTask1(*this);
    }
  }
};

struct ComputeForceBlocked {
  //typedef galois::gdeque<Body*,32> Bodies;
  typedef std::array<Body*,MaxBlockSizeX> Bodies;
  Bodies bodies;
  OctreeInternal* top;
  double dsq;
  int bmax;

  ComputeForceBlocked(OctreeInternal* n, double d): top(n), dsq(d) { }
  
//  void inspect() { } // XXX

  struct Frame {
    double dsq;
    Body* body;
    OctreeInternal* node;
    Frame(Body* b, OctreeInternal* _node, double _dsq) : dsq(_dsq), body(b), node(_node) { }
  };

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    typedef galois::gdeque<Frame> Stack;
    typedef std::array<Stack,MaxBlockSizeX> Stacks; // XXX

    Stacks stacks;

    Bodies::iterator ii = bodies.begin();
    for (int i = 0; i < bmax; ++i) {
      stacks[i].push_back(Frame(*ii++, top, dsq));
    }

    Point delta;
    while (true) {
      int numEmpty = 0;

      for (int bindex = 0; bindex < bmax; ++bindex) {
        if (stacks[bindex].empty()) {
          ++numEmpty;
          continue;
        }
        for (int yindex = 0; !stacks[bindex].empty() && yindex < blockSizeY; ++yindex) {
          Frame f = stacks[bindex].back();
          stacks[bindex].pop_back();

          computeDelta(delta, f.body, f.node);
          double psq = delta.dist2();
          if (psq >= f.dsq) {
            updateForce(f.body->acc, delta, psq, f.node->mass);
            continue;
          }

          double new_dsq = f.dsq * 0.25;

          for (int i = 0; i < 8; i++) {
            Octree *next = f.node->child[i];
            if (next == NULL)
              break;
            if (f.body == next)
              continue;
            if (next->isLeaf()) {
              Point delta;
              computeDelta(delta, f.body, next);
              // body->mass is fine because every body has the same mass
              updateForce(f.body->acc, delta, delta.dist2(), f.body->mass);
            } else {
              if (blockFIFO)
                stacks[bindex].push_front(Frame(f.body, static_cast<OctreeInternal*>(next), new_dsq));
              else
                stacks[bindex].push_back(Frame(f.body, static_cast<OctreeInternal*>(next), new_dsq));
            }
          }
        }
      }
      if (numEmpty == bmax)
        break;
    }
  }
};

struct ComputeForceOrig {
  Body* body;
  OctreeInternal* node;
  double dsq;

  ComputeForceOrig(Body* b, OctreeInternal* n, double d): body(b), node(n), dsq(d) { }
  
  void inspect() { } // XXX

  struct Frame {
    double dsq;
    OctreeInternal* node;
    Frame(OctreeInternal* _node, double _dsq) : dsq(_dsq), node(_node) { }
  };

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    galois::gdeque<Frame> stack;
    stack.push_back(Frame(node, dsq));

    Point delta;
    while (!stack.empty()) {
      Frame f = stack.back();
      stack.pop_back();

      computeDelta(delta, body, f.node);

      double psq = delta.dist2();
      // Node is far enough away, summarize contribution
      if (psq >= f.dsq) {
        updateForce(body->acc, delta, psq, f.node->mass);
        continue;
      }

      double new_dsq = f.dsq * 0.25;
      
      for (int i = 0; i < 8; i++) {
        Octree *next = f.node->child[i];
        if (next == NULL)
          break;
        if (body == next)
          continue;
        if (next->isLeaf()) {
          Point delta;
          computeDelta(delta, body, next);
          // body->mass is fine because every body has the same mass
          updateForce(body->acc, delta, delta.dist2(), body->mass);
        } else {
          stack.push_back(Frame(static_cast<OctreeInternal*>(next), new_dsq));
        }
      }
    }
  }
};

struct FinishForceTree {
  typedef std::array<Point,8> Accs;
  Accs accs;
  Point* parent;

  FinishForceTree(Point* p): parent(p) { }

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    for (Accs::iterator ii = accs.begin(), ei = accs.end(); ii != ei; ++ii) {
      *parent += *ii;
    }
  }
};

struct ComputeForceTree {
  typedef std::list<galois::Task,galois::PerIterAllocTy::rebind<galois::Task>::other> Tasks;

  Point* acc;
  Body* body;
  OctreeInternal* node;
  double dsq;
  galois::Task parent;

  ComputeForceTree(Point* a, Body* b, OctreeInternal* n, double d, galois::Task p): 
    acc(a), body(b), node(n), dsq(d), parent(p) { }
  
  void inspect() { } // XXX

  template<typename PipelineTy>
  void recurse(galois::TaskContext<PipelineTy>& ctx, OctreeInternal* node, double dsq, int depth, Tasks& tasks) {
    Point delta;
    computeDelta(delta, body, node);

    double psq = delta.dist2();
    // Node is far enough away, summarize contribution
    if (psq >= dsq) {
      updateForce(*acc, delta, psq, node->mass);
      return;
    }

    galois::runtime::Task::GTask<FinishForceTree>* after = NULL;

    double new_dsq = dsq * 0.25;
    
    for (int i = 0; i < 8; i++) {
      Octree *next = node->child[i];
      if (next == NULL)
        break;
      if (body == next)
        continue;
      if (next->isLeaf()) {
        Point delta;
        computeDelta(delta, body, next);
        // body->mass is fine because every body has the same mass
        updateForce(*acc, delta, delta.dist2(), body->mass);
      } else if (depth > 0 || depth < 0) {
        recurse(ctx, static_cast<OctreeInternal*>(next), new_dsq, depth - 1, tasks);
      } else {
        if (!after) {
          after = ctx.addTask2(FinishForceTree(acc));
          if (parent)
            ctx.addDependence(after, parent);
          tasks.push_back(after);
        }

        galois::Task t = 
          ctx.addTask1(ComputeForceTree(&after->task.accs[i], body, static_cast<OctreeInternal*>(next), new_dsq, after));
        ctx.addDependence(t, after);
        ctx.markDeferred(after);
      }
    }
  }

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    Tasks tasks(ctx.getPerIterAlloc());
    recurse(ctx, node, dsq, 3, tasks); // XXX
    //galois::Task prev = { };
    galois::Task prev = 0;
    //for (Tasks::reverse_iterator ii = tasks.rbegin(), ei = tasks.rend(); ii != ei; ++ii) { //XXX
    for (Tasks::iterator ii = tasks.begin(), ei = tasks.end(); ii != ei; ++ii) {
      if (prev)
        ctx.addDependence(prev, *ii);
      prev = *ii;
    }
  }
};

struct ComputeForceUnroll {
  typedef int tt_needs_aborts;

  Body* body;
  OctreeInternal* node;
  double dsq;

  ComputeForceUnroll(Body* b, OctreeInternal* n, double d): body(b), node(n), dsq(d) { }
  
  void inspect() { } // XXX

  template<typename PipelineTy>
  void recurse(galois::TaskContext<PipelineTy>& ctx, OctreeInternal* node, double dsq, int depth) {
    Point delta;
    computeDelta(delta, body, node);

    double psq = delta.dist2();
    // Node is far enough away, summarize contribution
    if (psq >= dsq) {
      updateForce(body->acc, delta, psq, node->mass);
      return;
    }

    double new_dsq = dsq * 0.25;
    
    for (int i = 0; i < 8; i++) {
      Octree *next = node->child[i];
      if (next == NULL)
        break;
      if (body == next)
        continue;
      if (next->isLeaf()) {
        Point delta;
        computeDelta(delta, body, next);
        // body->mass is fine because every body has the same mass
        updateForce(body->acc, delta, delta.dist2(), body->mass);
      } else if (depth > 0) {
        recurse(ctx, static_cast<OctreeInternal*>(next), new_dsq, depth - 1);
      } else {
        ctx.addTask1(ComputeForceUnroll(body, static_cast<OctreeInternal*>(next), new_dsq));
      }
    }
  }

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    galois::runtime::acquire(body, galois::MethodFlag::WRITE);
    recurse(ctx, node, dsq, 4);
  }
};

struct ComputeForceBase {
  typedef int tt_needs_aborts;

  Body* body;
  OctreeInternal* node;
  double dsq;

  ComputeForceBase(Body* b, OctreeInternal* n, double d): body(b), node(n), dsq(d) { }
  
  void inspect() { } // XXX

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>& ctx) {
    Point delta;
    galois::runtime::acquire(body, galois::MethodFlag::WRITE);
    computeDelta(delta, body, node);

    double psq = delta.dist2();
    // Node is far enough away, summarize contribution
    if (psq >= dsq) {
      updateForce(body->acc, delta, psq, node->mass);
      return;
    }

    double new_dsq = dsq * 0.25;
    
    for (int i = 0; i < 8; i++) {
      Octree *next = node->child[i];
      if (next == NULL)
        break;
      if (body == next)
        continue;
      if (next->isLeaf()) {
        Point delta;
        computeDelta(delta, body, next);
        // body->mass is fine because every body has the same mass
        updateForce(body->acc, delta, delta.dist2(), body->mass);
      } else {
        ctx.addTask1(ComputeForceBase(body, static_cast<OctreeInternal*>(next), new_dsq));
      }
    }
  }
};

struct ComputeForceTaskOrig {
  struct Task { 
    Body* body;
    Octree* other;
    Task(Body* b, Octree* o): body(b), other(o) { }
  };

  struct Frame {
    double dsq;
    OctreeInternal* node;
    Frame(OctreeInternal* n, double d) : dsq(d), node(n) { }
  };

  template<typename IterTy,typename OutTy>
  void generateTasks(double dsq, OctreeInternal* top, IterTy ii, IterTy ei, OutTy& out) {
    size_t dist = std::distance(ii, ei);
    size_t block = (dist + (numThreads - 1)) / numThreads;
    size_t count = 0;

    for (; ii != ei; ++ii, ++count) {
      int tid = count / block;
      Body* body = *ii;
      galois::gdeque<Frame> stack;
      stack.push_back(Frame(top, dsq));

      Point delta;
      while (!stack.empty()) {
        Frame f = stack.back();
        stack.pop_back();

        computeDelta(delta, body, f.node);
        double psq = delta.dist2();
        if (psq >= f.dsq) {
          out[tid].push_back(Task(body, f.node));
          continue;
        }

        double new_dsq = f.dsq * 0.25;

        for (int i = 0; i < 8; i++) {
          Octree *next = f.node->child[i];
          if (next == NULL)
            break;
          if (body == next)
            continue;
          if (next->isLeaf()) {
            out[tid].push_back(Task(body, next));
          } else {
            stack.push_back(Frame(static_cast<OctreeInternal*>(next), new_dsq));
          }
        }
      }
    }
  }

  void operator()(const Task& t) {
    Point delta;
    computeDelta(delta, t.body, t.other);
    updateForce(t.body->acc, delta, delta.dist2(), t.other->mass);
  }
};

struct ComputeForceTaskBlocked {
  struct Task { 
    Body* body;
    Octree* other;
    Task(Body* b, Octree* o): body(b), other(o) { }
  };

  struct Frame {
    double dsq;
    Body* body;
    OctreeInternal* node;
    int tid;
    Frame(int tid, Body* b, OctreeInternal* n, double d): dsq(d), body(b), node(n), tid(tid) { }
  };

  template<typename IterTy,typename OutTy>
  void generateTasks(double dsq, OctreeInternal* top, IterTy ii, IterTy ei, OutTy& out) {
    typedef galois::gdeque<Frame> Stack;
    typedef std::array<Stack,MaxBlockSizeX> Stacks;

    Stacks stacks;
    //stacks.resize(blockSizeX);

    size_t dist = std::distance(ii, ei);
    size_t block = (dist + (numThreads - 1)) / numThreads;
    size_t count = 0;

    while (ii != ei) {
      int bmax;
      for (bmax = 0; ii != ei && bmax < blockSizeX; ++ii, ++count, ++bmax) {
        int tid = count / block;
        Body* body = *ii;
        stacks[bmax].push_back(Frame(tid, body, top, dsq));
      }

      Point delta;
      while (true) {
        int numEmpty = 0;

        for (int bindex = 0; bindex < bmax; ++bindex) {
          if (stacks[bindex].empty()) {
            ++numEmpty;
            continue;
          }
          for (int yindex = 0; !stacks[bindex].empty() && yindex < blockSizeY; ++yindex) {
            Frame f = stacks[bindex].back();
            stacks[bindex].pop_back();

            computeDelta(delta, f.body, f.node);
            double psq = delta.dist2();
            if (psq >= f.dsq) {
              out[f.tid].push_back(Task(f.body, f.node));
              continue;
            }

            double new_dsq = f.dsq * 0.25;

            for (int i = 0; i < 8; i++) {
              Octree *next = f.node->child[i];
              if (next == NULL)
                break;
              if (f.body == next)
                continue;
              if (next->isLeaf()) {
                out[f.tid].push_back(Task(f.body, next));
              } else {
                if (blockFIFO)
                  stacks[bindex].push_front(Frame(f.tid, f.body, static_cast<OctreeInternal*>(next), new_dsq));
                else
                  stacks[bindex].push_back(Frame(f.tid, f.body, static_cast<OctreeInternal*>(next), new_dsq));
              }
            }
          }
        }
        if (numEmpty == bmax)
          break;
      }
    }
  }

  void operator()(const Task& t) {
    Point delta;
    computeDelta(delta, t.body, t.other);
    updateForce(t.body->acc, delta, delta.dist2(), t.other->mass);
  }
};

struct AdvanceBody {
  Body* body;
  Bodies* new_bodies;

  AdvanceBody(Body* b, Bodies* n): body(b), new_bodies(n) { }

  void inspect() { } // XXX

  template<typename PipelineTy>
  void operator()(galois::TaskContext<PipelineTy>&) {
    for (int i = 0; i < 3; ++i) {
      body->vel[i] += (body->acc[i] - body->oldacc[i]) * config.dthf;
    }

    Point dvel(body->acc);
    dvel *= config.dthf;

    Point velh(body->vel);
    velh += dvel;

    for (int i = 0; i < 3; ++i) {
      body->pos[i] += velh[i] * config.dtime;
      body->vel[i] = velh[i] + dvel[i];
      body->oldacc[i] = body->acc[i];
      body->acc[i] = 0;
    }
    
    // new_bodies->push_back(*body); XXX
  }
};

// XXX use reduction support? or NUMA?
struct ReduceBoxes {
  BoundingBox& initial;

  ReduceBoxes(BoundingBox& _initial): initial(_initial) { }

  void operator()(Body* b) {
    initial.merge(b->pos);
  }
};

double nextDouble() {
  return rand() / (double) RAND_MAX;
}

/**
 * Generates random input according to the Plummer model, which is more
 * realistic but perhaps not so much so according to astrophysicists
 */
void generateInput(Bodies& bodies, int nbodies, int seed) {
  double v, sq, scale;
  Point p;
  double PI = boost::math::constants::pi<double>();

  srand(seed);

  double rsc = (3 * PI) / 16;
  double vsc = sqrt(1.0 / rsc);

  for (int body = 0; body < nbodies; body++) {
    double r = 1.0 / sqrt(pow(nextDouble() * 0.999, -2.0 / 3.0) - 1);
    do {
      for (int i = 0; i < 3; i++)
        p[i] = nextDouble() * 2.0 - 1.0;
      sq = p.x * p.x + p.y * p.y + p.z * p.z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);

    Body b;
    b.mass = 1.0 / nbodies;
    for (int i = 0; i < 3; i++)
      b.pos[i] = p[i] * scale;

    do {
      p.x = nextDouble();
      p.y = nextDouble() * 0.1;
    } while (p.y > p.x * p.x * pow(1 - p.x * p.x, 3.5));
    v = p.x * sqrt(2.0 / sqrt(1 + r * r));
    do {
      for (int i = 0; i < 3; i++)
        p[i] = nextDouble() * 2.0 - 1.0;
      sq = p.x * p.x + p.y * p.y + p.z * p.z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    for (int i = 0; i < 3; i++) {
      b.vel[i] = p[i] * scale;
      b.acc[i] = 0;
      b.oldacc[i] = 0;
    }

    bodies.push_back(b);
  }
}

template<typename T>
struct deref: public std::unary_function<T, T*> {
  T* operator()(T& item) const { return &item; }
};

template<typename ComputeForceTy,typename WLTy,typename Enable=void>
struct DoComputeForce {
  struct MakeComputeForce: public std::unary_function<Body*,ComputeForceTy> {
    OctreeInternal* top;
    double dsq;

    MakeComputeForce(OctreeInternal* t, double d): top(t), dsq(d) { }

    ComputeForceTy operator()(Body* b) const {
      return ComputeForceTy(b, top, dsq);
    }
  };

  template<typename IterTy>
  void operator()(double dsq, OctreeInternal* top, IterTy begin, IterTy end) {
    typedef galois::Pipeline<WLTy,ComputeForceTy> P;
    galois::for_each_task<P>(
        boost::make_transform_iterator(begin, MakeComputeForce(top, dsq)),
        boost::make_transform_iterator(end, MakeComputeForce(top, dsq)),
        "MakeComputeForce");
  }
};

//! Returned blocked ranges for a fixed block size
template<typename IterTy>
class blocked_iterator: public std::iterator<std::forward_iterator_tag, std::pair<IterTy,IterTy> > {
  IterTy ii;
  IterTy ei;
  int block;
public:
  blocked_iterator() { }
  blocked_iterator(const IterTy& ii, const IterTy& ei, int block): ii(ii), ei(ei), block(block) { }
  bool operator==(const blocked_iterator& rhs) const { return ii == rhs.ii; }
  bool operator!=(const blocked_iterator& rhs) const { return ii != rhs.ii; }
  std::pair<IterTy,IterTy> operator*() const { 
    IterTy b(ii);
    IterTy e(ii);
    for (int i = 0; e != ei && i < block; ++i, ++e)
      ;
    return std::make_pair(b, e);
  }
  blocked_iterator& operator++() {
    for (int i = 0; ii != ei && i < block; ++i, ++ii)
      ;
    return *this;
  }
  blocked_iterator& operator++(int) { blocked_iterator tmp(*this); ++(*this); return tmp; }
};

template<typename WLTy,typename Enable>
struct DoComputeForce<ComputeForceBlocked,WLTy,Enable> {
  template<typename T>
  struct MakeComputeForce: public std::unary_function<T,ComputeForceBlocked> {
    OctreeInternal* top;
    double dsq;

    MakeComputeForce(OctreeInternal* t, double d): top(t), dsq(d) { }

    ComputeForceBlocked operator()(const T& p) const {
      ComputeForceBlocked x(top, dsq);
      std::copy(p.first, p.second, x.bodies.begin());
      x.bmax = std::distance(p.first, p.second);
      return x;
    }
  };

  template<typename IterTy>
  void operator()(double dsq, OctreeInternal* top, IterTy begin, IterTy end) {
    typedef galois::Pipeline<WLTy,ComputeForceBlocked> P;
    typedef blocked_iterator<IterTy> BlockedIt;
    typedef typename std::iterator_traits<BlockedIt>::value_type value_type;

    galois::for_each_task<P>(
        boost::make_transform_iterator(BlockedIt(begin, end, blockSizeX),
          MakeComputeForce<value_type>(top, dsq)),
        boost::make_transform_iterator(BlockedIt(end, end, blockSizeX),
          MakeComputeForce<value_type>(top, dsq)),
        "MakeComputeForce");
  }
};

template<typename WLTy,typename Enable>
struct DoComputeForce<ComputeForceTree,WLTy,Enable> {
  struct MakeComputeForce: public std::unary_function<Body*,ComputeForceTree> {
    OctreeInternal* top;
    double dsq;

    MakeComputeForce(OctreeInternal* t, double d): top(t), dsq(d) { }

    ComputeForceTree operator()(Body* b) const {
      return ComputeForceTree(&b->acc, b, top, dsq, 0);
    }
  };

  template<typename IterTy>
  void operator()(double dsq, OctreeInternal* top, IterTy begin, IterTy end) {
    typedef galois::Pipeline<WLTy,ComputeForceTree,FinishForceTree> P;
    galois::for_each_task<P>(
        boost::make_transform_iterator(begin, MakeComputeForce(top, dsq)),
        boost::make_transform_iterator(end, MakeComputeForce(top, dsq)),
        "MakeComputeForce");
  }
};

//! Some template meta programming
template<typename T>
struct has_task {
  typedef char yes[1];
  typedef char no[2];
  template<typename U> static yes& test(typename U::Task*);
  template<typename> static no& test(...);
  static const bool value = sizeof(test<T>(0)) == sizeof(yes);
};

//! Simple iterator that returns iterators rather than values
template<typename IterTy>
class passthrough: public std::iterator<std::forward_iterator_tag, IterTy> {
  IterTy ii;
public:
  passthrough() { }
  passthrough(const IterTy& ii): ii(ii) { }
  bool operator==(const passthrough& rhs) const { return ii == rhs.ii; }
  bool operator!=(const passthrough& rhs) const { return ii != rhs.ii; }
  const IterTy& operator*() const { return ii; }
  IterTy& operator*() { return ii; }
  passthrough& operator++() { ++ii; return *this; }
  passthrough operator++(int) { passthrough tmp(*this); ++ii; return tmp; }
};

template<typename T>
double refineDelta(const T& a, const T& b) {
  double bdiff = reinterpret_cast<intptr_t>(a.body) - reinterpret_cast<intptr_t>(b.body);
  double odiff = reinterpret_cast<intptr_t>(a.other) - reinterpret_cast<intptr_t>(b.other);
  return bdiff * bdiff + odiff * odiff;
}

template<typename IterTy>
struct Refine {
  IterTy ii;
  IterTy ei;
  size_t dist;

  bool improvement(const IterTy& a, const IterTy& b) {
    IterTy anext = skip(a, 1);
    IterTy bnext = skip(b, 1);
    IterTy aprev = skip(a, -1);
    IterTy bprev = skip(b, -1);

    return refineDelta(*aprev, *b) + refineDelta(*b, *anext) < refineDelta(*aprev, *a) + refineDelta(*a, *anext)
      && refineDelta(*bprev, *a) + refineDelta(*a, *bnext) < refineDelta(*bprev, *b) + refineDelta(*b, *bnext);
    //return refineDelta(*b, *anext) < refineDelta(*a, *anext) // XXX
    //  && refineDelta(*a, *bnext) < refineDelta(*b, *bnext);
  }

  IterTy skip(IterTy c, int n) {
    if (n > 0) {
      int remaining = std::distance(c, ei);
      if (remaining > n) {
        std::advance(c, n);
      } else {
        c = ii;
        std::advance(c, n - remaining);
      }
    } else {
      int remaining = std::distance(ii, c);
      if (remaining > -n) {
        std::advance(c, n);
      } else {
        c = ei;
        std::advance(c, n + remaining);
      }
    }
      
    return c;
  }

  Refine(const IterTy& ii, const IterTy& ei, size_t dist): ii(ii), ei(ei), dist(dist) { }

  void operator()(const IterTy& cur, galois::UserContext<IterTy>&) {
    IterTy next = skip(cur, dist);
    if (improvement(cur, next)) {
      galois::runtime::acquire((*cur).body, galois::MethodFlag::WRITE);
      galois::runtime::acquire((*next).body, galois::MethodFlag::WRITE);
      std::swap(*cur, *next);
    }
  }
};

template<typename IterTy>
struct ComputeRefineDelta {
  typedef typename std::iterator_traits<IterTy>::value_type value_type;
  IterTy ei;
  ComputeRefineDelta(const IterTy& ei): ei(ei) { }
  double operator()(IterTy cur) const {
    const value_type& v = *cur;
    std::advance(cur, 1);
    if (cur == ei)
      return 0.0;
    double retval = refineDelta(v, *cur);
    assert(retval >= 0);
    return retval;
  }
};

template<typename OutTy>
void refine(OutTy& out) {
  typedef typename OutTy::iterator Iter;
  double newDelta = 
    galois::ParallelSTL::map_reduce(
            passthrough<Iter>(out.begin()),
            passthrough<Iter>(out.end()),
            ComputeRefineDelta<Iter>(out.end()),
            0.0,
            std::plus<double>());
  double oldDelta = 0;
  //size_t dist = out.size();

  while (true) {
    std::cout << "DELTA: " << oldDelta << " " << newDelta << " " << (oldDelta - newDelta)/oldDelta << "\n";
    if (oldDelta != 0 && newDelta > oldDelta)
      break;
    if (oldDelta != 0 && (oldDelta - newDelta)/oldDelta < 0.01)
      break;

    for (size_t d = out.size() >> 1; d != 0; d = d >> 1) {
    //for (size_t d = 1024; d != 0; d = d >> 1) { // XXX
      galois::for_each(passthrough<Iter>(out.begin()), passthrough<Iter>(out.end()),
          Refine<Iter>(out.begin(), out.end(), d));
    }
    oldDelta = newDelta;
    newDelta = galois::ParallelSTL::map_reduce(
        passthrough<Iter>(out.begin()),
        passthrough<Iter>(out.end()),
        ComputeRefineDelta<Iter>(out.end()),
        0.0,
        std::plus<double>());
  }
}

template<typename ComputeForceTy,typename WLTy>
struct DoComputeForce<ComputeForceTy,WLTy,typename boost::enable_if<has_task<ComputeForceTy> >::type> {

  template<typename OutsTy>
  struct Wrapper {
    ComputeForceTy& w;
    OutsTy& outs;
    Wrapper(ComputeForceTy& w, OutsTy& outs): w(w), outs(outs) { }

    void operator()(unsigned tid, unsigned) {
      std::for_each(outs[tid].begin(), outs[tid].end(), w);
    }
  };

  template<typename IterTy>
  void operator()(double dsq, OctreeInternal* top, IterTy begin, IterTy end) {
    //typedef galois::gdeque<typename ComputeForceTy::Task> Deque; // XXX
    typedef std::vector<typename ComputeForceTy::Task> Deque;
    typedef std::vector<Deque> Outs;

    ComputeForceTy W;
    Outs outs;

    outs.resize(numThreads);

    galois::StatTimer genTime("GenerateTasksTime");
    genTime.start();
    W.generateTasks(dsq, top, begin, end, outs);
    genTime.stop();

    if (useTaskRefine) {
      galois::StatTimer refineTime("RefineTasksTime");
      refineTime.start();
      for (int i = 0; i < numThreads; ++i) {
        std::cout << "Refine " << i << "\n";
        refine(outs[i]);
      }
      refineTime.stop();
    }
    
    galois::StatTimer computeTime("ParallelComputeForceTime");
    computeTime.start();
    galois::on_each(Wrapper<Outs>(W, outs), galois::loopname("MakeComputeForce"));
    computeTime.stop();
  }
};

struct MakeAdvanceBody: public std::unary_function<Body*,AdvanceBody> {
  Bodies* new_bodies;

  MakeAdvanceBody(Bodies* n): new_bodies(n) { }

  AdvanceBody operator()(Body* b) const {
    return AdvanceBody(b, new_bodies);
  }
};

struct CheckAllPairs {
  Bodies& bodies;
  
  CheckAllPairs(Bodies& b): bodies(b) { }

  double operator()(const Body& body) const {
    const Body* me = &body;
    Point acc;
    for (Bodies::iterator ii = bodies.begin(), ei = bodies.end(); ii != ei; ++ii) {
      Body* b = &*ii;
      if (me == b)
        continue;

      Point delta;
      computeDelta(delta, me, b);
      double psq = delta.dist2();
      updateForce(acc, delta, psq, b->mass);
    }

    double dist2 = acc.dist2();
    for (int i = 0; i < 3; ++i)
      acc[i] -= me->acc[i];
    double retval = acc.dist2() / dist2;
    return retval;
  }
};

double checkAllPairs(Bodies& bodies, int N) {
  Bodies::iterator end(bodies.begin());
  std::advance(end, N);
  
  return galois::ParallelSTL::map_reduce(bodies.begin(), end,
      CheckAllPairs(bodies),
      0.0,
      std::plus<double>()) / N;
}

void sort(Bodies& bodies) {
  // XXX resort
  // XXX resort better
  BodyPtrs ptrs;
  std::copy(
      boost::make_transform_iterator(bodies.begin(), deref<Body>()),
      boost::make_transform_iterator(bodies.end(), deref<Body>()),
      std::back_inserter(ptrs));

  BoundingBox box;
  ReduceBoxes reduceBoxes(box);
  std::for_each(ptrs.begin(), ptrs.end(), ReduceBoxes(box)); // XXX

  BuildOctree::BodyPtrsAlloc bodyPtrsAlloc;
  BuildOctree::OctreeInternalAlloc octreeInternalAlloc;
  OctreeInternal* top = octreeInternalAlloc.allocate(1);
  octreeInternalAlloc.construct(top, OctreeInternal(box.center(), static_cast<Octree*>(0)));

  // XXX
  typedef galois::Pipeline<galois::worklists::dChunkedLIFO<>,BuildOctree,ComputeCenterOfMass> P1;
  galois::for_each_task<P1>(BuildOctree(bodyPtrsAlloc, octreeInternalAlloc, 
        ptrs.begin(), ptrs.end(), top, box.radius(), 0), "Sort1");

  if (false) {
    typedef galois::InsertBag<Body> Bag;
    typedef galois::Pipeline<galois::worklists::dChunkedFIFO<>, Sort<Bag> > P2;
    Bag bag;
    galois::for_each_task<P2>(Sort<Bag>(top, bag), "Sort2");
    //delete top;
    bodies.clear();
    std::copy(bag.begin(), bag.end(), std::back_inserter(bodies));
  } else {
    typedef galois::gdeque<Body> Bag;
    Bag bag;
    Sort<Bag> sorter(top, bag);
    sorter(top);
    //delete top;
    bodies.clear();
    std::copy(bag.begin(), bag.end(), std::back_inserter(bodies));
  }
}

void run(Bodies& bodies) {
  Bodies new_bodies;
  BodyPtrs ptrs;

  std::copy(
      boost::make_transform_iterator(bodies.begin(), deref<Body>()),
      boost::make_transform_iterator(bodies.end(), deref<Body>()),
      std::back_inserter(ptrs));

  for (int step = 0; step < ntimesteps; step++) {
    BuildOctree::BodyPtrsAlloc bodyPtrsAlloc;
    BuildOctree::OctreeInternalAlloc octreeInternalAlloc;

    BoundingBox box;
    ReduceBoxes reduceBoxes(box);
    std::for_each(ptrs.begin(), ptrs.end(), ReduceBoxes(box)); // XXX

    OctreeInternal* top = octreeInternalAlloc.allocate(1);
    octreeInternalAlloc.construct(top, box.center(), static_cast<Octree*>(0));

    galois::StatTimer Toctree("BuildOctreeTime");
    Toctree.start();
    if (useCilkTreeBuild) {
      BuildOctreeCilk c(bodyPtrsAlloc, octreeInternalAlloc);
      c.buildOctree(ptrs.begin(), ptrs.end(), top, box.radius());
    } else if (allocateFIFO) {
      typedef galois::Pipeline<galois::worklists::dChunkedFIFO<>,BuildOctree,ComputeCenterOfMass> P1;
      galois::for_each_task<P1>(BuildOctree(bodyPtrsAlloc, octreeInternalAlloc, ptrs.begin(), ptrs.end(), top, box.radius(), 0), "BuildOctree");
    } else {
      typedef galois::Pipeline<galois::worklists::dChunkedLIFO<>,BuildOctree,ComputeCenterOfMass> P1;
      galois::for_each_task<P1>(BuildOctree(bodyPtrsAlloc, octreeInternalAlloc, ptrs.begin(), ptrs.end(), top, box.radius(), 0), "BuildOctree");
    }
    Toctree.stop();

    typedef galois::worklists::dChunkedFIFO<> FIFO;
    typedef galois::worklists::dChunkedLIFO<> LIFO;
    typedef galois::worklists::LocalWorklist<galois::worklists::GFIFO<> > LocalFIFO;

    galois::StatTimer T("ComputeForceTime");
    T.start();
    // XXX Use octree to align bodies in next version of bodies vector
    double dsq = box.diameter() * box.diameter() * config.itolsq;
    switch (algo) {
      case algoBase: DoComputeForce<ComputeForceBase,FIFO>()(dsq, top, ptrs.begin(), ptrs.end()); break;
      case algoOrig: DoComputeForce<ComputeForceOrig,FIFO>()(dsq, top, ptrs.begin(), ptrs.end()); break;
      case algoBlocked: DoComputeForce<ComputeForceBlocked,LocalFIFO>()(dsq, top, ptrs.begin(), ptrs.end()); break;
      case algoBlockedC: DoComputeForce<ComputeForceBlockedContinuation,FIFO>()(dsq, top, ptrs.begin(), ptrs.end()); break;
      case algoIndirect: DoComputeForce<ComputeForceIndirect,FIFO>()(dsq, top, ptrs.begin(), ptrs.end()); break;
      case algoUnroll: DoComputeForce<ComputeForceUnroll,FIFO>()(dsq, top, ptrs.begin(), ptrs.end()); break;
      case algoTree: DoComputeForce<ComputeForceTree,FIFO>()(dsq, top, ptrs.begin(), ptrs.end()); break;
      case algoTaskOrig: DoComputeForce<ComputeForceTaskOrig,FIFO>()(dsq, top, ptrs.begin(), ptrs.end()); break;
      case algoTaskBlocked: DoComputeForce<ComputeForceTaskBlocked,FIFO>()(dsq, top, ptrs.begin(), ptrs.end()); break;
      default: abort();
    }
    T.stop();

    if (!skipVerify) {
      std::cout << "MSE (sampled) " << checkAllPairs(bodies, std::min((int) nbodies, 100)) << "\n";
    }

    typedef galois::Pipeline<galois::worklists::dChunkedFIFO<>,AdvanceBody> P3;
    galois::for_each_task<P3>(
        boost::make_transform_iterator(ptrs.begin(), MakeAdvanceBody(&new_bodies)),
        boost::make_transform_iterator(ptrs.end(), MakeAdvanceBody(&new_bodies)),
        "AdvanceBody");

    std::cout << "Timestep " << step << " Center of Mass = ";
    std::ios::fmtflags flags = 
      std::cout.setf(std::ios::showpos|std::ios::right|std::ios::scientific|std::ios::showpoint);
    std::cout << top->pos;
    std::cout.flags(flags);
    std::cout << "\n";

    // XXX some simple parallel deallocation too using a counter just to eliminate empty pages
    // should cover nested and streaming case
    //delete top;
    //bodies.clear(); XXX
    //ptrs.clear();
    //boost::swap(bodies, new_bodies);
  }
}

int main(int argc, char** argv) {
  galois::StatManager M;
  LonestarStart(argc, argv, name, desc, url);

  // FixThreads fixThreads;
  // tbb::task_scheduler_init iii(tbb::task_scheduler_init::default_num_threads()); XXX

  if (blockSizeX > MaxBlockSizeX) {
    std::cerr << "blockSizeX too large\n";
    abort();
  }

  std::cerr << "configuration: "
            << nbodies << " bodies, "
            << ntimesteps << " time steps" << std::endl << std::endl;

  Bodies bodies;
  generateInput(bodies, nbodies, seed);
  sort(bodies);

  galois::reportPageAlloc("MemPre");
  galois::StatTimer T;
  T.start();
  run(bodies);
  T.stop();
  galois::reportPageAlloc("MemPost");
}
