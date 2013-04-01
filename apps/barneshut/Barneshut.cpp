/** Barnes-hut application -*- C++ -*-
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
 * @author Martin Burtscher <burtscher@txstate.edu>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Bag.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/WorkList/WorkListAlt.h"

#include <boost/math/constants/constants.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <limits>
#include <iostream>
#include <strings.h>
#include <deque>

const char* name = "Barnshut N-Body Simulator";
const char* desc =
  "Simulation of the gravitational forces in a galactic cluster using the "
  "Barnes-Hut n-body algorithm\n";
const char* url = "barneshut";

static llvm::cl::opt<int> nbodies("n", llvm::cl::desc("Number of bodies"), llvm::cl::init(10000));
static llvm::cl::opt<int> ntimesteps("steps", llvm::cl::desc("Number of steps"), llvm::cl::init(1));
static llvm::cl::opt<int> seed("seed", llvm::cl::desc("Random seed"), llvm::cl::init(7));

struct Point : public Galois::Runtime::Lockable {
  double x, y, z;
  Point() : x(0.0), y(0.0), z(0.0) { }
  Point(double _x, double _y, double _z) : x(_x), y(_y), z(_z) { }
  explicit Point(double v) : x(v), y(v), z(v) { }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,x,y,z);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,x,y,z);
  }

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
struct Octree : public Galois::Runtime::Lockable {
  Point pos;
  double mass;
  bool Leaf;
/*
  virtual ~Octree() { }
 */
  Octree(bool l = true) :Leaf(l) {}
  Octree(const Point& p, double m, bool l) :pos(p), mass(m), Leaf(l) {}
  //  Octree(const Point& p, double m = 0.0) :pos(p), mass(m) {}
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,pos,mass,Leaf);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,pos,mass,Leaf);
  }
};

struct OctreeInternal : Octree {
  gptr<Octree> child[8];
  OctreeInternal() { }
  OctreeInternal(const Point& _pos) : Octree(_pos, 0.0, false) {
    bzero(child, sizeof(gptr<Octree>) * 8);
  }
 // FIX ME ---
/*
  virtual ~OctreeInternal() {
    for (int i = 0; i < 8; i++) {
      if (OctreeInternal* B = dynamic_cast<OctreeInternal*>(child[i])) {
        delete B;
      }
    }
  }
 */
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,*(const_cast<Octree*>(static_cast<const Octree*>(this))));
    for (int i = 0; i < 8; i++) {
      gSerialize(s,child[i]);
    }
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,*(const_cast<Octree*>(static_cast<const Octree*>(this))));
    for (int i = 0; i < 8; i++) {
      gDeserialize(s,child[i]);
    }
  }
};

struct Body : Octree {
  Point vel;
  Point acc;
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,*(const_cast<Octree*>(static_cast<const Octree*>(this))));
    gSerialize(s,vel,acc);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,*(const_cast<Octree*>(static_cast<const Octree*>(this))));
    gDeserialize(s,vel,acc);
  }
};

std::ostream& operator<<(std::ostream& os, const Body& b) {
  os << "(pos:" << b.pos
     << " vel:" << b.vel
     << " acc:" << b.acc
     << " mass:" << b.mass << ")";
  return os;
}

struct BoundingBox : public Galois::Runtime::Lockable {
  Point min;
  Point max;
  explicit BoundingBox(const Point& p) : min(p), max(p) { }
  BoundingBox() :
    min(std::numeric_limits<double>::max()),
    max(std::numeric_limits<double>::min()) { }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,min,max);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,min,max);
  }

  void merge(const BoundingBox& other) {
    for (int i = 0; i < 3; i++) {
      if (other.min[i] < min[i])
        min[i] = other.min[i];
      if (other.max[i] > max[i])
	max[i] = other.max[i];
    }
  }

  void merge(const Point& other) {
    for (int i = 0; i < 3; i++) {
      if (other[i] < min[i])
        min[i] = other[i];
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
  Config():
    dtime(0.5),
    eps(0.05),
    tol(0.025),
    dthf(dtime * 0.5),
    epssq(eps * eps),
    itolsq(1.0 / (tol * tol))  { }
};

std::ostream& operator<<(std::ostream& os, const Config& c) {
  os << "Barnes-Hut configuration:"
    << " dtime: " << c.dtime
    << " eps: " << c.eps
    << " tol: " << c.tol;
  return os;
}

Config config;

inline int getIndex(const Point& a, const Point& b) {
  int index = 0;
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

typedef Galois::Graph::Bag<Body> BodyBag;
typedef Galois::Graph::Bag<Body>::pointer Bodies;
typedef Galois::Graph::Bag<Body*>::pointer BodyPtrs;

struct BuildOctree : public Galois::Runtime::Lockable {
  gptr<OctreeInternal> root;
  double root_radius;

  BuildOctree() { }
  BuildOctree(gptr<OctreeInternal> _root, double radius) :
    root(_root),
    root_radius(radius) { }

  template<typename Context>
  void operator()(Body* b, Context& cnx) {
    insert(b, root, root_radius);
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,root,root_radius);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,root,root_radius);
  }

  void insert(Body* b, gptr<OctreeInternal> node, double radius) {
    int index = getIndex(node->pos, b->pos);

    gptr<Octree> child = node->child[index];
    
    if (!child) {
      gptr<Octree> tmp(static_cast<Body*>(b));
      node->child[index] = tmp;
      return;
    }
    
    radius *= 0.5;
    if (child->Leaf) {
      // Expand leaf
      Body* n = static_cast<Body*>(&(*child));
      Point new_pos(node->pos);
      updateCenter(new_pos, index, radius);
      OctreeInternal* nnode = new OctreeInternal(new_pos);
      gptr<OctreeInternal> new_node(nnode);
      gptr<Octree> new_octreenode(reinterpret_cast<Octree*>(nnode));

      assert(n->pos != b->pos);
      
      insert(b, new_node, radius);
      insert(n, new_node, radius);
      node->child[index] = new_octreenode;
    } else {
      OctreeInternal* ni = reinterpret_cast<OctreeInternal*>(child.getptr());
      gptr<OctreeInternal> nni;
      nni.initialize(ni,child.getowner());
      insert(b, nni, radius);
    }
  }
};

struct ComputeCenterOfMass {
  // NB: only correct when run sequentially or tree-like reduction
  typedef int tt_does_not_need_stats;
  gptr<OctreeInternal> root;

  ComputeCenterOfMass() { }
  ComputeCenterOfMass(gptr<OctreeInternal> _root) : root(_root) { }

  void operator()() {
    root->mass = recurse(root);
  }

private:
  double recurse(gptr<OctreeInternal>& node) {
    double mass = 0.0;
    int index = 0;
    Point accum;

    for (int i = 0; i < 8; i++) {
      gptr<Octree> child = node->child[i];
      if (!child)
        continue;

      // Reorganize leaves to be denser up front 
      if (index != i) {
	gptr<Octree> empty;
        node->child[index] = child;
        node->child[i] = empty;
      }
      index++;
      
      double m;
      const Point* p = &child->pos;
      if (child->Leaf) {
        m = child->mass;
      } else {
        // FIX THIS!
        gptr<OctreeInternal> tmp;
        tmp.initialize(static_cast<OctreeInternal*>(child.getptr()),child.getowner());
        m = recurse(tmp);
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

    return mass;
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

void computeDelta(Point& p, const Body* body, Octree* b) {
  for (int i = 0; i < 3; i++)
    p[i] = b->pos[i] - body->pos[i];
}

struct ComputeForces : Galois::Runtime::Lockable {
  // Optimize runtime for no conflict case
  typedef int tt_does_not_need_aborts;
  typedef int tt_needs_per_iter_alloc;

  gptr<OctreeInternal> top;
  double diameter;
  double root_dsq;

  ComputeForces() { }
  ComputeForces(gptr<OctreeInternal>& _top, double _diameter) :
    top(_top),
    diameter(_diameter) {
    root_dsq = diameter * diameter * config.itolsq;
  }
  
  template<typename Context>
  void operator()(Body* bb, Context& cnx) {
    Body& b = *bb;
    Point p = b.acc;
    for (int i = 0; i < 3; i++)
      b.acc[i] = 0;
    //recurse(b, top, root_dsq);
    iterate(b, root_dsq, cnx);
    for (int i = 0; i < 3; i++)
      b.vel[i] += (b.acc[i] - p[i]) * config.dthf;
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,top,diameter,root_dsq);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,top,diameter,root_dsq);
  }

  void forleaf(Body& __restrict__  b, Body* __restrict__ node, double dsq) {
    //check if it is me
    if (&b == node)
      return;
    Point p;
    computeDelta(p, &b, node);
    updateForce(b.acc, p, p.dist2(), b.mass);
  }

  struct Frame {
    double dsq;
    OctreeInternal* node;
    Frame(OctreeInternal* _node, double _dsq) : dsq(_dsq), node(_node) { }
  };

  template<typename Context>
  void iterate(Body& b, double root_dsq, Context& cnx) {
    std::deque<Frame, Galois::PerIterAllocTy::rebind<Frame>::other> stack(cnx.getPerIterAlloc());
    stack.push_back(Frame(&(*top), root_dsq));

    Point p;
    while (!stack.empty()) {
      Frame f = stack.back();
      stack.pop_back();

      computeDelta(p, &b, f.node);
      double psq = p.dist2();

      // Node is far enough away, summarize contribution
      if (psq >= f.dsq) {
        updateForce(b.acc, p, psq, f.node->mass);
        continue;
      }

      double dsq = f.dsq * 0.25;
      
      for (int i = 0; i < 8; i++) {
        gptr<Octree> next = f.node->child[i];
        if (!next)
          break;
        if (next->Leaf) {
	  forleaf(b, static_cast<Body*>(&(*next)), dsq);
	} else {
          stack.push_back(Frame(static_cast<OctreeInternal*>(&(*next)), dsq));
        }
      }
    }
  }

  void recurse(Body& b, OctreeInternal* node, double dsq) {
    Point p;
    computeDelta(p, &b, node);
    double psq = p.dist2();
    // Node is far enough away, summarize contribution
    if (psq >= dsq) {
      updateForce(b.acc, p, psq, node->mass);
      
      return;
    }

    dsq *= 0.25;
    
    for (int i = 0; i < 8; i++) {
      gptr<Octree> next = node->child[i];
      if (!next)
        break;
      if (next->Leaf) {
	forleaf(b, static_cast<Body*>(&(*next)), dsq);
      } else {
        recurse(b, static_cast<OctreeInternal*>(&(*next)), dsq);
      }
    }
  }
};

struct AdvanceBodies {
  // Optimize runtime for no conflict case
  typedef int tt_does_not_need_aborts;

  AdvanceBodies() { }

  template<typename Context>
  void operator()(Body* bb, Context&) {
    operator()(bb);
  }

  void operator()(Body* bb) {
    Body& b = *bb;
    Point dvel(b.acc);
    dvel *= config.dthf;

    Point velh(b.vel);
    velh += dvel;

    for (int i = 0; i < 3; ++i)
      b.pos[i] += velh[i] * config.dtime;
    for (int i = 0; i < 3; ++i)
      b.vel[i] = velh[i] + dvel[i];
  }
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const { }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) { }
};

struct ReduceBoxes : public Galois::Runtime::Lockable {
  // NB: only correct when run sequentially or tree-like reduction
  typedef int tt_does_not_need_stats;
  gptr<BoundingBox> initial;

  ReduceBoxes() { }
  ReduceBoxes(gptr<BoundingBox>& _initial): initial(_initial) { }

  template<typename Context>
  void operator()(Body* b, const Context& cnx) {
    assert(b);
    (*initial).merge(b->pos);
  }
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,initial);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,initial);
  }
};

double nextDouble() {
  return rand() / (double) RAND_MAX;
}

struct InsertBody : public Galois::Runtime::Lockable {
  BodyPtrs pBodies;
  Bodies bodies;
  InsertBody() { }
  InsertBody(BodyPtrs& pb, Bodies& b): pBodies(pb), bodies(b) { }
  template<typename Context>
  void operator()(Body& b, const Context& cnx) {
    bodies->push_back(b);
    pBodies->push_back(&bodies->back());
  }
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,pBodies);
    gSerialize(s,bodies);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,pBodies);
    gDeserialize(s,bodies);
  }
};

struct centerXCmp {
  template<typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return lhs.pos[0] < rhs.pos[0];
  }
};

struct centerYCmp {
  template<typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return lhs.pos[1] < rhs.pos[1];
  }
};

struct centerYCmpInv {
  template<typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return rhs.pos[1] < lhs.pos[1];
  }
};


template<typename Iter>
void divide(const Iter& b, const Iter& e) {
  if (std::distance(b,e) > 32) {
    std::sort(b,e, centerXCmp());
    Iter m = Galois::split_range(b,e);
    std::sort(b,m, centerYCmpInv());
    std::sort(m,e, centerYCmp());
    divide(b, Galois::split_range(b,m));
    divide(Galois::split_range(b,m), m);
    divide(m,Galois::split_range(m,e));
    divide(Galois::split_range(m,e), e);
  } else {
    std::random_shuffle(b,e);
  }
}

/**
 * Generates random input according to the Plummer model, which is more
 * realistic but perhaps not so much so according to astrophysicists
 */
void generateInput(Bodies& bodies, BodyPtrs& pBodies, int nbodies, int seed) {
  double v, sq, scale;
  Point p;
  double PI = boost::math::constants::pi<double>();

  srand(seed);

  double rsc = (3 * PI) / 16;
  double vsc = sqrt(1.0 / rsc);

  std::vector<Body> tmp;

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
    for (int i = 0; i < 3; i++)
      b.vel[i] = p[i] * scale;

    tmp.push_back(b);
    //pBodies.push_back(&bodies.push_back(b));
  }

  //sort and copy out
  divide(tmp.begin(), tmp.end());
  Galois::for_each<>(tmp.begin(), tmp.end(), InsertBody(pBodies, bodies));
}

struct CheckAllPairs {
  Bodies& bodies;
  
  CheckAllPairs(Bodies& b): bodies(b) { }

  double operator()(const Body& body) {
    const Body* me = &body;
    Point acc;
    for (auto ii = bodies->local_begin(), ei = bodies->local_end(); ii != ei; ++ii) {
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

/*
double checkAllPairs(Bodies& bodies, int N) {
  BodyBag::local_iterator end(bodies->local_begin());
  std::advance(end, N);
  
  return Galois::ParallelSTL::map_reduce(bodies.begin(), end,
      CheckAllPairs(bodies),
      0.0,
      std::plus<double>()) / N;
}
*/

void run(Bodies& bodies, BodyPtrs& pBodies) {
  typedef Galois::WorkList::dChunkedLIFO<256> WL_;
  typedef Galois::WorkList::ChunkedAdaptor<false,32> WL;

  for (int step = 0; step < ntimesteps; step++) {
    // Do tree building sequentially
    gptr<BoundingBox> box(new BoundingBox());
    Galois::for_each_local<>(pBodies, ReduceBoxes(box));
    OctreeInternal top(box->center());
    gptr<OctreeInternal> gtop(&top);

    Galois::for_each_local<>(pBodies, BuildOctree(gtop, box->radius()));

    ComputeCenterOfMass computeCenterOfMass(gtop);
    computeCenterOfMass();

    Galois::StatTimer T_parallel("ParallelTime");
    T_parallel.start();

    Galois::for_each_local<WL>(pBodies, ComputeForces(gtop, box->diameter()), "compute");
    if (!skipVerify) {
      //std::cout << "MSE (sampled) " << checkAllPairs(bodies, std::min((int) nbodies, 100)) << "\n";
    }
    Galois::for_each_local<>(pBodies, AdvanceBodies());//, "advance");
    T_parallel.stop();

    std::cout << "Timestep " << step << " Center of Mass = ";
    std::ios::fmtflags flags = 
      std::cout.setf(std::ios::showpos|std::ios::right|std::ios::scientific|std::ios::showpoint);
    std::cout << top.pos;
    std::cout.flags(flags);
    std::cout << "\n";
  }
}

int main(int argc, char** argv) {
  Galois::StatManager M;
  LonestarStart(argc, argv, name, desc, url);
  Galois::Runtime::Distributed::networkStart();

  std::cout << config << "\n";
  std::cout << nbodies << " bodies, "
            << ntimesteps << " time steps\n";

  Bodies bodies = Galois::Graph::Bag<Body>::allocate();
  BodyPtrs pBodies = Galois::Graph::Bag<Body*>::allocate();
  generateInput(bodies, pBodies, nbodies, seed);

  Galois::StatTimer T;
  T.start();
  run(bodies, pBodies);
  T.stop();
  Galois::Runtime::Distributed::networkTerminate();
  return 0;
}
