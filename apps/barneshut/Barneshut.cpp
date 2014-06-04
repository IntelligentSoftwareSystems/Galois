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
#include "Galois/config.h"
#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Bag.h"
#include "Galois/Graphs/Bag.h"
#include "Galois/Accumulator.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/WorkList/WorkListAlt.h"

#include <boost/math/constants/constants.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include GALOIS_CXX11_STD_HEADER(array)
#include <limits>
#include <iostream>
#include <fstream>
#include <random>
#include GALOIS_CXX11_STD_HEADER(deque)

#include <strings.h>

#include "Point.h"

const char* name = "Barnshut N-Body Simulator";
const char* desc =
  "Simulation of the gravitational forces in a galactic cluster using the "
  "Barnes-Hut n-body algorithm\n";
const char* url = "barneshut";

static llvm::cl::opt<int> nbodies("n", llvm::cl::desc("Number of bodies"), llvm::cl::init(10000));
static llvm::cl::opt<int> ntimesteps("steps", llvm::cl::desc("Number of steps"), llvm::cl::init(1));
static llvm::cl::opt<int> seed("seed", llvm::cl::desc("Random seed"), llvm::cl::init(7));

using Galois::Runtime::gptr;


/**
 * A node in an octree is either an internal node or a body (leaf).
 */
struct Octree : public Galois::Runtime::Lockable {
  Point pos;
  double mass;
  bool Leaf;
  Point vel;
  Point acc;
  gptr<Octree> child[8];
/*
  virtual ~Octree() { }
  SHOULD BE EVENTUALLY DONE
  virtual ~OctreeInternal() {
    for (int i = 0; i < 8; i++) {
      if (OctreeInternal* B = dynamic_cast<OctreeInternal*>(child[i])) {
        delete B;
      }
    }
  }
 */
  Octree(bool l = true) :Leaf(l) {
    bzero(child, sizeof(gptr<Octree>) * 8);
  }
  Octree(const Point& _pos) :Octree(_pos, 0.0, false) {
    bzero(child, sizeof(gptr<Octree>) * 8);
  }
  Octree(const Point& p, double m, bool l) :pos(p), mass(m), Leaf(l) {
    bzero(child, sizeof(gptr<Octree>) * 8);
  }
  //  Octree(const Point& p, double m = 0.0) :pos(p), mass(m) {}
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    Galois::Runtime::gSerialize(s,pos,mass,Leaf);
    Galois::Runtime::gSerialize(s,vel,acc);
    for (int i = 0; i < 8; i++) {
      Galois::Runtime::gSerialize(s,child[i]);
    }
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    Galois::Runtime::gDeserialize(s,pos,mass,Leaf);
    Galois::Runtime::gDeserialize(s,vel,acc);
    for (int i = 0; i < 8; i++) {
      Galois::Runtime::gDeserialize(s,child[i]);
    }
  }
};

std::ostream& operator<<(std::ostream& os, const Octree& b) {
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
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,min,max);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
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
    double diameter = max.x() - min.x();
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
        (max.x() + min.x()) * 0.5,
        (max.y() + min.y()) * 0.5,
        (max.z() + min.z()) * 0.5);
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
    tol(0.05), //0.025),
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
  for (int i = 0; i < 3; ++i)
    if (a[i] < b[i]) 
      index += (1 << i);
  return index;
}

inline void updateCenter(Point& p, int index, double radius) {
  for (int i = 0; i < 3; i++) {
    double v = (index & (1 << i)) > 0 ? radius : -radius;
    p[i] += v;
  }
}

typedef Galois::Graph::Bag<Octree> BodyBag;
typedef Galois::Graph::Bag<Octree>::pointer Bodies;
typedef Galois::Graph::Bag<gptr<Octree> >::pointer BodyPtrs;

struct PrintOctree : public Galois::Runtime::Lockable {
  typedef int tt_does_not_need_stats;
  PrintOctree() { }

  template<typename Context>
  void operator()(gptr<Octree> b, Context& cnx) {
    std::stringstream ss;
    //b.dump(ss);
    ss << " " << *b << " host: " << Galois::Runtime::NetworkInterface::ID << "\n";
    std::cout << ss.str();
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const { }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) { }
};

struct GetBackLocalNodes : public Galois::Runtime::Lockable {
  typedef int tt_does_not_need_stats;
  GetBackLocalNodes() { }

  template<typename Context>
  void operator()(gptr<Octree> b, Context& cnx) {
    std::stringstream ss;
    if (b->Leaf)
      ss << "Leaf Node: ";
    else
      ss << "Internal Node: ";
    //b.dump(ss);
    ss << " " << *b << " host: " << Galois::Runtime::NetworkInterface::ID << "\n";
    // std::cout << ss.str();
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const { }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) { }
};

struct BuildOctree : public Galois::Runtime::Lockable {
  gptr<Octree> root;
  BodyPtrs inNodes;
  double root_radius;

  BuildOctree() { }
  BuildOctree(gptr<Octree> _root, BodyPtrs _in, double radius) :
    root(_root),
    inNodes(_in),
    root_radius(radius) { }

  //template<typename Context>
  //void operator()(gptr<Octree> b, Context& cnx) {
  void operator()(gptr<Octree> b) {
    Octree* troot = Galois::Runtime::transientAcquire(root);
    Octree* tb = Galois::Runtime::transientAcquire(b);
    insert(b, tb, root, troot, root_radius);
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    Galois::Runtime::gSerialize(s,root,inNodes,root_radius);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    Galois::Runtime::gDeserialize(s,root,inNodes,root_radius);
  }

  void insert(gptr<Octree>& b, Octree* tb, gptr<Octree> node, Octree* tnode, double radius) {
    int index = getIndex(tnode->pos, tb->pos);

    gptr<Octree> child = tnode->child[index];
    
    if (!child) {
      tnode->child[index] = b;
      transientRelease(b);
      transientRelease(node);
      return;
    }
    
    radius *= 0.5;
    Octree* tchild = transientAcquire(child);
    if (tchild->Leaf) {
      // Expand leaf
      Point new_pos(tnode->pos);
      updateCenter(new_pos, index, radius);
      Octree* nnode = new Octree(new_pos);
      gptr<Octree> new_node(nnode);
      inNodes->push_back(new_node);

      if (tb->pos == tchild->pos) {
	double jitter = config.tol / 2;
	assert(jitter < radius);
	tb->pos += (nnode->pos - tb->pos) * jitter;
      }
      
      nnode = transientAcquire(new_node);
      insert(b, tb, new_node, nnode, radius);
      // new_node released in the previous insert
      nnode = transientAcquire(new_node);
      insert(child, tchild, new_node, nnode, radius);
      tnode->child[index] = new_node;
      transientRelease(node);
    } else {
      transientRelease(node);
      insert(b, tb, child, tchild, radius);
    }
  }
};

struct ComputeCenterOfMass {
  // NB: only correct when run sequentially or tree-like reduction
  typedef int tt_does_not_need_stats;
  gptr<Octree> root;

  ComputeCenterOfMass() { }
  ComputeCenterOfMass(gptr<Octree> _root) : root(_root) { }

  void operator()() {
    root->mass = recurse(root);
  }

private:
  double recurse(gptr<Octree>& node) {
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
        m = recurse(child);
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

void computeDelta(Point& p, const Octree* body, Octree* b) {
  for (int i = 0; i < 3; i++)
    p[i] = b->pos[i] - body->pos[i];
}

struct ComputeForces : Galois::Runtime::Lockable {
  typedef int tt_needs_per_iter_alloc;

  gptr<Octree> top;
  double diameter;
  double root_dsq;

  ComputeForces() { }
  ComputeForces(gptr<Octree>& _top, double _diameter) :
    top(_top),
    diameter(_diameter) {
    root_dsq = diameter * diameter * config.itolsq;
  }
  
  template<typename Context>
  void operator()(gptr<Octree> bb, Context& cnx) {
    Octree b = *bb;
    Point p = b.acc;
    for (int i = 0; i < 3; i++)
      b.acc[i] = 0;
    //recurse(b, top, root_dsq);
    iterate(b, root_dsq, cnx);
    for (int i = 0; i < 3; i++)
      b.vel[i] += (b.acc[i] - p[i]) * config.dthf;
    *bb = b;
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    Galois::Runtime::gSerialize(s,top,diameter,root_dsq);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,top,diameter,root_dsq);
  }

  void forleaf(Octree& __restrict__  b, Octree* __restrict__ node, double dsq) {
    //check if it is me
    if (&b == node)
      return;
    Point p;
    computeDelta(p, &b, node);
    updateForce(b.acc, p, p.dist2(), b.mass);
  }

  struct Frame {
    double dsq;
    Octree* node;
    Frame(Octree* _node, double _dsq) : dsq(_dsq), node(_node) { }
  };

  template<typename Context>
  void iterate(Octree& b, double root_dsq, Context& cnx) {
    std::deque<Frame, Galois::PerIterAllocTy::rebind<Frame>::other> stack(cnx.getPerIterAlloc());
    stack.push_back(Frame(getSharedObj(top), root_dsq));

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
        Octree* nextp = getSharedObj(next);
        if (nextp->Leaf) {
	  forleaf(b, nextp, dsq);
	} else {
          stack.push_back(Frame(nextp,dsq));
        }
      }
    }
  }

  void recurse(Octree& b, Octree* node, double dsq) {
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
	forleaf(b, &(*next), dsq);
      } else {
        recurse(b, &(*next), dsq);
      }
    }
  }
};

struct AdvanceBodies {

  AdvanceBodies() { }

  template<typename Context>
  void operator()(gptr<Octree>& bb, Context&) {
    operator()(&(*bb));
  }

  void operator()(Octree* bb) {
    Octree& b = *bb;
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
  void serialize(Galois::Runtime::SerializeBuffer& s) const { }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) { }
};

struct ReduceBoxes : public Galois::Runtime::Lockable {
  // NB: only correct when run sequentially or tree-like reduction
  typedef int tt_does_not_need_stats;
  gptr<BoundingBox> initial;

  ReduceBoxes() { }
  ReduceBoxes(gptr<BoundingBox>& _initial): initial(_initial) { }

  template<typename Context>
  void operator()(gptr<Octree> b, const Context& cnx) {
    assert(b);
    (*initial).merge(b->pos);
  }
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    Galois::Runtime::gSerialize(s,initial);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    Galois::Runtime::gDeserialize(s,initial);
  }
};

struct InsertBody : public Galois::Runtime::Lockable {
  BodyPtrs pBodies;
  Bodies bodies;
  InsertBody() { }
  InsertBody(BodyPtrs& pb, Bodies& b): pBodies(pb), bodies(b) { }
  template<typename Context>
  void operator()(Octree& b, const Context& cnx) {
    gptr<Octree> bodyptr(&*bodies->emplace(b));
    pBodies->push_back(bodyptr);
  }
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    Galois::Runtime::gSerialize(s,pBodies);
    Galois::Runtime::gSerialize(s,bodies);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    Galois::Runtime::gDeserialize(s,pBodies);
    Galois::Runtime::gDeserialize(s,bodies);
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


template<typename Iter, typename Gen>
void divide(const Iter& b, const Iter& e, Gen& gen) {
  if (std::distance(b,e) > 32) {
    std::sort(b,e, centerXCmp());
    Iter m = Galois::split_range(b,e);
    std::sort(b, m, centerYCmpInv());
    std::sort(m, e, centerYCmp());
    divide(b, Galois::split_range(b, m), gen);
    divide(Galois::split_range(b, m), m, gen);
    divide(m, Galois::split_range(m, e), gen);
    divide(Galois::split_range(m, e), e, gen);
  } else {
    std::shuffle(b, e, gen);
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

  std::mt19937 gen(seed);
#if __cplusplus >= 201103L || defined(HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
  std::uniform_real_distribution<double> dist(0, 1);
#else
  std::uniform_real<double> dist(0, 1);
#endif

  double rsc = (3 * PI) / 16;
  double vsc = sqrt(1.0 / rsc);

  std::vector<Octree> tmp;

  for (int body = 0; body < nbodies; body++) {
    double r = 1.0 / sqrt(pow(dist(gen) * 0.999, -2.0 / 3.0) - 1);
    do {
      for (int i = 0; i < 3; i++)
        p[i] = dist(gen) * 2.0 - 1.0;
      sq = p.dist2();
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);

    Octree b;
    b.mass = 1.0 / nbodies;
    for (int i = 0; i < 3; i++)
      b.pos[i] = p[i] * scale;

    do {
      p[0] = dist(gen);
      p[1] = dist(gen) * 0.1;
    } while (p[1] > p[0] * p[0] * pow(1 - p[0] * p[0], 3.5));
    v = p[0] * sqrt(2.0 / sqrt(1 + r * r));
    do {
      for (int i = 0; i < 3; i++)
        p[i] = dist(gen) * 2.0 - 1.0;
      sq = p.dist2();
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    for (int i = 0; i < 3; i++)
      b.vel[i] = p[i] * scale;

    tmp.push_back(b);
    //pBodies.push_back(&bodies.push_back(b));
  }

  //sort and copy out
  divide(tmp.begin(), tmp.end(), gen);
  Galois::do_all(tmp.begin(), tmp.end(), InsertBody(pBodies, bodies));
}

struct CheckAllPairs {
  Bodies& bodies;
  
  CheckAllPairs(Bodies& b): bodies(b) { }

  double operator()(const Octree& body) {
    const Octree* me = &body;
    Point acc;
    for (auto ii = bodies->local_begin(), ei = bodies->local_end(); ii != ei; ++ii) {
      Octree* b = &*ii;
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

void run(Bodies& bodies, BodyPtrs& pBodies, BodyPtrs& inNodes) {
  typedef Galois::WorkList::AltChunkedLIFO<32> WL;
  typedef Galois::WorkList::StableIterator<decltype(pBodies.local_begin()), true> WLL;

  Galois::preAlloc (Galois::getActiveThreads () + (3*sizeof (Octree) + 2*sizeof (Body))*nbodies/Galois::Runtime::MM::hugePageSize);
  Galois::reportPageAlloc("MeminfoPre");

  for (int step = 0; step < ntimesteps; step++) {
    gptr<BoundingBox> box(new BoundingBox());

    // Do ReduceBoxes sequentially
    for(auto ii = pBodies->begin(), ee = pBodies->end(); ii != ee; ++ii)
      box->merge((*ii)->pos);

    Octree top(box->center());
    gptr<Octree> gtop(&top);

    inNodes->push_back(gtop);
    //Galois::for_each_local<>(pBodies, BuildOctree(gtop, inNodes, box->radius()));
    std::for_each(pBodies->begin(), pBodies->end(), BuildOctree(gtop, inNodes, box->radius()));

    ComputeCenterOfMass computeCenterOfMass(gtop);
    computeCenterOfMass();

    // return all the local objs sent to other hosts
    Galois::for_each_local<>(inNodes, GetBackLocalNodes());
    Galois::for_each_local<>(pBodies, GetBackLocalNodes());
    clearSharedCache();

    Galois::StatTimer T_parallel("ParallelTime");
    T_parallel.start();

    Galois::for_each_local<WL>(pBodies, ComputeForces(gtop, box->diameter()), "compute");
    if (!skipVerify) {
      //std::cout << "MSE (sampled) " << checkAllPairs(bodies, std::min((int) nbodies, 100)) << "\n";
    }
    Galois::for_each_local<>(pBodies, AdvanceBodies());//, "advance");
    T_parallel.stop();

    clearSharedCache();

    std::cout << "Timestep " << step << " Center of Mass = ";
    std::ios::fmtflags flags = 
      std::cout.setf(std::ios::showpos|std::ios::right|std::ios::scientific|std::ios::showpoint);
    std::cout << gtop->pos;
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

  Bodies bodies = Galois::Graph::Bag<Octree>::allocate();
  BodyPtrs pBodies = Galois::Graph::Bag<gptr<Octree> >::allocate();
  BodyPtrs inNodes = Galois::Graph::Bag<gptr<Octree> >::allocate();
  generateInput(bodies, pBodies, nbodies, seed);

  Galois::StatTimer T;
  T.start();
  run(bodies, pBodies, inNodes);
  T.stop();
  Galois::Runtime::Distributed::networkTerminate();
  return 0;
}
