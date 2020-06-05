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

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/Bag.h"
#include "galois/Reduction.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/Profile.h"

#include <boost/math/constants/constants.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <array>
#include <limits>
#include <iostream>
#include <fstream>
#include <random>
#include <deque>

#include <strings.h>

#include "Point.h"

const char* name = "Barnes-Hut N-Body Simulator";
const char* desc =
    "Simulates gravitational forces in a galactic cluster using the "
    "Barnes-Hut n-body algorithm";
const char* url = "barneshut";

static llvm::cl::opt<int>
    nbodies("n", llvm::cl::desc("Number of bodies (default value 10000)"),
            llvm::cl::init(10000));
static llvm::cl::opt<int>
    ntimesteps("steps", llvm::cl::desc("Number of steps (default value 1)"),
               llvm::cl::init(1));
static llvm::cl::opt<int> seed("seed",
                               llvm::cl::desc("Random seed (default value 7)"),
                               llvm::cl::init(7));

struct Node {
  Point pos;
  double mass;
  bool Leaf;
};

struct Body : public Node {
  Point vel;
  Point acc;
};

/**
 * A node in an octree is either an internal node or a leaf.
 */
struct Octree : public Node {
  std::array<galois::substrate::PtrLock<Node>, 8> child;
  char cLeafs;
  char nChildren;

  Octree(const Point& p) {
    Node::pos  = p;
    Node::Leaf = false;
    cLeafs     = 0;
    nChildren  = 0;
  }
};

std::ostream& operator<<(std::ostream& os, const Body& b) {
  os << "(pos:" << b.pos << " vel:" << b.vel << " acc:" << b.acc
     << " mass:" << b.mass << ")";
  return os;
}

struct BoundingBox {
  Point min;
  Point max;
  explicit BoundingBox(const Point& p) : min(p), max(p) {}
  BoundingBox()
      : min(std::numeric_limits<double>::max()),
        max(std::numeric_limits<double>::min()) {}

  BoundingBox merge(const BoundingBox& other) const {
    BoundingBox copy(*this);

    copy.min.pairMin(other.min);
    copy.max.pairMax(other.max);
    return copy;
  }

  double diameter() const { return (max - min).minDim(); }
  double radius() const { return diameter() * 0.5; }
  Point center() const { return (min + max) * 0.5; }
};

std::ostream& operator<<(std::ostream& os, const BoundingBox& b) {
  os << "(min:" << b.min << " max:" << b.max << ")";
  return os;
}

struct Config {
  const double dtime; // length of one time step
  const double eps;   // potential softening parameter
  const double tol;   // tolerance for stopping recursion, <0.57 to bound error
  const double dthf, epssq, itolsq;
  Config()
      : dtime(0.5), eps(0.05), tol(0.05), // 0.025),
        dthf(dtime * 0.5), epssq(eps * eps), itolsq(1.0 / (tol * tol)) {}
};

std::ostream& operator<<(std::ostream& os, const Config& c) {
  os << "Barnes-Hut configuration:"
     << " dtime: " << c.dtime << " eps: " << c.eps << " tol: " << c.tol;
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

inline Point updateCenter(Point v, int index, double radius) {
  for (int i = 0; i < 3; i++)
    v[i] += (index & (1 << i)) > 0 ? radius : -radius;
  return v;
}

typedef galois::InsertBag<Body> Bodies;
typedef galois::InsertBag<Body*> BodyPtrs;
// FIXME: reclaim memory for multiple steps
typedef galois::InsertBag<Octree> Tree;

struct BuildOctree {

  Tree& T;

  void insert(Body* b, Octree* node, double radius) const {
    int index   = getIndex(node->pos, b->pos);
    Node* child = node->child[index].getValue();

    // go through the tree lock-free while we can
    if (child && !child->Leaf) {
      insert(b, static_cast<Octree*>(child), radius);
      return;
    }

    node->child[index].lock();
    child = node->child[index].getValue();

    if (child == NULL) {
      node->child[index].unlock_and_set(b);
      return;
    }

    radius *= 0.5;
    if (child->Leaf) {
      // Expand leaf

      Octree* new_node = &T.emplace(updateCenter(node->pos, index, radius));
      if (b->pos == child->pos) {
        // Jitter point to gaurantee uniqueness.
        double jitter = config.tol / 2;
        assert(jitter < radius);
        b->pos += (new_node->pos - b->pos) * jitter;
      }

      // assert(node->pos != b->pos);
      // node->child[index].unlock_and_set(new_node);
      insert(b, new_node, radius);
      insert(static_cast<Body*>(child), new_node, radius);
      node->child[index].unlock_and_set(new_node);
    } else {
      node->child[index].unlock();
      insert(b, static_cast<Octree*>(child), radius);
    }
  }
};

unsigned computeCenterOfMass(Octree* node) {
  double mass = 0.0;
  Point accum;
  unsigned num = 1;

  // Reorganize leaves to be dense
  // remove copies values
  int index = 0;
  for (int i = 0; i < 8; ++i)
    if (node->child[i].getValue())
      node->child[index++].setValue(node->child[i].getValue());
  for (int i = index; i < 8; ++i)
    node->child[i].setValue(NULL);
  node->nChildren = index;

  for (int i = 0; i < index; i++) {
    Node* child = node->child[i].getValue();
    if (!child->Leaf) {
      num += computeCenterOfMass(static_cast<Octree*>(child));
    } else {
      node->cLeafs |= (1 << i);
      ++num;
    }
    mass += child->mass;
    accum += child->pos * child->mass;
  }

  node->mass = mass;

  if (mass > 0.0)
    node->pos = accum / mass;
  return num;
}

/*
void printRec(std::ofstream& file, Node* node, unsigned level) {
  static const char* ct[] = {
    "blue", "cyan", "aquamarine", "chartreuse",
    "darkorchid", "darkorange",
    "deeppink", "gold", "chocolate"
  };
  if (!node) return;
  file << "\"" << node << "\" [color=" << ct[node->owner / 4] << (node->owner %
4 + 1) << (level ? "" : " style=filled") << " label = \"" << (node->Leaf ? "L" :
"N") << "\"];\n"; if (!node->Leaf) { Octree* node2 = static_cast<Octree*>(node);
    for (int i = 0; i < 8 && node2->child[i]; ++i) {
      if (level == 3 || level == 6)
        file << "subgraph cluster_" << level << "_" << i << " {\n";
      file << "\"" << node << "\" -> \"" << node2->child[i] << "\"
[weight=0.01]\n"; printRec(file, node2->child[i], level + 1); if (level == 3 ||
level == 6) file << "}\n";
    }
  }
}

void printTree(Octree* node) {
  std::ofstream file("out.txt");
  file << "digraph octree {\n";
  file << "ranksep = 2\n";
  file << "root = \"" << node << "\"\n";
  //  file << "overlap = scale\n";
  printRec(file, node, 0);
  file << "}\n";
}
*/

Point updateForce(Point delta, double psq, double mass) {
  // Computing force += delta * mass * (|delta|^2 + eps^2)^{-3/2}
  double idr   = 1 / sqrt((float)(psq + config.epssq));
  double scale = mass * idr * idr * idr;
  return delta * scale;
}

struct ComputeForces {
  // Optimize runtime for no conflict case

  Octree* top;
  double root_dsq;

  ComputeForces(Octree* _top, double diameter) : top(_top) {
    assert(diameter > 0.0 && "non positive diameter of bb");
    root_dsq = diameter * diameter * config.itolsq;
  }

  template <typename Context>
  void computeForce(Body* b, Context& cnx) {
    Point p = b->acc;
    b->acc  = Point(0.0, 0.0, 0.0);
    iterate(*b, cnx);
    b->vel += (b->acc - p) * config.dthf;
  }

  struct Frame {
    double dsq;
    Octree* node;
    Frame(Octree* _node, double _dsq) : dsq(_dsq), node(_node) {}
  };

  template <typename Context>
  void iterate(Body& b, Context& cnx) {
    std::deque<Frame, galois::PerIterAllocTy::rebind<Frame>::other> stack(
        cnx.getPerIterAlloc());
    stack.push_back(Frame(top, root_dsq));

    while (!stack.empty()) {
      const Frame f = stack.back();
      stack.pop_back();

      Point p    = b.pos - f.node->pos;
      double psq = p.dist2();

      // Node is far enough away, summarize contribution
      if (psq >= f.dsq) {
        b.acc += updateForce(p, psq, f.node->mass);
        continue;
      }

      double dsq = f.dsq * 0.25;
      for (int i = 0; i < f.node->nChildren; i++) {
        Node* n = f.node->child[i].getValue();
        assert(n);
        if (f.node->cLeafs & (1 << i)) {
          assert(n->Leaf);
          if (static_cast<const Node*>(&b) != n) {
            Point p = b.pos - n->pos;
            b.acc += updateForce(p, p.dist2(), n->mass);
          }
        } else {
#ifndef GALOIS_CXX11_DEQUE_HAS_NO_EMPLACE
          stack.emplace_back(static_cast<Octree*>(n), dsq);
#else
          stack.push_back(Frame(static_cast<Octree*>(n), dsq));
#endif
          __builtin_prefetch(n);
        }
      }
    }
  }
};

struct centerXCmp {
  template <typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return lhs.pos[0] < rhs.pos[0];
  }
};

struct centerYCmp {
  template <typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return lhs.pos[1] < rhs.pos[1];
  }
};

struct centerYCmpInv {
  template <typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return rhs.pos[1] < lhs.pos[1];
  }
};

template <typename Iter, typename Gen>
void divide(const Iter& b, const Iter& e, Gen& gen) {
  if (std::distance(b, e) > 32) {
    std::sort(b, e, centerXCmp());
    Iter m = galois::split_range(b, e);
    std::sort(b, m, centerYCmpInv());
    std::sort(m, e, centerYCmp());
    divide(b, galois::split_range(b, m), gen);
    divide(galois::split_range(b, m), m, gen);
    divide(m, galois::split_range(m, e), gen);
    divide(galois::split_range(m, e), e, gen);
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

  std::vector<Body> tmp;

  for (int body = 0; body < nbodies; body++) {
    double r = 1.0 / sqrt(pow(dist(gen) * 0.999, -2.0 / 3.0) - 1);
    do {
      for (int i = 0; i < 3; i++)
        p[i] = dist(gen) * 2.0 - 1.0;
      sq = p.dist2();
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);

    Body b;
    b.mass = 1.0 / nbodies;
    b.pos  = p * scale;
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
    scale  = vsc * v / sqrt(sq);
    b.vel  = p * scale;
    b.Leaf = true;
    tmp.push_back(b);
    // pBodies.push_back(&bodies.push_back(b));
  }

  // sort and copy out
  divide(tmp.begin(), tmp.end(), gen);

  galois::do_all(
      galois::iterate(tmp),
      [&pBodies, &bodies](const Body& b) {
        pBodies.push_back(&(bodies.push_back(b)));
      },
      galois::loopname("InsertBody"));
}

struct CheckAllPairs {
  Bodies& bodies;

  CheckAllPairs(Bodies& b) : bodies(b) {}

  double operator()(const Body& body) const {
    const Body* me = &body;
    Point acc;
    for (Bodies::iterator ii = bodies.begin(), ei = bodies.end(); ii != ei;
         ++ii) {
      Body* b = &*ii;
      if (me == b)
        continue;
      Point delta = me->pos - b->pos;
      double psq  = delta.dist2();
      acc += updateForce(delta, psq, b->mass);
    }

    double dist2 = acc.dist2();
    acc -= me->acc;
    double retval = acc.dist2() / dist2;
    return retval;
  }
};

double checkAllPairs(Bodies& bodies, int N) {
  Bodies::iterator end(bodies.begin());
  std::advance(end, N);

  return galois::ParallelSTL::map_reduce(bodies.begin(), end,
                                         CheckAllPairs(bodies),
                                         std::plus<double>(), 0.0) /
         N;
}

void run(Bodies& bodies, BodyPtrs& pBodies, size_t nbodies) {
  typedef galois::worklists::StableIterator<true> WLL;

  galois::preAlloc(galois::getActiveThreads() +
                   (3 * sizeof(Octree) + 2 * sizeof(Body)) * nbodies /
                       galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  for (int step = 0; step < ntimesteps; step++) {

    auto mergeBoxes = [](const BoundingBox& lhs, const BoundingBox& rhs) {
      return lhs.merge(rhs);
    };

    auto identity = []() { return BoundingBox(); };

    // Do tree building sequentially
    auto boxes = galois::make_reducible(mergeBoxes, identity);

    galois::do_all(
        galois::iterate(pBodies),
        [&boxes](const Body* b) { boxes.update(BoundingBox(b->pos)); },
        galois::loopname("reduceBoxes"));

    BoundingBox box = boxes.reduce();

    Tree t;
    BuildOctree treeBuilder{t};
    Octree& top = t.emplace(box.center());

    galois::StatTimer T_build("BuildTime");
    T_build.start();
    galois::do_all(
        galois::iterate(pBodies),
        [&](Body* body) { treeBuilder.insert(body, &top, box.radius()); },
        galois::loopname("BuildTree"));
    T_build.stop();

    // update centers of mass in tree
    galois::timeThis(
        [&](void) {
          unsigned size = computeCenterOfMass(&top);
          // printTree(&top);
          std::cout << "Tree Size: " << size << "\n";
        },
        "summarize-Serial");

    ComputeForces cf(&top, box.diameter());

    galois::StatTimer T_compute("ComputeTime");
    T_compute.start();
    galois::for_each(
        galois::iterate(pBodies),
        [&](Body* b, auto& cnx) { cf.computeForce(b, cnx); },
        galois::loopname("compute"), galois::wl<WLL>(),
        galois::disable_conflict_detection(), galois::no_pushes(),
        galois::per_iter_alloc());
    T_compute.stop();

    if (!skipVerify) {
      galois::timeThis(
          [&](void) {
            std::cout << "MSE (sampled) "
                      << checkAllPairs(bodies, std::min((int)nbodies, 100))
                      << "\n";
          },
          "checkAllPairs");
    }
    // Done in compute forces
    galois::do_all(
        galois::iterate(pBodies),
        [](Body* b) {
          Point dvel(b->acc);
          dvel *= config.dthf;
          Point velh(b->vel);
          velh += dvel;
          b->pos += velh * config.dtime;
          b->vel = velh + dvel;
        },
        galois::loopname("advance"));

    std::cout << "Timestep " << step << " Center of Mass = ";
    std::ios::fmtflags flags =
        std::cout.setf(std::ios::showpos | std::ios::right |
                       std::ios::scientific | std::ios::showpoint);
    std::cout << top.pos;
    std::cout.flags(flags);
    std::cout << "\n";
  }

  galois::reportPageAlloc("MeminfoPost");
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, nullptr);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  std::cout << config << "\n";
  std::cout << nbodies << " bodies, " << ntimesteps << " time steps\n";

  Bodies bodies;
  BodyPtrs pBodies;
  generateInput(bodies, pBodies, nbodies, seed);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  run(bodies, pBodies, nbodies);
  execTime.stop();

  totalTime.stop();

  return 0;
}
