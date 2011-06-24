/** Barnes-hut application -*- C++ -*-
 *
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
#include <limits>
#include <iostream>
#include <vector>
#include <boost/math/constants/constants.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include "Galois/Galois.h"
#include "Galois/Launcher.h"
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

static const char* name = "Barnshut N-Body Simulator";
static const char* description =
  "Simulation of the gravitational forces in a galactic cluster using the "
  "Barnes-Hut n-body algorithm\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/barneshut.html";
static const char* help = "<numbodies> <ntimesteps> <seed>";

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
  }

  double& operator[](const int index) {
    switch (index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
    }
    assert(false && "index out of bounds");
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
};

std::ostream& operator<<(std::ostream& os, const Point& p) {
  os << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
  return os;
}

/**
 * A node in an octree is either an internal node or a body (leaf).
 */
struct Octree {
  virtual ~Octree() { }
  virtual bool isLeaf() const = 0;
};

struct OctreeInternal : Octree {
  Octree* child[8];
  Point pos;
  double mass;
  OctreeInternal(Point _pos) : pos(_pos), mass(0.0) {
    bzero(child, sizeof(*child) * 8);
  }
  virtual ~OctreeInternal() {
    for (int i = 0; i < 8; i++) {
      if (child[i] != NULL && !child[i]->isLeaf()) {
        delete child[i];
      }
    }
  }
  virtual bool isLeaf() const {
    return false;
  }
};

struct Body : Octree {
  Point pos;
  Point vel;
  Point acc;
  double mass;
  Body() { }
  virtual bool isLeaf() const {
    return true;
  }
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

static inline int getIndex(const Point& a, const Point& b) {
  int index = 0;
  if (a.x < b.x)
    index += 1;
  if (a.y < b.y)
    index += 2;
  if (a.z < b.z)
    index += 4;
  return index;
}

static inline void updateCenter(Point& p, int index, double radius) {
  for (int i = 0; i < 3; i++) {
    double v = (index & (1 << i)) > 0 ? radius : -radius;
    p[i] += v;
  }
}

typedef std::vector<Body> Bodies;

struct BuildOctree {
  // TODO sequential trait
  OctreeInternal* root;
  double root_radius;

  BuildOctree(OctreeInternal* _root, double radius) :
    root(_root),
    root_radius(radius) { }

  template<typename Context>
  void operator()(Body* b, Context&) {
    insert(b, root, root_radius);
  }

  void insert(Body* b, OctreeInternal* node, double radius) {
    int index = getIndex(node->pos, b->pos);

    assert(!node->isLeaf());

    Octree *child = node->child[index];
    
    if (child == NULL) {
      node->child[index] = b;
      return;
    }
    
    radius *= 0.5;
    if (child->isLeaf()) {
      // Expand leaf
      Body* n = static_cast<Body*>(child);
      Point new_pos(node->pos);
      updateCenter(new_pos, index, radius);
      OctreeInternal* new_node = new OctreeInternal(new_pos);

      assert(n->pos != b->pos);
      
      insert(b, new_node, radius);
      insert(n, new_node, radius);
      node->child[index] = new_node;
    } else {
      OctreeInternal* n = static_cast<OctreeInternal*>(child);
      insert(b, n, radius);
    }
  }
};

struct ComputeCenterOfMass {
  // TODO tree trait
  OctreeInternal* root;

  ComputeCenterOfMass(OctreeInternal* _root) : root(_root) { }

  void operator()() {
    root->mass = recurse(root);
  }

private:
  double recurse(OctreeInternal* node) {
    double mass = 0.0;
    int index = 0;
    Point accum;
    
    for (int i = 0; i < 8; i++) {
      Octree* child = node->child[i];
      if (child == NULL)
        continue;

      // Reorganize leaves to be denser up front 
      if (index != i) {
        node->child[index] = child;
        node->child[i] = NULL;
      }
      index++;
      
      double m;
      const Point* p;
      if (child->isLeaf()) {
        Body* n = static_cast<Body*>(child);
        m = n->mass;
        p = &n->pos;
      } else {
        OctreeInternal* n = static_cast<OctreeInternal*>(child);
        m = recurse(n);
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

    return mass;
  }
};

struct ComputeForces {
  // TODO no conflict trait
  OctreeInternal* top;
  double diameter;
  double root_dsq;
  int step;

  ComputeForces(OctreeInternal* _top, double _diameter, int _step) :
    top(_top),
    diameter(_diameter),
    step(_step) {
    root_dsq = diameter * diameter * config.itolsq;
  }
  
  template<typename Context>
  void operator()(Body* bb, Context&) {
    Body& b = *bb;
    Point p = b.acc;
    for (int i = 0; i < 3; i++)
      b.acc[i] = 0;
    //recurse(b, top, root_dsq);
    iterate(b, root_dsq);
    if (step > 0) {
      for (int i = 0; i < 3; i++)
        b.vel[i] += (b.acc[i] - p[i]) * config.dthf;
    }
  }

  void recurse(Body& b, Body* node, double dsq) {
    Point p;
    for (int i = 0; i < 3; i++)
      p[i] = node->pos[i] - b.pos[i];

    double psq = p.x * p.x + p.y * p.y + p.z * p.z;
    psq += config.epssq;
    double idr = 1 / sqrt(psq);
    // b.mass is fine because every body has the same mass
    double nphi = b.mass * idr;
    double scale = nphi * idr * idr;
    for (int i = 0; i < 3; i++) 
      b.acc[i] += p[i] * scale;
  }

  struct Frame {
    double dsq;
    OctreeInternal* node;
    Frame(OctreeInternal* _node, double _dsq) : dsq(_dsq), node(_node) { }
  };

  void iterate(Body& b, double root_dsq) {
    std::vector<Frame> stack;
    stack.push_back(Frame(top, root_dsq));

    Point p;
    while (!stack.empty()) {
      Frame f = stack.back();
      stack.pop_back();

      for (int i = 0; i < 3; i++)
        p[i] = f.node->pos[i] - b.pos[i];

      double psq = p.x * p.x + p.y * p.y + p.z * p.z;
      if (psq >= f.dsq) {
        // Node is far enough away, summarize contribution
        psq += config.epssq;
        double idr = 1 / sqrt(psq);
        double nphi = f.node->mass * idr;
        double scale = nphi * idr * idr;
        for (int i = 0; i < 3; i++) 
          b.acc[i] += p[i] * scale;
        
        continue;
      }

      double dsq = f.dsq * 0.25;
      
      for (int i = 0; i < 8; i++) {
        Octree *next = f.node->child[i];
        if (next == NULL)
          break;
        if (next->isLeaf()) {
          // Check if it is me
          if (&b != next) {
            recurse(b, static_cast<Body*>(next), dsq);
          }
        } else {
          stack.push_back(Frame(static_cast<OctreeInternal*>(next), dsq));
        }
      }
    }
  }

  void recurse(Body& b, OctreeInternal* node, double dsq) {
    Point p;

    for (int i = 0; i < 3; i++)
      p[i] = node->pos[i] - b.pos[i];
    double psq = p.x * p.x + p.y * p.y + p.z * p.z;
    if (psq >= dsq) {
      // Node is far enough away, summarize contribution
      psq += config.epssq;
      double idr = 1 / sqrt(psq);
      double nphi = node->mass * idr;
      double scale = nphi * idr * idr;
      for (int i = 0; i < 3; i++) 
        b.acc[i] += p[i] * scale;
      
      return;
    }

    dsq *= 0.25;
    
    for (int i = 0; i < 8; i++) {
      Octree *next = node->child[i];
      if (next == NULL)
        break;
      if (next->isLeaf()) {
        // Check if it is me
        if (&b != next) {
          recurse(b, static_cast<Body*>(next), dsq);
        }
      } else {
        recurse(b, static_cast<OctreeInternal*>(next), dsq);
      }
    }
  }
};

struct AdvanceBodies {
  // TODO no conflict trait
  AdvanceBodies() { }

  template<typename Context>
  void operator()(Body* bb, Context&) {
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
};

struct ReduceBoxes {
  // TODO reduction operator
  BoundingBox& initial;

  ReduceBoxes(BoundingBox& _initial): initial(_initial) { }

  template<typename Context>
  void operator()(Body* b, Context&) {
    initial.merge(b->pos);
  }
};

static double nextDouble() {
  return rand() / (double) RAND_MAX;
}

/**
 * Generates random input according to the Plummer model, which is more
 * realistic but perhaps not so much so according to astrophysicists
 */
static void generateInput(Bodies& bodies, int nbodies, int seed) {
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
    for (int i = 0; i < 3; i++)
      b.vel[i] = p[i] * scale;

    bodies.push_back(b);
  }
}

template<typename T>
struct Deref : public std::unary_function<T, T*> {
  T* operator()(T& item) const { return &item; }
};

static boost::transform_iterator<Deref<Body>, Bodies::iterator> 
wrap(Bodies::iterator it) {
  return boost::make_transform_iterator(it, Deref<Body>());
}

static void run(int nbodies, int ntimesteps, int seed) {
  Bodies bodies;
  generateInput(bodies, nbodies, seed);

  typedef GaloisRuntime::WorkList::dChunkedLIFO<256> WL;

  for (int step = 0; step < ntimesteps; step++) {
    BoundingBox box;
    ReduceBoxes reduceBoxes(box);
    Galois::for_each<WL>(wrap(bodies.begin()), wrap(bodies.end()),
        ReduceBoxes(box));
    OctreeInternal* top = new OctreeInternal(box.center());

    Galois::for_each<WL>(wrap(bodies.begin()), wrap(bodies.end()),
        BuildOctree(top, box.radius()));

    ComputeCenterOfMass computeCenterOfMass(top);
    computeCenterOfMass();

    Galois::for_each<WL>(wrap(bodies.begin()), wrap(bodies.end()),
        ComputeForces(top, box.diameter(), step));
    Galois::for_each<WL>(wrap(bodies.begin()), wrap(bodies.end()),
        AdvanceBodies());

    std::cout 
      << "Timestep " << step
      << " Center of Mass = " << top->pos << "\n";
    delete top;
  }
}

int main(int argc, const char** argv) {
  std::cout.setf(std::ios::right|std::ios::scientific|std::ios::showpoint);

  std::vector<const char*> args = parse_command_line(argc, argv, help);
  if (args.size() != 3) {
    std::cerr << "not enough arguments, use -help for usage information\n";
    exit(1);
  }
  int nbodies = atoi(args[0]);
  int ntimesteps = atoi(args[1]);
  int seed = atoi(args[2]);

  printBanner(std::cout, name, description, url);
  std::cerr << "configuration: "
            << nbodies << " bodies, "
            << ntimesteps << " time steps" << std::endl << std::endl;
  std::cout << "Num. of threads: " << numThreads << std::endl;

  // TODO(ddn): broken for numThreads > 1 until we get the sequential
  // annotations in
  if (numThreads > 1) {
    assert(numThreads == 1);
    abort();
  }

  Galois::Launcher::startTiming();
  run(nbodies, ntimesteps, seed);
  Galois::Launcher::stopTiming();
}
