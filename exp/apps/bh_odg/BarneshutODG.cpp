/** Barnes-hut application -*- C++ -*-
 * @file
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
#include <fstream>
#include <vector>
#include <cstring>
#include <sstream>


#include <boost/math/constants/constants.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "Galois/Galois.h"
#include "Galois/GaloisUnsafe.h"
#include "Galois/Atomic.h"
#include "Galois/Statistic.h"
#include "Galois/Runtime/DoAllCoupled.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

namespace {
const char* name = "Barnshut N-Body Simulator";
const char* desc = "Simulates the gravitational forces in a galactic cluster using the "
  "Barnes-Hut n-body algorithm";
const char* url = "barneshut";

const char* const SUMM_SERIAL = "serial";
const char* const SUMM_ODG  = "odg";
const char* const SUMM_LEVEL = "level";

static llvm::cl::opt<int> nbodies("n", llvm::cl::desc("Number of bodies"), llvm::cl::init(10000));
static llvm::cl::opt<int> ntimesteps("steps", llvm::cl::desc("Number of steps"), llvm::cl::init(1));
static llvm::cl::opt<int> seed("seed", llvm::cl::desc("Random seed"), llvm::cl::init(7));

static std::string summHelp = std::string ("tree summarization method: ") + SUMM_SERIAL + "|" + SUMM_ODG + "|" + SUMM_LEVEL;
static llvm::cl::opt<std::string> summOpt("summ", llvm::cl::desc(summHelp.c_str ()), llvm::cl::init(SUMM_SERIAL));

enum TreeSummMethod {
  SERIAL, ODG, LEVEL_BY_LEVEL
};

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

  Octree (): pos (), mass (0.0) {}

  Octree (Point _pos): pos (_pos), mass (0.0) {}

  virtual ~Octree() { }
  virtual bool isLeaf() const = 0;
};

struct OctreeInternal : Octree {
  Octree* child[8];
  OctreeInternal(Point _pos) : Octree (_pos) {
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
  Point vel;
  Point acc;

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

typedef std::vector<Body> Bodies;

struct BuildOctree {
  // NB: only correct when run sequentially
  typedef int tt_does_not_need_stats;

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

struct TreeSummarizeSerial {

  void operator () (OctreeInternal* root) {
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

struct TreeSummarizeODG {

  typedef Galois::GAtomic<unsigned> UnsignedAtomic;
  static const unsigned CHUNK_SIZE = 64;
  typedef Galois::WorkList::dChunkedFIFO<CHUNK_SIZE, unsigned> WLty;

  struct ODGnode {
    UnsignedAtomic numChild;

    OctreeInternal* node;
    unsigned idx;
    unsigned prtidx;

    ODGnode (OctreeInternal* _node, unsigned _idx, unsigned _prtidx) 
      : numChild (0), node (_node), idx (_idx), prtidx (_prtidx)

    {}

  };

  void fillWL (OctreeInternal* root, std::vector<ODGnode>& odgNodes, WLty& wl) const {

    ODGnode root_wrap (root, 0, 0);
    odgNodes.push_back (root_wrap);

    std::deque<unsigned> fifo;
    fifo.push_back (root_wrap.idx);

    unsigned idCntr = 1; // already generated the root;

    while (!fifo.empty ()) {

      unsigned nid = fifo.front ();
      fifo.pop_front ();

      OctreeInternal* node = odgNodes[nid].node;
      assert ((node != NULL) && (!node->isLeaf ()));

      bool allLeaves = true;
      for (unsigned i = 0; i < 8; ++i) {
        if (node->child [i] != NULL) {

          if (!node->child [i]->isLeaf ()) {
            allLeaves = false;

            OctreeInternal* child = static_cast <OctreeInternal*> (node->child [i]);

            ODGnode c_wrap (child, idCntr, nid);
            ++idCntr;

            odgNodes.push_back (c_wrap);
            fifo.push_back (c_wrap.idx);

            // also count the number of children
            ++(odgNodes [nid].numChild);
          }

        }
      }

      if (allLeaves) {
        wl.push (nid);
      }

    }

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void treeCompute (OctreeInternal* node) {

    assert ((node != NULL) && (!node->isLeaf ()));

    unsigned index = 0;
    double massSum = 0.0;
    Point accum;

    for (unsigned i = 0; i < 8; ++i) {
      Octree* child = node->child [i];

      if (child == NULL) {
        continue;
      }

      // compact the children together
      if (index != i) {
        node->child [index] = child;
        node->child [i] = NULL;
      }
      ++index;

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

  struct SummarizeOp {

    std::vector <ODGnode>& odgNodes;

    SummarizeOp (std::vector <ODGnode>& _odgNodes) 
      : odgNodes (_odgNodes)
    {}

    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE static void addToWL (C& lwl, unsigned v) {
      lwl.push (v);
    }

    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void updateODG (unsigned nid, C& lwl) {

      unsigned prtidx = odgNodes[nid].prtidx;

      if (nid != 0) { // not root

        unsigned x = --(odgNodes[prtidx].numChild);

        assert (x < 8);

        if (x == 0) {
          // lwl.push (prtidx);
          addToWL (lwl, prtidx);
        }
      } else {
        assert (nid == prtidx && nid == 0);
      }
    }

    template <typename ContextTy>
    void operator () (unsigned nid, ContextTy& lwl) {
      assert (odgNodes[nid].numChild == 0);

      treeCompute (odgNodes[nid].node);

      updateODG (nid, lwl);
    }

  };



  void operator () (OctreeInternal* root) const {
    WLty wl;
    std::vector<ODGnode> odgNodes;

    Galois::StatTimer t_fill_wl ("Time to fill worklist for tree summarization: ");

    t_fill_wl.start ();
    fillWL (root, odgNodes, wl);
    t_fill_wl.stop ();

    Galois::StatTimer t_feach ("Time taken by for_each in tree summarization");

    t_feach.start ();
    Galois::Runtime::beginSampling ();
    // Galois::for_each_wl<Galois::Runtime::WorkList::ParaMeter<WLty> > (wl, SummarizeOp (odgNodes), "tree_summ");
    Galois::for_each_wl (wl, SummarizeOp (odgNodes), "tree_summ");
    Galois::Runtime::endSampling ();
    t_feach.stop ();

  }

};


struct TreeSummarizeLevelByLevel {

  void fillWL (OctreeInternal* root, std::vector<std::vector<OctreeInternal*> >& levelWL) const {
    levelWL.push_back (std::vector<OctreeInternal*> ());

    levelWL[0].push_back (root);

    unsigned currLevel = 0;

    while (!levelWL[currLevel].empty ()) {
      unsigned nextLevel = currLevel + 1;

      // creating vector for nextLevel
      levelWL.push_back (std::vector<OctreeInternal*> ());

      for (std::vector<OctreeInternal*>::const_iterator i = levelWL[currLevel].begin ()
          , ei = levelWL[currLevel].end (); i != ei; ++i) {

        for (unsigned c = 0; c < 8; ++c) {
          Octree* child = (*i)->child[c];
          if (child != NULL) {

            if (!child->isLeaf ()) {
              levelWL[nextLevel].push_back (static_cast<OctreeInternal*> (child));
            }
          }
        } // for child c

      }

      ++currLevel;
    }

  }


  struct SummarizeOp {

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (OctreeInternal* node) const {
      TreeSummarizeODG::treeCompute (node);
    }

  };



  void operator () (OctreeInternal* root) const {
    const bool USE_PARAMETER = false;


    std::vector<std::vector<OctreeInternal*> > levelWL;


    Galois::StatTimer t_fill_wl ("Time to fill worklist for tree summarization: ");

    t_fill_wl.start ();
    fillWL (root, levelWL);
    t_fill_wl.stop ();


    size_t iter = 0;

    std::ofstream* statsFile = NULL;
    if (USE_PARAMETER) {
      statsFile = new std::ofstream ("parameter_barneshut.csv");
      (*statsFile) << "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE" << std::endl;
    }

    Galois::StatTimer t_feach ("Time taken by for_each in tree summarization");

    t_feach.start ();
    Galois::Runtime::beginSampling ();
    for (unsigned i = levelWL.size (); i > 0;) {
      
      --i; // size - 1

      if (!levelWL[i].empty ()) {
        Galois::Runtime::do_all_coupled (levelWL[i].begin (), levelWL[i].end (),
            SummarizeOp ());


        if (USE_PARAMETER) {
          unsigned step = (levelWL.size () - i - 2);
          (*statsFile) << "tree_summ, " << step << ", " << levelWL[i].size () 
            << ", " << levelWL[i].size () << std::endl;
        }
      }

      iter += levelWL[i].size ();


    }
    Galois::Runtime::endSampling ();
    t_feach.stop ();

    std::cout << "TreeSummarizeLevelByLevel: iterations = " << iter << std::endl;

    if (USE_PARAMETER) {
      delete statsFile;
    }

  }

};


struct ComputeForces {
  // Optimize runtime for no conflict case
  typedef int tt_does_not_need_aborts;

  OctreeInternal* top;
  double diameter;
  double root_dsq;

  ComputeForces(OctreeInternal* _top, double _diameter) :
    top(_top),
    diameter(_diameter) {
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
    for (int i = 0; i < 3; i++)
      b.vel[i] += (b.acc[i] - p[i]) * config.dthf;
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
  // Optimize runtime for no conflict case
  typedef int tt_does_not_need_aborts;

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
  // NB: only correct when run sequentially or tree-like reduction
  typedef int tt_does_not_need_stats;
  BoundingBox& initial;

  ReduceBoxes(BoundingBox& _initial): initial(_initial) { }

  template<typename Context>
  void operator()(Body* b, Context&) {
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
    for (int i = 0; i < 3; i++)
      b.vel[i] = p[i] * scale;

    bodies.push_back(b);
  }
}

template<typename T>
struct Deref : public std::unary_function<T, T*> {
  T* operator()(T& item) const { return &item; }
};

boost::transform_iterator<Deref<Body>, Bodies::iterator> 
wrap(Bodies::iterator it) {
  return boost::make_transform_iterator(it, Deref<Body>());
}

void run(int nbodies, int ntimesteps, int seed, TreeSummMethod summMethod) {
  Bodies bodies;

  Galois::StatTimer t_input_gen ("Time taken by input generation: ");

  t_input_gen.start ();
  generateInput(bodies, nbodies, seed);
  t_input_gen.stop ();

  typedef Galois::WorkList::dChunkedLIFO<256> WL;

  for (int step = 0; step < ntimesteps; step++) {
    // Do tree building sequentially
    Galois::setActiveThreads(1);

    BoundingBox box;
    ReduceBoxes reduceBoxes(box);
    Galois::StatTimer t_bbox ("Time taken by Bounding Box computation: ");


    t_bbox.start ();
    Galois::for_each<WL>(wrap(bodies.begin()), wrap(bodies.end()),
        ReduceBoxes(box));
    t_bbox.stop ();


    OctreeInternal* top = new OctreeInternal(box.center());
    Galois::StatTimer t_tree_build ("Time taken by Octree building: " );

    t_tree_build.start ();
    Galois::for_each<WL>(wrap(bodies.begin()), wrap(bodies.end()),
        BuildOctree(top, box.radius()));
    t_tree_build.stop ();

    // reset the number of threads
    Galois::setActiveThreads(numThreads);


    Galois::StatTimer t_tree_summ ("Time taken by Tree Summarization: ");

    t_tree_summ.start ();

    switch (summMethod) {
      case SERIAL:
        TreeSummarizeSerial computeCenterSerial;
        computeCenterSerial (top);
        break;

      case ODG:
        TreeSummarizeODG computeCenterODG;
        computeCenterODG (top);
        break;

      case LEVEL_BY_LEVEL:
        TreeSummarizeLevelByLevel computeCenterLevel;
        computeCenterLevel (top);
        break;

      default:
        std::abort ();
    }
    t_tree_summ.stop ();


    if (false) { // disabling remaining phases

      Galois::StatTimer T_parallel("ParallelTime");
      T_parallel.start();

      Galois::for_each<WL>(wrap(bodies.begin()), wrap(bodies.end()),
          ComputeForces(top, box.diameter()));
      Galois::for_each<WL>(wrap(bodies.begin()), wrap(bodies.end()),
          AdvanceBodies());
      T_parallel.stop();
    }

    std::cout 
      << "Timestep " << step
      << ", Root's Center of Mass = " << top->pos << std::endl;
    delete top;
  }
}

} // end namespace

int main(int argc, char** argv) {
  Galois::StatManager sm;
  LonestarStart(argc, argv, name, desc, url);

  std::cout.setf(std::ios::right|std::ios::scientific|std::ios::showpoint);

  TreeSummMethod summMethod = SERIAL;

  if (summOpt == SUMM_SERIAL) {
    summMethod = SERIAL;

  } else if (summOpt == SUMM_ODG) {
    summMethod = ODG;

  } else if (summOpt == SUMM_LEVEL) {
    summMethod = LEVEL_BY_LEVEL;

  } else { std::abort (); }

  std::cerr << "configuration: "
            << nbodies << " bodies, "
            << ntimesteps << " time steps" << std::endl << std::endl;
  std::cout << "Num. of threads: " << numThreads << std::endl;

  Galois::StatTimer T;
  T.start();
  run(nbodies, ntimesteps, seed, summMethod);
  T.stop();
}
