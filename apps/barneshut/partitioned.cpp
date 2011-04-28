#include <limits>
#include <iostream>
#include <list>
#include <vector>
#include <boost/utility.hpp>
#include <pthread.h>
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Galois.h"
#include <boost/math/constants/constants.hpp>
#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>

//#define VALGRIND 1
#ifdef VALGRIND
#include <stdlib.h>
#endif

template<typename T>
class Bag : boost::noncopyable {
  static const size_t max_page_size = 4096 * 16;
  static const size_t max_items = max_page_size / sizeof(T);
  static const size_t page_size = max_items * sizeof(T);

  struct Page {
    typedef T* iterator;
    iterator begin_ptr;
    iterator end_ptr;
    int refcount;

    Page() : begin_ptr(NULL), end_ptr(NULL), refcount(0) { }
    iterator begin() const { return reinterpret_cast<T*>(begin_ptr); }
    iterator end() const { return reinterpret_cast<T*>(end_ptr); }

    void alloc() {
#ifdef VALGRIND
      void *ptr = malloc(page_size);
#else
      void *ptr = mmap(0, page_size, PROT_READ | PROT_WRITE,
        MAP_HUGETLB | MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
#endif
      begin_ptr = reinterpret_cast<T*>(ptr);
      if (begin_ptr == NULL) 
        assert(false && "failed to mmap memory");

      end_ptr = begin_ptr;
    }

    void dealloc() {
#ifdef VALGRIND
      bzero(begin_ptr, page_size);
      free((void*)begin_ptr);
#else
      munmap(begin_ptr, page_size);
#endif
      end_ptr = begin_ptr = NULL;
    }

    void push_back(T& item) {
      memcpy(end_ptr, &item, sizeof(T));
      end_ptr++;
    }

    bool full() const {
      return end_ptr == begin_ptr + max_items;
    }
  };

  typedef std::list<Page> PartTy;
  typedef GaloisRuntime::PerCPU< PartTy > PartsTy;

  struct PartIterator {
    typedef typename PartTy::const_iterator PartItTy;
    typedef typename Page::iterator PageItTy;
    PartItTy lit, lend;
    PageItTy pit, pend;

    explicit PartIterator(const PartTy& part) {
      lit = part.begin();
      lend = part.end();
      if (lit != lend) {
        pit = lit->begin();
        pend = lit->end();
      } else {
        pit = NULL;
        pend = NULL;
      }
    }
    PartIterator(const PartTy& part, bool end) {
      if (part.empty())
        pit = NULL;
      else
        pit = part.back().end();
    }
    PartIterator& operator++() {
      if (++pit == pend) {
        if (++lit != lend) {
          pit = lit->begin();
          pend = lit->end();
        }
      }
      return *this;
    }
    bool operator!=(const PartIterator& other) {
      return pit != other.pit;
    }
    bool operator==(const PartIterator& other) {
      return !(*this != other);
    }
    T& operator*() {
      return *pit;
    }
  };

  template<typename PartFn>
  struct Partitioner {
    Bag<T>* self;
    PartFn& f;
    PartsTy& parts;
    int numThreads;
    pthread_barrier_t barrier;

    Partitioner(Bag<T>* _self, int _numThreads, PartFn& _f, PartsTy& _parts) : self(_self), f(_f), parts(_parts), numThreads(_numThreads) { 
      int rc = pthread_barrier_init(&barrier, NULL, numThreads);
      assert(rc == 0 && "failed to initialize barrier");
    }

    ~Partitioner() {
      int rc = pthread_barrier_destroy(&barrier);
      assert(rc == 0 && "failed to destroy barrier");
    }
    
    void operator()(unsigned int id) {
      PartTy new_part;
      self->new_page(new_part);

      for (int i = 0; i < numThreads; i++) {
        PartTy& pages = self->parts.get(i);
        for (typename PartTy::iterator it = pages.begin(), end = pages.end(); it != end; ++it) {
          for (typename Page::iterator iit = it->begin(), eend = it->end(); iit != eend; ++iit) {
            T& item = *iit;
            if (id == f(item)) 
              self->push_back(new_part, item);
          }

          if (__sync_add_and_fetch(&it->refcount, 1) == numThreads) {
            it->dealloc();
            it = pages.erase(it);
            --it;
          }
        }
      }
      int rc = pthread_barrier_wait(&barrier);
      assert(rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD);
      parts.get(id) = new_part;
    }
  };

  void new_page(PartTy& part) {
    Page last;
    last.alloc();
    part.push_back(last);
  }

  void push_back(PartTy& part, T& item) {
    Page& last = part.back();
    if (last.full()) {
      new_page(part);
      part.back().push_back(item);
      return;
    }
    last.push_back(item);
  }

  PartsTy parts;

public:
  typedef PartIterator part_iterator;

  Bag() { 
    new_page(parts.get(0));
  }
  
  virtual ~Bag() {
    int num = parts.size();
    for (int i = 0; i < num; i++) {
      PartTy& pages = parts.get(i);
      for (typename PartTy::iterator it = pages.begin(), end = pages.end(); it != end; ++it) {
        it->dealloc();
      }
    }
  }

  part_iterator part_begin(unsigned int id) {
    return PartIterator(parts.get(id));
  }

  part_iterator part_end(unsigned int id) {
    return PartIterator(parts.get(id), true);
  }

  void push_back(T& item) {
    push_back(parts.get(0), item);
  }

  template<typename PartFn>
  void partition(PartFn f) {
    int numThreads = GaloisRuntime::getSystemThreadPool().getActiveThreads();
    Galois::for_all(Partitioner<PartFn>(this, numThreads, f, parts));
  }
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

struct Octree {
  virtual bool is_leaf() const = 0;
};

struct OctreeInternal : Octree {
  Point pos;
  Octree* child[8];
  OctreeInternal(Point _pos) : pos(_pos) { bzero(child, sizeof(*child) * 8); }
  virtual ~OctreeInternal() {
    for (int i = 0; i < 8; i++) {
      if (child[i] != NULL && !child[i]->is_leaf())
        delete child[i];
    }
  }
  virtual bool is_leaf() const {
    return false;
  }
};

struct Body : Octree {
  Point pos;
  Point vel;
  Point acc;
  double mass;
  Body() { }
  virtual bool is_leaf() const {
    return true;
  }
};

std::ostream& operator<<(std::ostream& os, const Body& b) {
  os << "(pos:" << b.pos << " vel:" << b.vel << " acc:" << b.acc << " mass:" << b.mass << ")";
  return os;
}

struct BoundingBox {
  Point min;
  Point max;
  explicit BoundingBox(const Point& p) : min(p), max(p) { }
  BoundingBox() : min(std::numeric_limits<double>::max()), max(std::numeric_limits<double>::min()) { }

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
    return Point((max.x + min.x) * 0.5, (max.y + min.y) * 0.5, (max.z + min.z) * 0.5);
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
  Config() : dtime(0.5), eps(0.05), tol(0.025), dthf(dtime * 0.5), epssq(eps * eps), itolsq(1.0 / (tol * tol))  { }
};

Config config;

static inline int get_index(const Point& a, const Point& b) {
  int index = 0;
  if (a.x < b.x)
    index += 1;
  if (a.y < b.y)
    index += 2;
  if (a.z < b.z)
    index += 4;
  return index;
}

static inline void update_center(Point& p, int index, double radius) {
  for (int i = 0; i < 3; i++) {
    double v = (index & (1 << i)) > 0 ? radius : -radius;
    p[i] += v;
  }
}

struct Partitioner {
  OctreeInternal* root;
  const std::vector<int>& part_map;
  
  Partitioner(OctreeInternal* _root, const std::vector<int>& _part_map) : root(_root), part_map(_part_map) { }
  int operator()(const Body& b) {
    int acc = 0;
    OctreeInternal* node = root;
    while (node != NULL) {
      int index = get_index(node->pos, b.pos);
      acc = (acc << 3) + index;
      node = static_cast<OctreeInternal*>(node->child[index]);
    } 
    // acc is index of NULL cell, correct for that
    acc = acc >> 3;
    return part_map[acc];
  }
};

struct BuildLocalOctree {
  typedef GaloisRuntime::PerCPU<OctreeInternal*> result_type;
  Bag<Body>& bodies;
  OctreeInternal* root;
  BoundingBox& box;

  BuildLocalOctree(Bag<Body>& _bodies, OctreeInternal* _root, BoundingBox& _box) : bodies(_bodies), root(_root), box(_box) { }
  void operator()(unsigned int id) {
    double radius = box.radius();
    for (Bag<Body>::part_iterator it = bodies.part_begin(id), end = bodies.part_end(id); it != end; ++it) {
      Body* b = &*it;
      insert(b, root, radius);
    }
  }

  void insert(Body* b, OctreeInternal* node, double radius) {
    int index = get_index(node->pos, b->pos);

    assert(!node->is_leaf());
    Octree *child = node->child[index];
    
    if (child == NULL) {
      node->child[index] = b;
      return;
    }
    
    radius *= 0.5;
    if (child->is_leaf()) {
      // Expand leaf
      Body* leaf = static_cast<Body*>(child);
      Point new_pos(leaf->pos);
      update_center(new_pos, index, radius);
      assert(leaf->pos != b->pos);
      OctreeInternal* new_node = new OctreeInternal(new_pos);
      
      insert(b, new_node, radius);
      insert(leaf, new_node, radius);
      node->child[index] = new_node;
    } else {
      insert(b, static_cast<OctreeInternal*>(child), radius);
    }
  }
};

struct ComputeCenterOfMass {
  OctreeInternal* root;
  const std::vector<int>& part_map;
  std::vector<double>& masses;
  int levels;

  ComputeCenterOfMass(OctreeInternal* _root, const std::vector<int>& _part_map, std::vector<double>& _masses, int _levels) :
   root(_root), part_map(_part_map), masses(_masses), levels(_levels) { 
    int nleaves = pow(8, levels);
    masses.reserve(nleaves);
  }

  void operator()(unsigned int id) {
    traverse(root, id, 0, levels);
  }

  double finish() {
    return finish_recurse(root, 0, levels);
  }

private:
  void traverse(OctreeInternal* node, int id, int acc, int level) {
    if (level == 0) {
      if (part_map[acc] == id) 
        masses[acc] = recurse(node);
      
      return;
    }

    for (int i = 0; i < 8; i++) {
      assert(!node->is_leaf());
      OctreeInternal* child = static_cast<OctreeInternal*>(node->child[i]);
      if (child != NULL)
        traverse(child, id, (acc << 3) + i, level - 1);
    }
  }

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
      if (child->is_leaf()) {
        // TODO Copy to iterator in leaf order --- partition here or use explicit partition function?
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
    
    if (mass > 0.0) {
      double inv_mass = 1.0 / mass;
      for (int j = 0; j < 3; j++)
        node->pos[j] = accum[j] * inv_mass;
    }

    return mass;
  }

  double finish_recurse(OctreeInternal* node, int acc, int level) {
    if (level == 0) 
      return masses[acc];

    double mass = 0.0;
    int index = 0;
    Point accum;
		for (int i = 0; i < 8; i++) {
      Octree* child = node->child[i];
      if (child == NULL)
        continue;

      // Reorganize leaves to be denser up front 
      node->child[index++] = child;
      
      double m;
      const Point* p;
      if (child->is_leaf()) {
        // TODO Copy to iterator --- partition here or use explicit partition function?
        Body* n = static_cast<Body*>(child);
        m = n->mass;
        p = &n->pos;
      } else {
        OctreeInternal* n = static_cast<OctreeInternal*>(child);
        m = finish_recurse(n, (acc << 3) + i, level - 1);
        p = &n->pos;
      }

      mass += m;
      for (int j = 0; j < 3; j++) 
        accum[j] += (*p)[j] * m;
    }

    if (mass > 0.0) {
      double inv_mass = 1.0 / mass;
      for (int j = 0; j < 3; j++)
        node->pos[j] = accum[j] * inv_mass;
    }

    return mass;
  }
};

struct ComputeForces {
  Bag<Body>& bodies;
  OctreeInternal* top;
  double diameter;
  int step;

  ComputeForces(Bag<Body>& _bodies, OctreeInternal* _top, double _diameter, int _step) : bodies(_bodies), top(_top), diameter(_diameter), step(_step) { }
  
  void operator()(unsigned int id) {
    double dsq = diameter * diameter * config.itolsq;

    for (Bag<Body>::part_iterator it = bodies.part_begin(id), end = bodies.part_end(id); it != end; ++it) {
      Body& b = *it;
      Point p = b.acc;
      for (int i = 0; i < 3; i++)
        b.acc[i] = 0;
      recurse(b, top, dsq);
      if (step > 0) {
        for (int i = 0; i < 3; i++)
          b.vel[i] += (b.vel[i] - p[i]) * config.dthf;
      }
    }
  }

  void recurse(Body& b, Body* node, double dsq) {
    Point p;
    for (int i = 0; i < 3; i++)
      p[i] = node->pos[i] - b.pos[i];

    double psq = p.x * p.x + p.y * p.y + p.z * p.z;
    psq += config.epssq;
    double idr = 1 / sqrt(psq);
    double nphi = b.mass * idr;
    double scale = nphi * idr * idr;
    for (int i = 0; i < 3; i++) 
      b.acc[i] += p[i] * scale;
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
      double nphi = b.mass * idr;
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
      if (next->is_leaf()) {
        // Check if it is me
        if (&b != next)
          recurse(b, static_cast<Body*>(next), dsq);
      } else {
        recurse(b, static_cast<OctreeInternal*>(next), dsq);
      }
    }
  }
};

struct AdvanceBodies {
  Bag<Body>& bodies;
  AdvanceBodies(Bag<Body>& _bodies) : bodies(_bodies) { }
  void operator()(unsigned int id) {
    for (Bag<Body>::part_iterator it = bodies.part_begin(id), end = bodies.part_end(id); it != end; ++it) {
      Body& b = *it;
      
      Point dvel(b.acc);
      dvel *= config.dthf;

      Point velh(b.vel);
      velh += dvel;

      for (int i = 0; i < 3; ++i)
        b.pos[i] += velh[i] * config.dtime;
      for (int i = 0; i < 3; ++i)
        b.vel[i] = velh[i] + dvel[i];
    }
  }
};

struct ReduceBoxes {
  typedef GaloisRuntime::PerCPU_merge<BoundingBox> result_type;
  Bag<Body>& bodies;
  result_type& result;

  static void mergeBoxes(BoundingBox& b1, BoundingBox& b2) {
    b1.merge(b2);
  }

  ReduceBoxes(Bag<Body>& _bodies, result_type& _result) : bodies(_bodies), result(_result) { }
  void operator()(unsigned int id) {
    BoundingBox& identity = result.get();
    for (Bag<Body>::part_iterator it = bodies.part_begin(id), end = bodies.part_end(id); it != end; ++it) {
      identity.merge((*it).pos);
    }
  }
};

static double next_double() {
  return rand() / (double) RAND_MAX;
}

static void generate_input(Bag<Body>& bodies, int seed, int nbodies) {
  double v, sq, scale;
  Point p;
  double PI = boost::math::constants::pi<double>();

  srand(seed);

  double rsc = (3 * PI) / 16;
  double vsc = sqrt(1.0 / rsc);

  for (int i = 0; i < nbodies; i++) {
    double r = 1.0 / sqrt(pow(next_double() * 0.999, -2.0 / 3.0) - 1);
    do {
      for (int i = 0; i < 3; i++)
        p[i] = next_double() * 2.0 - 1.0;
      sq = p.x * p.x + p.y * p.y + p.z * p.z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);

    Body b;
    b.mass = 1.0 / nbodies;
    for (int i = 0; i < 3; i++)
      b.pos[i] = p[i] * scale;

    do {
      p.x = next_double();
      p.y = next_double() * 0.1;
    } while (p.y > p.x * p.x * pow(1 - p.x * p.x, 3.5));
    v = p.x * sqrt(2.0 / sqrt(1 + r * r));
    do {
      for (int i = 0; i < 3; i++)
        p[i] = next_double() * 2.0 - 1.0;
      sq = p.x * p.x + p.y * p.y + p.z * p.z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    for (int i = 0; i < 3; i++)
      b.vel[i] = p[i] * scale;

    bodies.push_back(b);
  }
}

void make_octree_top_rec(OctreeInternal* parent, double radius, int level) {
  for (int i = 0; i < 8; i++) {
    Point p(parent->pos);
    update_center(p, i, radius);
    parent->child[i] = new OctreeInternal(p);
  }

  if (level > 1) {
    level--;
    radius *= 0.5;
    for (int i = 0; i < 8; i++) 
      make_octree_top_rec(static_cast<OctreeInternal*>(parent->child[i]), radius, level);
  }
}

void make_octree_top(
    const BoundingBox& box, int nparts,
    OctreeInternal*& root, std::vector<int>& part_map, int& levels) {
  double radius = box.diameter() / 2;
  Point p = box.center();
  levels = ceil(log(nparts) / log(8));
  root = new OctreeInternal(p);

  if (levels > 0)
    make_octree_top_rec(root, radius * 0.5, levels - 1);

  int nleaves = pow(8, levels);
  int block_size = nleaves / nparts;
  int index = 0;

  part_map.reserve(nleaves);
  //std::fill(part_map.begin(), part_map.begin() + nleaves, 0);
  for (int i = 0; i < nparts; i++) {
    for (int j = 0; j < block_size; j++) 
      part_map[index++] = i;
  }

  for (; index < nleaves; index++) 
    part_map[index] = index % nparts;
}

void pmain(int nbodies, int ntimesteps, int seed) {
  Bag<Body> bodies;

  generate_input(bodies, seed, nbodies);

  for (int step = 0; step < ntimesteps; step++) {
    ReduceBoxes::result_type result(ReduceBoxes::mergeBoxes);
    ReduceBoxes reduceBoxes(bodies, result);
    Galois::for_all(reduceBoxes);
    BoundingBox box = result.get(0);
    OctreeInternal* top;
    std::vector<int> part_map;
    int levels;
    make_octree_top(box, GaloisRuntime::getSystemThreadPool().getActiveThreads(), top, part_map, levels);

    bodies.partition(Partitioner(top, part_map));

    Galois::for_all(BuildLocalOctree(bodies, top, box));
    std::vector<double> masses;
    ComputeCenterOfMass computeCenterOfMass(top, part_map, masses, levels);
    Galois::for_all(computeCenterOfMass);
    computeCenterOfMass.finish();

    Galois::for_all(ComputeForces(bodies, top, box.diameter(), step));
    Galois::for_all(AdvanceBodies(bodies));

    std::cout << "Timestep " << step << " Center of Mass = " << top->pos << std::endl;
    delete top;
  }
}
