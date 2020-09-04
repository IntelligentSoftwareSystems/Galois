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

#include "Point.h"
#include "Cavity.h"
#include "QuadTree.h"
#include "Verifier.h"

#include "galois/Galois.h"
#include "galois/Bag.h"
#include "galois/Timer.h"

#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <string.h>
#include <unistd.h>

namespace cll = llvm::cl;

static const char* name = "Delaunay Triangulation";
static const char* desc =
    "Produces a Delaunay triangulation for a set of points";
static const char* url = "delaunay_triangulation";

static cll::opt<std::string>
    doWriteMesh("writemesh",
                cll::desc("Write the mesh out to files with basename"),
                cll::value_desc("basename"));
static cll::opt<std::string>
    doWritePoints("writepoints",
                  cll::desc("Write the (reordered) points to filename"),
                  cll::value_desc("filename"));
static cll::opt<bool>
    noReorderPoints("noreorder",
                    cll::desc("Don't reorder points to improve locality"),
                    cll::init(false));
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

enum DetAlgo { nondet, detBase, detPrefix, detDisjoint };

static cll::opt<DetAlgo>
    detAlgo(cll::desc("Deterministic algorithm:"),
            cll::values(clEnumVal(nondet, "Non-deterministic"),
                        clEnumVal(detBase, "Base execution"),
                        clEnumVal(detPrefix, "Prefix execution"),
                        clEnumVal(detDisjoint, "Disjoint execution")),
            cll::init(nondet));

//! Flag that forces user to be aware that they should be passing in a
//! mesh graph.
static cll::opt<bool>
    meshGraph("meshGraph", cll::desc("Specify that the input graph is a mesh"),
              cll::init(false));

struct GetPointer {
  Point* operator()(Point& p) const { return &p; }
};

typedef std::vector<Point> PointList;

class ReadPoints {
  void addBoundaryPoints() {
    double minX, maxX, minY, maxY;

    minX = minY = std::numeric_limits<double>::max();
    maxX = maxY = std::numeric_limits<double>::min();

    for (auto& p : points) {
      double x = p.t().x();
      double y = p.t().y();
      if (x < minX)
        minX = x;
      else if (x > maxX)
        maxX = x;
      if (y < minY)
        minY = y;
      else if (y > maxY)
        maxY = y;
    }

    size_t size      = points.size();
    double width     = maxX - minX;
    double height    = maxY - minY;
    double maxLength = std::max(width, height);
    double centerX   = minX + width / 2.0;
    double centerY   = minY + height / 2.0;
    double radius =
        maxLength * 3.0; // radius of circle that should cover all points

    for (int i = 0; i < 3; ++i) {
      double dX = radius * cos(2 * M_PI * (i / 3.0));
      double dY = radius * sin(2 * M_PI * (i / 3.0));
      points.push_back(Point(centerX + dX, centerY + dY, size + i));
    }
  }

  void nextLine(std::ifstream& scanner) {
    scanner.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  void fromTriangle(std::ifstream& scanner) {
    double x, y;
    long numPoints;

    scanner >> numPoints;

    int dim;
    scanner >> dim;
    assert(dim == 2);
    int k;
    scanner >> k; // number of attributes
    assert(k == 0);
    scanner >> k; // has boundary markers?

    for (long id = 0; id < numPoints; ++id) {
      scanner >> k; // point id
      scanner >> x >> y;
      nextLine(scanner);
      points.push_back(Point(x, y, id));
    }
  }

  void fromPointList(std::ifstream& scanner) {
    double x, y;

    // comment line
    nextLine(scanner);
    size_t id = 0;
    while (!scanner.eof()) {
      scanner >> x >> y;
      if (x == 0 && y == 0)
        break;
      points.push_back(Point(x, y, id++));
      x = y = 0;
      nextLine(scanner);
    }
  }

  PointList& points;

public:
  ReadPoints(PointList& p) : points(p) {}

  void from(const std::string& name) {
    std::ifstream scanner(name.c_str());
    if (!scanner.good()) {
      GALOIS_DIE("could not open file: ", name);
    }
    if (name.find(".node") == name.size() - 5) {
      fromTriangle(scanner);
    } else {
      fromPointList(scanner);
    }
    scanner.close();

    if (points.size())
      addBoundaryPoints();
    else {
      GALOIS_DIE("no points found in file: ", name);
    }
  }
};

static void writePoints(const std::string& filename, const PointList& points) {
  std::ofstream out(filename.c_str());
  // <num vertices> <dimension> <num attributes> <has boundary markers>
  out << points.size() << " 2 0 0\n";
  // out.setf(std::ios::fixed, std::ios::floatfield);
  out.setf(std::ios::scientific, std::ios::floatfield);
  out.precision(10);
  long id = 0;
  for (const auto& p : points) {
    const Tuple& t = p.t();
    out << id++ << " " << t.x() << " " << t.y() << " 0\n";
  }

  out.close();
}

using BasePoints = galois::InsertBag<Point>;
using PtrPoints  = galois::InsertBag<Point*>;
using Rounds     = std::vector<PtrPoints*>;

size_t maxRounds;
const int roundShift = 4; //! round sizes are portional to (1 << roundsShift)

static void copyPointsFromRounds(PointList& points, Rounds& rounds) {
  for (int i = maxRounds - 1; i >= 0; --i) {
    //! [Access elements of InsertBag]
    // PtrPoints expands to galois::InsertBag<Point*>
    // points is of type std::vector<Point>
    PtrPoints& pptrs = *rounds[i];
    for (auto ii : pptrs) {
      points.push_back(*ii);
    }
    //! [Access elements of InsertBag]
  }
}

struct ReadInput {
  Graph& graph;
  BasePoints& basePoints;
  Rounds& rounds;
  std::random_device rng;
  std::mt19937 urng;

  ReadInput(Graph& g, BasePoints& b, Rounds& r)
      : graph(g), basePoints(b), rounds(r), urng(rng()) {}

  void addBoundaryNodes(Point* p1, Point* p2, Point* p3) {
    Element large_triangle(p1, p2, p3);
    GNode large_node = graph.createNode(large_triangle);
    graph.addNode(large_node);

    p1->addElement(large_node);
    p2->addElement(large_node);
    p3->addElement(large_node);

    Element border_ele1(p1, p2);
    Element border_ele2(p2, p3);
    Element border_ele3(p3, p1);

    GNode border_node1 = graph.createNode(border_ele1);
    GNode border_node2 = graph.createNode(border_ele2);
    GNode border_node3 = graph.createNode(border_ele3);

    graph.addNode(border_node1);
    graph.addNode(border_node2);
    graph.addNode(border_node3);

    graph.getEdgeData(graph.addEdge(large_node, border_node1)) = 0;
    graph.getEdgeData(graph.addEdge(large_node, border_node2)) = 1;
    graph.getEdgeData(graph.addEdge(large_node, border_node3)) = 2;

    graph.getEdgeData(graph.addEdge(border_node1, large_node)) = 0;
    graph.getEdgeData(graph.addEdge(border_node2, large_node)) = 0;
    graph.getEdgeData(graph.addEdge(border_node3, large_node)) = 0;
  }

  template <typename L>
  void generateRoundsImpl(const L& loop, size_t size, PointList& points,
                          size_t log2) {
    loop(
        galois::iterate(size_t{0}, size),
        [&, this](size_t index) {
          const Point& p = points[index];

          Point* ptr = &(basePoints.push(p));
          int r      = 0;
          for (size_t i = 0; i < log2; ++i) {
            size_t mask = (1UL << (i + 1)) - 1;
            if ((index & mask) == (1UL << i)) {
              r = i;
              break;
            }
          }

          rounds[r / roundShift]->push(ptr);
        },
        galois::loopname("generateRoundsImpl"));
  }

  //! Blocked point distribution (exponentially increasing block size) with
  //! points randomized within a round
  void generateRoundsOld(PointList& points, bool randomize) {
    size_t counter = 0;
    size_t round   = 0;
    size_t next    = 1 << roundShift;
    std::vector<Point*> buf;

    PointList::iterator ii = points.begin(), ei = points.end();
    while (ii != ei) {
      Point* ptr = &(basePoints.push(*ii));
      buf.push_back(ptr);
      ++ii;
      if (ii == ei || counter > next) {
        next *= next;
        int r = maxRounds - 1 - round;
        if (randomize)
          std::shuffle(buf.begin(), buf.end(), urng);
        std::copy(buf.begin(), buf.end(), std::back_inserter(*rounds[r]));
        buf.clear();
        ++round;
      }
      ++counter;
    }
  }

  void generateRounds(PointList& points, bool addBoundary) {
    size_t size = points.size() - 3;

    size_t log2 = std::max((size_t)floor(log(size) / log(2)), (size_t)1);
    maxRounds   = log2 / roundShift;
    for (size_t i = 0; i <= maxRounds;
         i++) { // rounds[maxRounds+1] for boundary points
      rounds.push_back(new galois::InsertBag<Point*>);
    }

    PointList ordered;
    // ordered.reserve(size);

    if (noReorderPoints) {
      std::copy(points.begin(), points.begin() + size,
                std::back_inserter(ordered));
      generateRoundsOld(ordered, false);
    } else {
      // Reorganize spatially
      QuadTree q(
          boost::make_transform_iterator(points.begin(), GetPointer()),
          boost::make_transform_iterator(points.begin() + size, GetPointer()));

      q.output(std::back_inserter(ordered));

      if (true) {
        if (detAlgo == nondet) {
          generateRoundsImpl(galois::DoAll(), size, ordered, log2);

        } else {
          generateRoundsImpl(galois::StdForEach(), size, ordered, log2);
        }
      } else {
        generateRoundsOld(ordered, true);
      }
    }

    if (!addBoundary)
      return;

    // Now, handle boundary points
    size_t last = points.size();
    //! [Insert elements into InsertBag]
    // basePoints is of type galois::InsertBag<Point>
    // points is of type std::vector<Point>
    Point* p1 = &(basePoints.push(points[last - 1]));
    Point* p2 = &(basePoints.push(points[last - 2]));
    Point* p3 = &(basePoints.push(points[last - 3]));
    //! [Insert elements into InsertBag]

    rounds[maxRounds]->push(p1);
    rounds[maxRounds]->push(p2);
    rounds[maxRounds]->push(p3);

    addBoundaryNodes(p1, p2, p3);
  }

  void operator()(const std::string& filename, bool addBoundary) {
    PointList points;
    ReadPoints(points).from(filename);

    std::cout << "configuration: " << points.size() << " points\n";

#if 1
    galois::preAlloc(
        32 * points.size() * sizeof(Element) *
        1.5 // mesh is about 2x number of points (for random points)
        / (galois::runtime::pagePoolSize()) // in pages
    );
#else
    galois::preAlloc(1 * numThreads // some per-thread state
                     + 2 * points.size() *
                           sizeof(Element) // mesh is about 2x number of points
                                           // (for random points)
                           * 32            // include graph node size
                           / (galois::runtime::hugePageSize) // in pages
    );
#endif
    galois::reportPageAlloc("MeminfoPre");

    galois::StatTimer T("generateRounds");
    T.start();
    generateRounds(points, addBoundary);
    T.stop();
  }
};

static void writeMesh(const std::string& filename, Graph& graph) {
  long numTriangles = 0;
  long numSegments  = 0;
  for (auto n : graph) {
    Element& e = graph.getData(n);
    if (e.boundary()) {
      numSegments++;
    } else {
      numTriangles++;
    }
  }

  long tid = 0;
  long sid = 0;
  std::string elementName(filename);
  std::string polyName(filename);

  elementName.append(".ele");
  polyName.append(".poly");

  std::ofstream eout(elementName.c_str());
  std::ofstream pout(polyName.c_str());
  // <num triangles> <nodes per triangle> <num attributes>
  eout << numTriangles << " 3 0\n";
  // <num vertices> <dimension> <num attributes> <has boundary markers>
  // ...
  // <num segments> <has boundary markers>
  pout << "0 2 0 0\n";
  pout << numSegments << " 1\n";
  for (auto n : graph) {
    const Element& e = graph.getData(n);
    if (e.boundary()) {
      // <segment id> <vertex> <vertex> <is boundary>
      pout << sid++ << " " << e.getPoint(0)->id() << " " << e.getPoint(1)->id()
           << " 1\n";
    } else {
      // <triangle id> <vertex> <vertex> <vertex> [in ccw order]
      eout << tid++ << " " << e.getPoint(0)->id() << " ";
      if (e.clockwise()) {
        eout << e.getPoint(2)->id() << " " << e.getPoint(1)->id() << "\n";
      } else {
        eout << e.getPoint(1)->id() << " " << e.getPoint(2)->id() << "\n";
      }
    }
  }

  eout.close();
  // <num holes>
  pout << "0\n";
  pout.close();
}

struct DelaunayTriangulation {

  QuadTree* tree;
  Graph& graph;

  struct ContainsTuple {
    const Graph& graph;
    Tuple tuple;
    ContainsTuple(const Graph& g, const Tuple& t) : graph(g), tuple(t) {}
    bool operator()(const GNode& n) const {
      assert(!graph.getData(n, galois::MethodFlag::UNPROTECTED).boundary());
      return graph.getData(n, galois::MethodFlag::UNPROTECTED)
          .inTriangle(tuple);
    }
  };

  void computeCenter(const Element& e, Tuple& t) const {
    for (int i = 0; i < 3; ++i) {
      const Tuple& o = e.getPoint(i)->t();
      for (int j = 0; j < 2; ++j) {
        t[j] += o[j];
      }
    }
    for (int j = 0; j < 2; ++j) {
      t[j] *= 1 / 3.0;
    }
  }

  void findBestNormal(const Element& element, const Point* p,
                      const Point*& bestP1, const Point*& bestP2) {
    Tuple center(0);
    computeCenter(element, center);
    int scale = element.clockwise() ? 1 : -1;

    Tuple origin = p->t() - center;
    //        double length2 = origin.x() * origin.x() + origin.y() *
    //        origin.y();
    bestP1 = bestP2 = NULL;
    double bestVal  = 0.0;
    for (int i = 0; i < 3; ++i) {
      int next = i + 1;
      if (next > 2)
        next -= 3;

      const Point* p1 = element.getPoint(i);
      const Point* p2 = element.getPoint(next);
      double dx       = p2->t().x() - p1->t().x();
      double dy       = p2->t().y() - p1->t().y();
      Tuple normal(scale * -dy, scale * dx);
      double val = normal.dot(origin); // / length2;
      if (bestP1 == NULL || val > bestVal) {
        bestVal = val;
        bestP1  = p1;
        bestP2  = p2;
      }
    }
    assert(bestP1 != NULL && bestP2 != NULL && bestVal > 0);
  }

  GNode findCorrespondingNode(GNode start, const Point* p1, const Point* p2) {
    for (auto ii : graph.edges(start)) {
      GNode dst  = graph.getEdgeDst(ii);
      Element& e = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      int count  = 0;
      for (int i = 0; i < e.dim(); ++i) {
        if (e.getPoint(i) == p1 || e.getPoint(i) == p2) {
          if (++count == 2)
            return dst;
        }
      }
    }
    GALOIS_DIE("unreachable");
    return start;
  }

  bool planarSearch(const Point* p, GNode start, GNode& node) {
    // Try simple hill climbing instead
    ContainsTuple contains(graph, p->t());
    while (!contains(start)) {
      Element& element = graph.getData(start, galois::MethodFlag::WRITE);
      if (element.boundary()) {
        // Should only happen when quad tree returns a boundary point which is
        // rare There's only one way to go from here
        assert(std::distance(graph.edge_begin(start), graph.edge_end(start)) ==
               1);
        start = graph.getEdgeDst(
            graph.edge_begin(start, galois::MethodFlag::WRITE));
      } else {
        // Find which neighbor will get us to point fastest by computing normal
        // vectors
        const Point *p1, *p2;
        findBestNormal(element, p, p1, p2);
        start = findCorrespondingNode(start, p1, p2);
      }
    }

    node = start;
    return true;
  }

  bool findContainingElement(const Point* p, GNode& node) {
    Point* result;
    if (!tree->find(p, result)) {
      return false;
    }

    result->get(galois::MethodFlag::WRITE);

    GNode someNode = result->someElement();

    // Not in mesh yet
    if (!someNode) {
      return false;
    }

    return planarSearch(p, someNode, node);
  }

  using Alloc = galois::PerIterAllocTy;

  struct LocalState {
    Cavity<Alloc> cav;
    LocalState(Graph& graph, Alloc& alloc) : cav(graph, alloc) {}
  };

  template <int Version, typename C>
  void processPoint(Point* p, C& ctx) {
    Cavity<Alloc>* cavp = NULL;

    if (Version == detDisjoint) {

      if (ctx.isFirstPass()) {
        LocalState* localState = ctx.template createLocalState<LocalState>(
            graph, ctx.getPerIterAlloc());
        cavp = &localState->cav;

      } else {

        LocalState* localState = ctx.template getLocalState<LocalState>();
        localState->cav.update();
        return;
      }
    }

    p->get(galois::MethodFlag::WRITE);
    assert(!p->inMesh());

    GNode node;
    if (!findContainingElement(p, node)) {
      // Someone updated an element while we were searching, producing
      // a semi-consistent state
      // ctx.push(p);
      // Current version is safe with locking so this shouldn't happen
      GALOIS_DIE("unreachable");
      return;
    }

    assert(graph.getData(node).inTriangle(p->t()));
    assert(graph.containsNode(node));

    if (Version == detDisjoint && ctx.isFirstPass()) {
      cavp->init(node, p);
      cavp->build();
    } else {
      Cavity<Alloc> cav(graph, ctx.getPerIterAlloc());
      cav.init(node, p);
      cav.build();
      if (Version == detPrefix)
        return;
      ctx.cautiousPoint();
      cav.update();
    }
  }

  template <int Version, typename WL, typename B, typename... Args>
  void generateMesh(B& pptrs, Args&&... args) {

    galois::for_each(
        galois::iterate(pptrs),
        [&, this](Point* p, auto& ctx) { this->processPoint<Version>(p, ctx); },
        galois::wl<WL>(), galois::loopname("generateMesh"),
        galois::local_state<LocalState>(), galois::per_iter_alloc(),
        galois::no_pushes(), std::forward<Args>(args)...);
  }
};

/*
template<int Version=detBase>
struct Process {



  //! Serial operator
  void operator()(Point* p) {
    p->get(galois::MethodFlag::WRITE);
    assert(!p->inMesh());

    GNode node;
    if (!findContainingElement(p, node)) {
      GALOIS_DIE("Could not find triangle containing point");
      return;
    }

    assert(graph.getData(node).inTriangle(p->t()));
    assert(graph.containsNode(node));

    Cavity<> cav(graph);
    cav.init(node, p);
    cav.build();
    cav.update();
  }
};
*/

static void run(Rounds& rounds, Graph& graph) {
  typedef galois::worklists::PerThreadChunkLIFO<32> Chunk;
  typedef galois::worklists::Deterministic<> DWL;

  for (int i = maxRounds - 1; i >= 0; --i) {

    galois::StatTimer BT("buildtree");
    BT.start();
    assert(rounds[i + 1]);
    PtrPoints& tptrs = *(rounds[i + 1]);
    QuadTree tree(tptrs.begin(), tptrs.end());
    BT.stop();

    galois::StatTimer PT("ParallelTime");
    PT.start();

    assert(rounds[i]);
    galois::InsertBag<Point*>& pptrs = *(rounds[i]);

    DelaunayTriangulation dt{&tree, graph};
    switch (detAlgo) {
    case nondet:
      dt.generateMesh<detBase, Chunk>(pptrs);
      break;
    case detBase:
      dt.generateMesh<detBase, DWL>(pptrs);
      break;
    case detPrefix: {
      auto nv = [&dt](Point* p, auto& ctx) {
        dt.processPoint<detPrefix>(p, ctx);
      };
      dt.generateMesh<detBase, DWL>(
          pptrs, galois::neighborhood_visitor<decltype(nv)>(nv));
      break;
    }
    case detDisjoint:
      dt.generateMesh<detDisjoint, DWL>(pptrs);
      break;
    default:
      GALOIS_DIE("unknown algorithm: ", detAlgo);
    }

    PT.stop();
  }
}

void deleteRounds(Rounds& rounds) {
  for (auto r : rounds)
    delete r;
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  if (!meshGraph) {
    GALOIS_DIE("This application requires a mesh graph input;"
               " please use the -meshGraph flag "
               " to indicate the input is a mesh graph.");
  }

  Graph graph;

  //! All Point* refer to elements in this bag
  //! [Define InsertBag]
  // BasePoints expands to galois::InsertBag<Point>
  BasePoints basePoints;
  //! [Define InsertBag]

  Rounds rounds;

  bool writepoints = doWritePoints.size() > 0;
  ReadInput(graph, basePoints, rounds)(inputFile, !writepoints);
  if (writepoints) {
    std::cout << "Writing " << doWritePoints << "\n";
    PointList points;
    copyPointsFromRounds(points, rounds);
    writePoints(doWritePoints, points);
    deleteRounds(rounds);
    return 0;
  }

  const char* name = 0;
  switch (detAlgo) {
  case nondet:
    name = "nondet";
    break;
  case detBase:
    name = "detBase";
    break;
  case detPrefix:
    name = "detPrefix";
    break;
  case detDisjoint:
    name = "detDisjoint";
    break;
  default:
    name = "unknown";
    break;
  }
  galois::gInfo("Algorithm ", name);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  run(rounds, graph);
  execTime.stop();
  std::cout << "mesh size: " << graph.size() << "\n";

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    Verifier verifier;
    if (!verifier.verify(&graph)) {
      GALOIS_DIE("triangulation failed");
    }
    std::cout << "Triangulation OK\n";
  }

  if (doWriteMesh.size()) {
    std::string base = doWriteMesh;
    std::cout << "Writing " << base << "\n";
    writeMesh(base.c_str(), graph);

    PointList points;
    // Reordering messes up connection between id and place in pointlist
    ReadPoints(points).from(inputFile);
    writePoints(base.append(".node"), points);
  }

  deleteRounds(rounds);

  totalTime.stop();

  return 0;
}
