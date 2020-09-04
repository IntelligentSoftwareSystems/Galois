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
#include "Verifier.h"

#include "galois/Galois.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/SpatialTree.h"
#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"

#include "galois/runtime/Profile.h"

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
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string>
    doWriteMesh("writemesh",
                cll::desc("Write the mesh out to files with basename"),
                cll::value_desc("basename"));

//! Flag that forces user to be aware that they should be passing in a
//! mesh graph.
static cll::opt<bool>
    meshGraph("meshGraph", cll::desc("Specify that the input graph is a mesh"),
              cll::init(false));

using Tree = typename galois::graphs::SpatialTree2d<Point*>;

//! All Point* refer to elements in this bag
using basePointBag = typename galois::InsertBag<Point>;

//! [Define Insert Bag]
using ptrPointBag = typename galois::InsertBag<Point*>;

//! Our main functor
struct Process {
  Graph& graph;
  Tree& tree;
  ptrPointBag& ptrPoints;

  Process(Graph& g, Tree& t, ptrPointBag& p)
      : graph(g), tree(t), ptrPoints(p) {}

  typedef galois::PerIterAllocTy Alloc;

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
    Point** rp = tree.find(p->t().x(), p->t().y());
    if (!rp)
      return false;

    (*rp)->get(galois::MethodFlag::WRITE);

    GNode someNode = (*rp)->someElement();

    // Not in mesh yet
    if (!someNode) {
      GALOIS_DIE("unreachable");
      return false;
    }

    return planarSearch(p, someNode, node);
  }

  void generateMesh() {
    typedef galois::worklists::PerThreadChunkLIFO<32> CA;
    galois::for_each(
        galois::iterate(ptrPoints),
        [&, self = this](Point* p, auto& ctx) {
          p->get(galois::MethodFlag::WRITE);
          assert(!p->inMesh());

          GNode node;
          if (!self->findContainingElement(p, node)) {
            // Someone updated an element while we were searching,
            // producing a semi-consistent state ctx.push(p);
            // Current version is safe with locking so this
            // shouldn't happen
            GALOIS_DIE("unreachable");
            return;
          }

          assert(self->graph.getData(node).inTriangle(p->t()));
          assert(self->graph.containsNode(node));

          Cavity<Alloc> cav(self->graph, ctx.getPerIterAlloc());
          cav.init(node, p);
          cav.build();
          cav.update();
          self->tree.insert(p->t().x(), p->t().y(), p);
        },
        galois::no_pushes(), galois::per_iter_alloc(), galois::loopname("Main"),
        galois::wl<CA>());
  }
};

typedef std::vector<Point> PointList;

class ReadPoints {
  void addBoundaryPoints() {
    double minX, maxX, minY, maxY;

    minX = minY = std::numeric_limits<double>::max();
    maxX = maxY = std::numeric_limits<double>::min();

    for (const auto& p : points) {
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

    tree.init(minX, minY, maxX, maxY);

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
  Tree& tree;

public:
  ReadPoints(PointList& p, Tree& t) : points(p), tree(t) {}

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

struct ReadInput {
  Graph& graph;
  Tree& tree;
  basePointBag& basePoints;
  ptrPointBag& ptrPoints;
  std::random_device rng;
  std::mt19937 urng;

  ReadInput(Graph& g, Tree& t, basePointBag& b, ptrPointBag& p)
      : graph(g), tree(t), basePoints(b), ptrPoints(p), urng(rng()) {}

  void addBoundaryNodes(Point* p1, Point* p2, Point* p3) {
    Element large_triangle(p1, p2, p3);
    GNode large_node = graph.createNode(large_triangle);
    graph.addNode(large_node);

    p1->addElement(large_node);
    p2->addElement(large_node);
    p3->addElement(large_node);

    tree.insert(p1->t().x(), p1->t().y(), p1);

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

  struct centerXCmp {
    template <typename T>
    bool operator()(const T& lhs, const T& rhs) const {
      return lhs.t().x() < rhs.t().x();
    }
  };

  struct centerYCmp {
    template <typename T>
    bool operator()(const T& lhs, const T& rhs) const {
      return lhs.t().y() < rhs.t().y();
    }
  };

  struct centerYCmpInv {
    template <typename T>
    bool operator()(const T& lhs, const T& rhs) const {
      return rhs.t().y() < lhs.t().y();
    }
  };

  template <typename Iter>
  void divide(const Iter& b, const Iter& e) {
    if (std::distance(b, e) > 64) {
      std::sort(b, e, centerXCmp());
      Iter m = galois::split_range(b, e);
      std::sort(b, m, centerYCmpInv());
      std::sort(m, e, centerYCmp());
      divide(b, galois::split_range(b, m));
      divide(galois::split_range(b, m), m);
      divide(m, galois::split_range(m, e));
      divide(galois::split_range(m, e), e);
    } else {
      std::shuffle(b, e, urng);
    }
  }

  void layoutPoints(PointList& points) {
    divide(points.begin(), points.end() - 3);
    galois::do_all(galois::iterate(points.begin(), points.end() - 3),
                   [&](Point& p) {
                     Point* pr = &basePoints.push(p);
                     ptrPoints.push(pr);
                   });
    //! [Insert elements into InsertBag]
    Point* p1 = &basePoints.push(*(points.end() - 1));
    Point* p2 = &basePoints.push(*(points.end() - 2));
    Point* p3 = &basePoints.push(*(points.end() - 3));
    //! [Insert elements into InsertBag]
    addBoundaryNodes(p1, p2, p3);
  }

  void operator()(const std::string& filename) {
    PointList points;
    ReadPoints(points, tree).from(filename);

    std::cout << "configuration: " << points.size() << " points\n";

    galois::preAlloc(2 * numThreads // some per-thread state
                     + 2 * points.size() *
                           sizeof(Element) // mesh is about 2x number of points
                                           // (for random points)
                           * 32            // include graph node size
                           / (galois::runtime::pagePoolSize()) // in pages
    );
    galois::reportPageAlloc("MeminfoPre");

    layoutPoints(points);
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
  Tree tree;
  basePointBag basePoints;
  ptrPointBag ptrPoints;

  ReadInput(graph, tree, basePoints, ptrPoints)(inputFile);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  galois::runtime::profileVtune(
      [&]() { Process(graph, tree, ptrPoints).generateMesh(); },
      "MeshGeneration");
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
    ReadPoints(points, tree).from(inputFile);
    writePoints(base.append(".node"), points);
  }

  totalTime.stop();

  return 0;
}
