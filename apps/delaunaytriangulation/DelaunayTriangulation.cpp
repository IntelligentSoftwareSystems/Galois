/** Delaunay triangulation -*- C++ -*-
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
 * @section Description
 *
 * Delaunay triangulation of points in 2d.
 *
 * @author Xin Sui <xinsui@cs.utexas.edu>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "Point.h"
#include "Cavity.h"
#include "QuadTree.h"
#include "Verifier.h"

#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/UserContext.h"

#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"

#include <boost/iterator/transform_iterator.hpp>

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
static const char* desc = "Produces a Delaunay triangulation for a set of points\n";
static const char* url = "delaunay_triangulation";

static cll::opt<std::string> doWriteMesh("writemesh", 
    cll::desc("Write the mesh out to files with basename"),
    cll::value_desc("basename"));
static cll::opt<std::string> inputname(cll::Positional, cll::desc("<input file>"), cll::Required);

typedef std::vector<Point> PointList;

Graph* graph;

struct GetPointer: public std::unary_function<Point,Point*> {
  Point* operator()(Point& p) const { return &p; }
};

//! Our main functor
struct Process {
  typedef int tt_needs_per_iter_alloc;
  typedef int tt_does_not_need_parallel_push;
  typedef Galois::PerIterAllocTy Alloc;

  QuadTree* tree;

  struct ContainsTuple {
    const Graph& graph;
    Tuple tuple;
    ContainsTuple(const Graph& g, const Tuple& t): graph(g), tuple(t) { }
    bool operator()(const GNode& n) const {
      return graph.getData(n, Galois::NONE).inTriangle(tuple);
    }
  };

  Process(QuadTree* t): tree(t) { }

  template<typename Alloc>
  bool findContainingElement(const Point* p, const Alloc& alloc, GNode& node) {
    Point* result;
    if (!tree->find(p, result)) {
      return false;
    }

    result->acquire(Galois::CHECK_CONFLICT);

    GNode someNode = result->someElement();

    if (someNode == GNode()) {
      // Not in mesh yet
      return false;
    }

    ContainsTuple contains(*graph, p->t());
    Searcher<Alloc> searcher(*graph, alloc);
    searcher.useMark(p, 0, p->numTries());
    searcher.findFirst(someNode, contains);

    if (searcher.matches.empty()) {
      return false;
    }

    node = searcher.matches.front();
    return true;
  }

  //! Parallel operator
  void operator()(Point* p, Galois::UserContext<Point*>& ctx) {
    p->acquire();
    p->nextTry();
    assert(!p->inMesh());

    GNode node;
    if (!findContainingElement(p, ctx.getPerIterAlloc(), node)) {
      // Someone updated an element while we were searching, producing
      // a semi-consistent state
      ctx.push(p);
      return;
    }
  
    assert(graph->getData(node).inTriangle(p->t()));
    assert(graph->containsNode(node));

    Cavity<Alloc> cav(*graph, ctx.getPerIterAlloc());
    cav.init(node, p);
    cav.build();
    cav.update();
  }

  //! Serial operator
  void operator()(Point* p) {
    p->acquire();
    p->nextTry();
    assert(!p->inMesh());

    GNode node;
    if (!findContainingElement(p, std::allocator<char>(), node)) {
      std::cerr << "Couldn't find triangle containing point\n";
      abort();
    }
  
    assert(graph->getData(node).inTriangle(p->t()));
    assert(graph->containsNode(node));

    Cavity<> cav(*graph);
    cav.init(node, p);
    cav.build();
    cav.update();
  }
};

struct ReadPoints {
  PointList& result;
  ReadPoints(PointList& r): result(r) { }

  void from(const std::string& name) {
    PointList points;
    std::ifstream scanner(name.c_str());
    if (!scanner.good()) {
      std::cerr << "Couldn't open file: " << name << "\n";
      abort();
    }
    if (name.find(".node") == name.size() - 5) {
      fromTriangle(scanner, points);
    } else {
      fromPointList(scanner, points);
    }
    scanner.close();
    
    // Improve locality
    QuadTree t(
      boost::make_transform_iterator(&points[0], GetPointer()),
      boost::make_transform_iterator(&points[points.size()], GetPointer()));
    t.output(std::back_insert_iterator<PointList>(result));
    //std::copy(points.begin(), points.end(), std::back_insert_iterator<PointList>(result));

    addBoundaryPoints();
  }

  void addBoundaryPoints() {
    double minX, maxX, minY, maxY;

    minX = minY = std::numeric_limits<double>::max();
    maxX = maxY = std::numeric_limits<double>::min();

    for (PointList::iterator ii = result.begin(), ei = result.end(); ii != ei; ++ii) {
      double x = ii->t().x();
      double y = ii->t().y();
      if (x < minX)
        minX = x;
      else if (x > maxX)
        maxX = x;
      if (y < minY)
        minY = y;
      else if (y > maxY)
        maxY = y;
    }

    size_t size = result.size();
    double width = maxX - minX;
    double height = maxY - minY;
    double maxLength = std::max(width, height);
    double centerX = minX + width / 2.0;
    double centerY = minY + height / 2.0;
    double radius = maxLength * 3.0; // radius of circle that should cover all points

    for (int i = 0; i < 3; ++i) {
      double dX = radius * cos(2*M_PI*(i/3.0));
      double dY = radius * sin(2*M_PI*(i/3.0));
      result.push_back(Point(centerX + dX, centerY + dY, size + i));
    }
  }

  void nextLine(std::ifstream& scanner) {
    scanner.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  void fromTriangle(std::ifstream& scanner, PointList& points) {
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

  void fromPointList(std::ifstream& scanner, PointList& points) {
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
};

static void writePoints(const std::string& filename, const PointList& points) {
  std::ofstream out(filename.c_str());
  // <num vertices> <dimension> <num attributes> <has boundary markers>
  out << points.size() << " 2 0 0\n";
  //out.setf(std::ios::fixed, std::ios::floatfield);
  out.precision(10);
  long id = 0;
  for (PointList::const_iterator it = points.begin(), end = points.end(); it != end; ++it) {
    const Tuple& t = it->t();
    out << id++ << " " << t.x() << " " << t.y() << " 0\n";
  }

  out.close();
}

static void addBoundaryNodes(PointList& points) {
  size_t last = points.size();
  Point* p1 = &points[last-1];
  Point* p2 = &points[last-2];
  Point* p3 = &points[last-3];

  Element large_triangle(p1, p2, p3);
  GNode large_node = graph->createNode(large_triangle);
  
  p1->addElement(large_node);
  p2->addElement(large_node);
  p3->addElement(large_node);

  Element border_ele1(p1, p2);
  Element border_ele2(p2, p3);
  Element border_ele3(p3, p1);
    
  GNode border_node1 = graph->createNode(border_ele1);
  GNode border_node2 = graph->createNode(border_ele2);
  GNode border_node3 = graph->createNode(border_ele3);

  graph->getEdgeData(graph->addEdge(large_node, border_node1)) = 0;
  graph->getEdgeData(graph->addEdge(large_node, border_node2)) = 1;
  graph->getEdgeData(graph->addEdge(large_node, border_node3)) = 2;

  graph->getEdgeData(graph->addEdge(border_node1, large_node)) = 0;
  graph->getEdgeData(graph->addEdge(border_node2, large_node)) = 0;
  graph->getEdgeData(graph->addEdge(border_node3, large_node)) = 0;
}

static void makeGraph(const std::string& filename, PointList& points) {
  ReadPoints(points).from(filename);

  graph = new Graph();

  addBoundaryNodes(points);
}

static void writeMesh(const std::string& filename) {
  long numTriangles = 0;
  long numSegments = 0;
  for (Graph::active_iterator ii = graph->active_begin(), ei = graph->active_end(); ii != ei; ++ii) {
    Element& e = graph->getData(*ii);
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
  for (Graph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
    const Element& e = graph->getData(*ii);
    if (e.boundary()) {
      // <segment id> <vertex> <vertex> <is boundary>
      pout << sid << " " << e.getPoint(0)->id() << " " << e.getPoint(1)->id() << " 1\n";
      sid++;
    } else {
      // <triangle id> <vertex> <vertex> <vertex> [in ccw order]
      eout << tid << " " << e.getPoint(0)->id() << " ";
      if (e.clockwise()) {
        eout << e.getPoint(2)->id() << " " << e.getPoint(1)->id() << "\n";
      } else {
        eout << e.getPoint(1)->id() << " " << e.getPoint(2)->id() << "\n";
      }
      tid++;
    }
  }

  eout.close();
  // <num holes>
  pout << "0\n";
  pout.close();
}

static void generateMesh(PointList& points) {
  size_t end = points.size();
  size_t eend = end - 3; // end of "real" points

  // Random order is the best algorithmically
  Galois::StatTimer T("build");
  T.start();
  typedef std::vector<Point*> OrderTy;
  OrderTy order;
  order.reserve(eend);
  std::copy(
      boost::make_transform_iterator(&points[0], GetPointer()),
      boost::make_transform_iterator(&points[end], GetPointer()),
      std::back_insert_iterator<OrderTy>(order));
  // But don't randomize boundary points
  std::random_shuffle(&order[0], &order[eend]);
  T.stop();

  size_t prefix = eend - std::min(16U*numThreads, eend);
 
  Galois::StatTimer T1("serial");
  T1.start();
  QuadTree q(&order[eend], &order[end]);
  std::for_each(&order[prefix], &order[eend], Process(&q));
  T1.stop();

  const int multiplier = 8;
  size_t nextStep = multiplier;
  size_t top = prefix;
  size_t prevTop = end;

  using namespace GaloisRuntime::WorkList;
  typedef GaloisRuntime::WorkList::dChunkedLIFO<32> WL;

  do {
    Galois::StatTimer T("build");
    T.start();
    //QuadTree q(&order[top], &order[prevTop]);
    QuadTree q(&order[top], &order[end]);
    prevTop = top;
    top = top > nextStep ? top - nextStep : 0;
    nextStep = nextStep*multiplier; //std::min(nextStep*multiplier, 1000000UL);
    T.stop();

    Galois::StatTimer PT("ParallelTime");
    PT.start();
    Galois::for_each<WL>(&order[top], &order[prevTop], Process(&q));
    PT.stop();
  } while (top > 0);
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

  PointList points;
  makeGraph(inputname, points);
  
  std::cout << "configuration: " << points.size() << " points\n";

  std::cout << "MEMINFO PRE: " << GaloisRuntime::MM::pageAllocInfo() << "\n";
  Galois::preAlloc(1 * numThreads // some per-thread state
      + 2 * points.size() * sizeof(Element) // mesh is about 2x number of points (for random points)
      * 32 // include graph node size
      / (GaloisRuntime::MM::pageSize) // in pages
      );
  std::cout << "MEMINFO MID: " << GaloisRuntime::MM::pageAllocInfo() << "\n";

  Galois::StatTimer T;
  T.start();
  generateMesh(points);
  T.stop();
  std::cout << "mesh size: " << graph->size() << "\n";

  std::cout << "MEMINFO POST: " << GaloisRuntime::MM::pageAllocInfo() << "\n";

  if (!skipVerify) {
    Verifier verifier;
    if (!verifier.verify(graph)) {
      std::cerr << "Triangulation failed.\n";
      assert(0 && "Triangulation failed");
      abort();
    }
    std::cout << "Triangulation OK\n";
  }

  if (doWriteMesh.size()) {
    std::string base = doWriteMesh;
    std::cout << "Writing " << base << "\n";
    writeMesh(base.c_str());

    PointList points;
    ReadPoints(points).from(inputname);
    writePoints(base.append(".node"), points);
  }

  delete graph;

  return 0;
}
