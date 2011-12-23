/** Delaunay triangulation -*- C++ -*-
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
 * @section Description
 *
 * Delaunay triangulation of points in 2d.
 *
 * @author Xin Sui <xinsui@cs.utexas.edu>
 */
#include <vector>
#include <iostream>
#include <fstream>
#include <string.h>
#include <limits>
#include <unistd.h>

#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Tuple.h"
#include "Element.h"

#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

static const char* name = "Delaunay Triangulation";
static const char* desc = "Produces a Delaunay triangulation from a given a set of points\n";
static const char* url = "delaunay_triangulation";

cll::opt<std::string> writeMesh("writemesh", cll::desc("Write the mesh out to files with basename"), cll::value_desc("basename"));
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

typedef Galois::Graph::FirstGraph<Element,int,true>            Graph;
typedef Galois::Graph::FirstGraph<Element,int,true>::GraphNode GNode;

#include "Cavity.h"
#include "Verifier.h"

Graph* Mesh;

struct Process {
  typedef int tt_needs_per_iter_alloc;

  std::vector<GNode>* m_shadow_wl;
  int m_prologue_count;
  const int m_prologue_iterations;

  Process(std::vector<GNode>* wl, int it):
    m_shadow_wl(wl), m_prologue_count(0), m_prologue_iterations(it) { }
  Process(): m_shadow_wl(NULL), m_prologue_iterations(0) { }

  template<typename Context>
  void operator()(GNode item, Context& lwl) {
    if (m_shadow_wl != NULL) {
      if (m_prologue_count >= m_prologue_iterations) {
        m_shadow_wl->push_back(item);
        return;
      } else {
        ++m_prologue_count;
      }
    } 

    Element& data = item.getData(Galois::ALL); //lock

    if (!Mesh->containsNode(item)) 
      return;
  
    TupleList& tuples = data.getTuples();
    // Discard duplicate tuples
    while (!tuples.empty()) {
      Tuple& t = tuples.back();

      int i;
      for (i = 0; i < 3; ++i) {
        if (data.getPoint(i) == t) {
          tuples.pop_back();
          break;
        }
      }
      if (i == 3)
        break;
    }

    if (tuples.empty())
      return;

    Cavity cav(Mesh, item, tuples.back(), lwl.getPerIterAlloc());
    cav.build();
    
    Cavity::GNodeVector newNodes(lwl.getPerIterAlloc());
    cav.update(&newNodes);
    for (Cavity::GNodeVector::iterator iter = newNodes.begin(); iter != newNodes.end(); ++iter)
        lwl.push(*iter);
  }
};

void read_points(const char* filename, TupleList& tuples) {
  double x, y, min_x, max_x, min_y, max_y;
  long numPoints;

  min_x = min_y = std::numeric_limits<double>::max();
  max_x = max_y = std::numeric_limits<double>::min();

  std::ifstream scanner(filename);
  scanner >> numPoints;
  tuples.clear();

  int dim;
  scanner >> dim;
  assert(dim == 2);
  int k;
  scanner >> k; // number of attributes
  assert(k == 0);
  scanner >> k; // has boundary markers?
  assert(k == 0);

  for (long i = 0; i < numPoints; ++i) {
    scanner >> k; // point id
    scanner >> x >> y;
    scanner.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    if (x < min_x)
      min_x = x;
    else if (x > max_x)
      max_x = x;
    if (y < min_y)
      min_y = y;
    else if (y > max_y)
      max_y = y;
    tuples.push_back(Tuple(x, y, i));
  }
  scanner.close();

  double width = max_x - min_x;
  double height = max_y - min_y;
  double max_length = std::max(width, height);
  double centerX = min_x + width / 2.0;
  double centerY = min_y + height / 2.0;

  tuples.push_back(Tuple(centerX, centerY + 3.0 * max_length, numPoints));
  tuples.push_back(Tuple(centerX - 3.0 * max_length, centerY - 2.0 * max_length, numPoints + 1));
  tuples.push_back(Tuple(centerX + 3.0 * max_length, centerY - 2.0 * max_length, numPoints + 2));
}

void write_points(const char* filename, const TupleList& tuples) {
  std::ofstream out(filename);
  // <num vertices> <dimension> <num attributes> <has boundary markers>
  out << tuples.size() << " 2 0 0\n";
  //out.setf(std::ios::fixed, std::ios::floatfield);
  out.precision(10);
  long id = 0;
  for (TupleList::const_iterator it = tuples.begin(), end = tuples.end(); it != end; ++it) {
    const Tuple &t = *it;
    out << id++ << " " << t.x() << " " << t.y() << " 0\n";
  }

  out.close();
}

GNode make_graph(const char* filename) {
  TupleList tuples;
  read_points(filename, tuples);
  
  Tuple t1 = tuples.back();
  tuples.pop_back();

  Tuple t2 = tuples.back();
  tuples.pop_back();

  Tuple t3 = tuples.back();
  tuples.pop_back();

  Mesh = new Graph();
  Element large_triangle(t1, t2, t3);
  GNode large_node = Mesh->createNode(large_triangle);
  
  Mesh->addNode(large_node, Galois::NONE);

  Element border_ele1(t1, t2);
  Element border_ele2(t2, t3);
  Element border_ele3(t3, t1);
    
  GNode border_node1 = Mesh->createNode(border_ele1);
  GNode border_node2 = Mesh->createNode(border_ele2);
  GNode border_node3 = Mesh->createNode(border_ele3);

  Mesh->addNode(border_node1, Galois::NONE);
  Mesh->addNode(border_node2, Galois::NONE);
  Mesh->addNode(border_node3, Galois::NONE);

  Mesh->addEdge(large_node, border_node1, 0);
  Mesh->addEdge(large_node, border_node2, 1);
  Mesh->addEdge(large_node, border_node3, 2);

  Mesh->addEdge(border_node1, large_node, 0);
  Mesh->addEdge(border_node2, large_node, 0);
  Mesh->addEdge(border_node3, large_node, 0);
  
  large_node.getData().getTuples().swap(tuples);

  return large_node;
}

void write_mesh(const char* filename) {
  long num_triangles = 0, num_segments = 0;
  for (Graph::active_iterator ii = Mesh->active_begin(), ee = Mesh->active_end(); ii != ee; ++ii) {
    GNode node = *ii;
    Element& e = node.getData(Galois::NONE);
    if (e.getBDim()) {
      num_triangles++;
    } else {
      num_segments++;
    }
  }

  long tid = 0, sid = 0;
  std::ofstream eout(std::string(filename).append(".ele").c_str());
  std::ofstream pout(std::string(filename).append(".poly").c_str());
  // <num triangles> <nodes per triangle> <num attributes>
  eout << num_triangles << " 3 0\n";
  // <num vertices> <dimension> <num attributes> <has boundary markers>
  // ...
  // <num segments> <has boundary markers>
  pout << "0 2 0 0\n";
  pout << num_segments << " 1\n";
  for (Graph::active_iterator ii = Mesh->active_begin(), ee = Mesh->active_end(); ii != ee; ++ii) {
    GNode node = *ii;
    Element& e = node.getData(Galois::NONE);
    if (e.getBDim()) {
      // <triangle id> <vertex> <vertex> <vertex> [in ccw order]
      eout << tid << " " << e.getPoint(0).id() << " ";
      if (e.clockwise()) {
        eout << e.getPoint(2).id() << " " << e.getPoint(1).id() << "\n";
      } else {
        eout << e.getPoint(1).id() << " " << e.getPoint(2).id() << "\n";
      }
      tid++;
    } else {
      // <segment id> <vertex> <vertex> <is boundary>
      pout << sid << " " << e.getPoint(0).id() << " " << e.getPoint(1).id() << " 1\n";
      sid++;
    }
  }

  eout.close();
  // <num holes>
  pout << "0\n";
  pout.close();
}

bool ends_with(const char* str, const char* end) {
  size_t slen = strlen(str);
  size_t elen = strlen(end);
  if (elen > slen)
    return false;
  size_t diff = slen - elen;
  return strcmp(str + diff, end) == 0;
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

  if (!ends_with(filename.c_str(), ".node")) {
    std::cout << "must pass .node file, use -help for usage information\n";
    return 1;
  }

  GNode initial_triangle = make_graph(filename.c_str());
  
  std::vector<GNode> wl;
  wl.push_back(initial_triangle);
  std::cout << "configuration: " << initial_triangle.getData().getTuples().size() << " points\n";

  Galois::StatTimer T;
  T.start();
  const int chunkSize = 1024; // XXX: Set this correctly
  using namespace GaloisRuntime::WorkList;
  typedef ChunkedLIFO<chunkSize> WL;
  if (false) {
    Galois::StatTimer T1("serial"), T2("parallel");
    T1.start();
    Galois::setMaxThreads(1);
    std::vector<GNode> shadow_wl;
    int num_prologue_iterations = 1; //numThreads*8; //1024; //std::max<int>(chunkSize*numThreads*4, 512);
    Galois::for_each(wl.begin(), wl.end(), Process(&shadow_wl, num_prologue_iterations));
    T1.stop();

    T2.start();
    Galois::setMaxThreads(numThreads);
    Galois::for_each<WL>(shadow_wl.begin(), shadow_wl.end(), Process());
    T2.stop();
  } else {
    Galois::for_each<WL>(wl.begin(), wl.end(), Process());
  }
  T.stop();
  std::cout << "mesh size: " << Mesh->size() << "\n";

  if (!skipVerify) {
    Verifier verifier;
    if (!verifier.verify(Mesh)) {
      std::cerr << "Triangulation failed.\n";
      assert(0 && "Triangulation failed");
      abort();
    }
    std::cout << "Triangulation OK\n";
  }

  if (writeMesh.size()) {
    std::string base = writeMesh;
    std::cout << "Writing " << base << "\n";
    write_mesh(base.c_str());

    TupleList tuples;
    read_points(filename.c_str(), tuples);
    write_points(base.append(".node").c_str(), tuples);
  }

  delete Mesh;

  return 0;
}
