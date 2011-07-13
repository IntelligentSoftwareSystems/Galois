/** Delaunay refinement -*- C++ -*-
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
 * Refinement of an initial, unrefined Delaunay mesh to eliminate triangles
 * with angles < 30 degrees, using a variation of Chew's algorithm.
 *
 * @author Milind Kulkarni <milind@purdue.edu>>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include <iostream>
#include <stack>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <cassert>

#include "Element.h"

#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

static const char* name = "Delaunay Mesh Refinement";
static const char* description = "Refines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/delaunayrefinement.html";
static const char* help = "<input file>";

typedef Galois::Graph::FirstGraph<Element,void,false>            Graph;
typedef Galois::Graph::FirstGraph<Element,void,false>::GraphNode GNode;


#include "Subgraph.h"
#include "Mesh.h"
#include "Cavity.h"

Graph* mesh;

struct process {
  typedef int tt_needs_per_iter_alloc;

  template<typename Context>
  void operator()(GNode item, Context& lwl) {
    if (!mesh->containsNode(item))
      return;
    
    item.getData(Galois::ALL); //lock

    Cavity cav(mesh, lwl.getPerIterAlloc());
    cav.initialize(item);
    cav.build();
    cav.update();
    
    for (Subgraph::iterator ii = cav.getPre().begin(),
	   ee = cav.getPre().end(); ii != ee; ++ii) 
      mesh->removeNode(*ii, Galois::NONE);
    
    //add new data
    for (Subgraph::iterator ii = cav.getPost().begin(),
	   ee = cav.getPost().end(); ii != ee; ++ii) {
      GNode node = *ii;
      mesh->addNode(node, Galois::ALL);
      Element& element = node.getData(Galois::ALL);
      if (element.isBad()) {
        lwl.push(node);
      }
    }
    
    for (Subgraph::edge_iterator ii = cav.getPost().edge_begin(),
	   ee = cav.getPost().edge_end(); ii != ee; ++ii) {
      Subgraph::tmpEdge edge = *ii;
      //bool ret = 
      mesh->addEdge(edge.src, edge.dst, Galois::ALL); //, edge.data);
      //assert ret;
    }
    if (mesh->containsNode(item)) {
      lwl.push(item);
    }
  }
};

int main(int argc, const char** argv) {
  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() != 1) {
    std::cout << "incorrect number of arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  mesh = new Graph();
  Mesh m;
  m.read(mesh, args[0]);
  std::vector<GNode> wl;
  int numbad = m.getBad(mesh, wl);

  std::cout << "configuration: " << mesh->size() << " total triangles, " << numbad << " bad triangles\n";

  Galois::StatTimer T;
  T.start();
  using namespace GaloisRuntime::WorkList;
  Galois::for_each<LocalQueues<ChunkedLIFO<1024>, LIFO<> > >(wl.begin(), wl.end(), process());
  T.stop();
  
  if (!skipVerify) {
    if (!m.verify(mesh)) {
      std::cerr << "Refinement failed.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    
    int size = m.getBad(mesh, wl);
    if (size != 0) {
      std::cerr << size << " bad triangles remaining.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    std::cout << "Refinement OK\n";
  }
  return 0;
}

