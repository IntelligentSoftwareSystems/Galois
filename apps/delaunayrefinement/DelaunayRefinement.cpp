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
#include "Galois/Graphs/FastGraph.h"
#include "Galois/Galois.h"

#include "llvm/Support/CommandLine.h"

#include "Galois/Runtime/ll/HWTopo.h"

#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

static const char* name = "Delaunay Mesh Refinement";
static const char* desc = "Refines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees\n";
static const char* url = "delaunay_mesh_refinement";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

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
    if (!mesh->containsNode(item)) //locks
      return;
    
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
      Element& element = mesh->getData(node,Galois::ALL);
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

int main(int argc, char** argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

  mesh = new Graph();
  Mesh m;
  m.read(mesh, filename.c_str());

  std::cout << "configuration: " << std::distance(mesh->active_begin(), mesh->active_end())
	    << " total triangles, " << std::count_if(mesh->active_begin(), mesh->active_end(), is_bad(mesh)) << " bad triangles\n";

  std::vector<GNode> wl;
  for (Graph::active_iterator ii = mesh->active_begin(), ee = mesh->active_end();
       ii != ee; ++ii)
    if (mesh->getData(*ii).isBad())
      wl.push_back(*ii);
  
  std::cout << "MEMINFO PRE: " << GaloisRuntime::MM::pageAllocInfo() << "\n";

  Galois::preAlloc(5600);
  std::cout << "MEMINFO MID: " << GaloisRuntime::MM::pageAllocInfo() << "\n";


  Galois::StatTimer T;
  T.start();
  using namespace GaloisRuntime::WorkList;
  Galois::for_each<LocalQueues<dChunkedLIFO<256>, LIFO<> > >(wl.begin(),wl.end(), process());
  //Galois::for_each<LocalQueues<dChunkedLIFO<256>, LIFO<> > >(mesh->active_begin(), mesh->active_end(), process(), is_bad(mesh));
  //Galois::for_each<LocalQueues<InitialIterator<std::vector<GNode>::iterator>, LIFO<> > >(wl.begin(), wl.end(), process());
  //Galois::for_each<LocalQueues<InitialIterator<GNode*>, LIFO<> > >(&wl[0], &wl[wl.size()], process());
  //Galois::for_each<dChunkedLIFO<1024> >(wl.begin(), wl.end(), process());
  T.stop();

  std::cout << "MEMINFO POST: " << GaloisRuntime::MM::pageAllocInfo() << "\n";

  if (!skipVerify) {
    if (!m.verify(mesh)) {
      std::cerr << "Refinement failed.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    
    int size = std::count_if(mesh->active_begin(), mesh->active_end(), is_bad(mesh));
    if (size != 0) {
      std::cerr << size << " bad triangles remaining.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    std::cout << "Refinement OK\n";
  }
  return 0;
}

