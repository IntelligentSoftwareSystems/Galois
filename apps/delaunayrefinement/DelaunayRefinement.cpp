/** Delaunay refinement -*- C++ -*-
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
 * Refinement of an initial, unrefined Delaunay mesh to eliminate triangles
 * with angles < 30 degrees, using a variation of Chew's algorithm.
 *
 * @author Milind Kulkarni <milind@purdue.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include <iostream>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <cassert>

#include "Element.h"

#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph2.h"
#include "Galois/Galois.h"

#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#ifdef GALOIS_EXP
#include "Galois/Runtime/WorkListAlt.h"
#include "PriorityScheduling/WorkList.h"
#endif

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
    if (!mesh->containsNode(item))
      return;
    
    Cavity cav(mesh, lwl.getPerIterAlloc());
    cav.initialize(item);
    cav.build();
    cav.update();
    
    //FAILSAFE POINT

    for (Subgraph::iterator ii = cav.getPre().begin(),
	   ee = cav.getPre().end(); ii != ee; ++ii) 
      mesh->removeNode(*ii, Galois::NONE);
    
    //add new data
    for (Subgraph::iterator ii = cav.getPost().begin(),
	   ee = cav.getPost().end(); ii != ee; ++ii) {
      GNode node = *ii;
      Element& element = mesh->getData(node,Galois::NONE);
      if (element.isBad()) {
        lwl.push(node);
      }
    }
    
    for (Subgraph::edge_iterator ii = cav.getPost().edge_begin(),
	   ee = cav.getPost().edge_end(); ii != ee; ++ii) {
      Subgraph::tmpEdge edge = *ii;
      mesh->addEdge(edge.src, edge.dst, Galois::NONE);
    }
    if (mesh->containsNode(item)) {
      lwl.push(item);
    }
  }
};

bool verify() {
  // ensure consistency of elements
  bool error = false;
  
  for (Graph::iterator ii = mesh->begin(), ee = mesh->end(); ii != ee; ++ii) {
    
    GNode node = *ii;
    Element& element = mesh->getData(node,Galois::NONE);
    int nsize = std::distance(mesh->edge_begin(node, Galois::NONE), mesh->edge_end(node, Galois::NONE));
    if (element.getDim() == 2) {
      if (nsize != 1) {
	std::cerr << "-> Segment " << element << " has " << nsize << " relation(s)\n";
	error = true;
      }
    } else if (element.getDim() == 3) {
      if (nsize != 3) {
	std::cerr << "-> Triangle " << element << " has " << nsize << " relation(s)";
	error = true;
      }
    } else {
      std::cerr << "-> Figures with " << element.getDim() << " edges";
      error = true;
    }
  }
  
  if (error)
    return false;
  
  // ensure reachability
  std::stack<GNode> remaining;
  std::set<GNode> found;
  remaining.push(*(mesh->begin()));
  
  while (!remaining.empty()) {
    GNode node = remaining.top();
    remaining.pop();
    if (!found.count(node)) {
      assert(mesh->containsNode(node) && "Reachable node was removed from graph");
      found.insert(node);
      int i = 0;
      for (Graph::edge_iterator ii = mesh->edge_begin(node, Galois::NONE), ee = mesh->edge_end(node, Galois::NONE); ii != ee; ++ii) {
	assert(i < 3);
	assert(mesh->containsNode(mesh->getEdgeDst(ii)));
	assert(node != mesh->getEdgeDst(ii));
	++i;
	//          if (!found.count(*ii))
	remaining.push(mesh->getEdgeDst(ii));
      }
    }
  }
  size_t msize = std::distance(mesh->begin(), mesh->end());
  
  if (found.size() != msize) {
    std::cerr << "Not all elements are reachable \n";
    std::cerr << "Found: " << found.size() << "\nMesh: " << msize << "\n";
    assert(0 && "Not all elements are reachable");
    return false;
  }
  return true;
}

GaloisRuntime::galois_insert_bag<GNode> wl;

struct preprocess {
  template<typename Context>
  void operator()(GNode item, Context& lwl) const {
    if (mesh->getData(item, Galois::NONE).isBad())
      wl.push(item);
  }
};



struct Indexer: public std::unary_function<const GNode&,unsigned> {
  unsigned operator()(const GNode& t) const { return 0; }
};

struct Less: public std::binary_function<const GNode&,const GNode&,bool>{
  bool operator()(const GNode& a, const GNode& b) const {
    return true;
  }
};

struct Greater: public std::binary_function<const GNode&,const GNode&,bool> {
  bool operator()(const GNode& a, const GNode& b) const {
    return true;
  }
};

int main(int argc, char** argv) {
  LonestarStart(argc, argv, std::cout, name, desc, url);

  mesh = new Graph();
  {
    Mesh m;
    m.read(mesh, filename.c_str());
  }

  std::cout << "configuration: " << std::distance(mesh->begin(), mesh->end())
	    << " total triangles, " << std::count_if(mesh->begin(), mesh->end(), is_bad(mesh)) << " bad triangles\n";

  Galois::StatTimer Touter("outertime");
  Touter.start();

  std::cout << "MEMINFO P1: " << GaloisRuntime::MM::pageAllocInfo() << "\n";
#ifdef GALOIS_EXP
  //  Galois::do_all(*mesh, preprocess());
  Galois::for_each(mesh->begin(), mesh->end(), preprocess());
#else
  for (Graph::iterator ii = mesh->begin(), ee = mesh->end(); ii != ee; ++ii)
    if (mesh->getData(*ii).isBad())
      wl.push(*ii);
#endif
  std::cout << "MEMINFO P2: " << GaloisRuntime::MM::pageAllocInfo() << "\n";

  Galois::preAlloc(10 * numThreads + GaloisRuntime::MM::pageAllocInfo() * 5);
  std::cout << "MEMINFO MID: " << GaloisRuntime::MM::pageAllocInfo() << "\n";

  Galois::StatTimer T;
  T.start();
  using namespace GaloisRuntime::WorkList;
#ifdef GALOIS_EXP
  //Galois::for_each<Alt::ChunkedAdaptor<Alt::InitialQueue<Alt::LevelStealingAlt, Alt::LevelLocalAlt>, 256*4*4> >(wl.begin(), wl.end(), process());
  typedef dChunkedLIFO<256> dChunk;
  typedef ChunkedLIFO<256> Chunk;
  Exp::StartWorklistExperiment<
    LocalQueues<dChunk, LIFO<> >, 
    dChunk,Chunk,Indexer,Less,Greater>()(
      std::cout, wl.begin(), wl.end(), process());
#else
  Galois::for_each<LocalQueues<dChunkedLIFO<256>, LIFO<> > >(wl.begin(), wl.end(), process());
#endif
  T.stop();
  Touter.stop();

  std::cout << "MEMINFO POST: " << GaloisRuntime::MM::pageAllocInfo() << "\n";

  if (!skipVerify) {
    int size = std::count_if(mesh->begin(), mesh->end(), is_bad(mesh));
    if (size != 0) {
      std::cerr << size << " bad triangles remaining.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    if (!verify()) {
      std::cerr << "Refinement failed.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    std::cout << "Refinement OK\n";
  }
  return 0;
}

