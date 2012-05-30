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

#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph2.h"

#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "Galois/Runtime/WorkListAlt.h"
#include "Galois/Runtime/WorkListDebug.h"

#ifdef GALOIS_DET
#include "Galois/Runtime/Deterministic.h"
#endif

namespace cll = llvm::cl;

static const char* name = "Delaunay Mesh Refinement";
static const char* desc = "Refines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees\n";
static const char* url = "delaunay_mesh_refinement";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

typedef Galois::Graph::FirstGraph<Element,void,false> Graph;
typedef Graph::GraphNode GNode;

#include "Subgraph.h"
#include "Mesh.h"
#include "Cavity.h"
#include "Verifier.h"

Graph* mesh;

struct Process {
  typedef int tt_needs_per_iter_alloc;

  template<typename Context>
  void operator()(GNode item, Context& lwl) {
    if (!mesh->containsNode(item, Galois::ALL))
      return;
    
    Cavity cav(mesh, lwl.getPerIterAlloc());
    cav.initialize(item);
    cav.build();
    cav.update(); //VTune: Most work
    
    //FAILSAFE POINT

    for (PreGraph::iterator ii = cav.getPre().begin(),
	   ee = cav.getPre().end(); ii != ee; ++ii) 
      mesh->removeNode(*ii, Galois::NONE);
    
    //add new data
    for (PostGraph::iterator ii = cav.getPost().begin(),
	   ee = cav.getPost().end(); ii != ee; ++ii) {
      GNode node = *ii;
      mesh->addNode(node, Galois::NONE);
      Element& element = mesh->getData(node, Galois::NONE);
      if (element.isBad()) {
        lwl.push(node);
      }
    }
    
    for (PostGraph::edge_iterator ii = cav.getPost().edge_begin(),
	   ee = cav.getPost().edge_end(); ii != ee; ++ii) {
      EdgeTuple edge = *ii;
      mesh->addEdge(edge.src, edge.dst, Galois::NONE);
    }

    if (mesh->containsNode(item, Galois::NONE)) {
      lwl.push(item);
    }
  }
};

GaloisRuntime::galois_insert_bag<GNode> wl;

struct Preprocess {
  void operator()(GNode item) const {
    if (mesh->getData(item, Galois::NONE).isBad())
      wl.push(item);
  }
  // void operator()(Graph::GTile item) const {
  //   for (Graph::GTile::iterator ii = item.begin(), ee = item.end();
  // 	 ii != ee; ++ii)
  //     if (mesh->getData(*ii, Galois::NONE).isBad())
  // 	wl.push(*ii);
  // }
};

struct LessThan {
  bool operator()(const GNode& a, const GNode& b) const {
    int idA = mesh->getData(a, Galois::NONE).id;
    int idB = mesh->getData(b, Galois::NONE).id;
    if (idA == 0 || idB == 0)
      abort();
    return idA < idB;
  }
};

static ptrdiff_t myrandom(ptrdiff_t i) { return rand() % i; }

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, std::cout, name, desc, url);

  mesh = new Graph();
  {
    Mesh m;
    m.read(mesh, filename.c_str());
    Verifier v;
    if (!skipVerify && !v.verify(mesh)) {
      std::cerr << "bad input mesh\n";
      assert(0 && "Refinement failed");
      abort();
    }
  }

  std::cout << "configuration: " << std::distance(mesh->begin(), mesh->end())
	    << " total triangles, " << std::count_if(mesh->begin(), mesh->end(), is_bad(mesh)) << " bad triangles\n";

  Galois::Statistic("MeminfoPre1", GaloisRuntime::MM::pageAllocInfo());
  Galois::preAlloc(15 * numThreads + GaloisRuntime::MM::pageAllocInfo() * 10);
  Galois::Statistic("MeminfoPre2", GaloisRuntime::MM::pageAllocInfo());

  Galois::StatTimer T;
  T.start();

#ifdef GALOIS_DET
  std::for_each(mesh->begin(), mesh->end(), Preprocess());
  std::vector<GNode> wlnew;
  std::copy(wl.begin(), wl.end(), std::back_inserter(wlnew));
  std::sort(wlnew.begin(), wlnew.end(), LessThan());
  ptrdiff_t (*myptr)(ptrdiff_t) = myrandom;
  srand(0xDEADBEEF);
  std::random_shuffle(wlnew.begin(), wlnew.end(), myptr);
#else
  Galois::do_all_local(*mesh, Preprocess());
  //Galois::do_all(mesh->tile_begin(), mesh->tile_end(), Preprocess());
#endif
  Galois::Statistic("MeminfoMid", GaloisRuntime::MM::pageAllocInfo());

  Galois::StatTimer Trefine("refine");
  Trefine.start();
  using namespace GaloisRuntime::WorkList;
#ifdef GALOIS_DET
  Galois::for_each<Deterministic<> >(wlnew.begin(), wlnew.end(), Process());
#else
  typedef LocalQueues<dChunkedLIFO<256>, ChunkedLIFO<256> > BQ;
  typedef LoadBalanceTracker<BQ, 2048 > DBQ;
  typedef ChunkedAdaptor<false,32> CA;

  typedef PerThreadQueues<LIFO<> > SHP;
  Galois::for_each_local<CA>(wl, Process());
  //Galois::for_each<SHP>(wl.begin(), wl.end(), Process());
#endif
  Trefine.stop();
  T.stop();

  Galois::Statistic("MeminfoPost", GaloisRuntime::MM::pageAllocInfo());

  if (!skipVerify) {
    int size = Galois::count_if(mesh->begin(), mesh->end(), is_bad(mesh));
    if (size != 0) {
      std::cerr << size << " bad triangles remaining.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    Verifier v;
    if (!v.verify(mesh)) {
      std::cerr << "Refinement failed.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    std::cout << "Refinement OK\n";
  }
  return 0;
}

