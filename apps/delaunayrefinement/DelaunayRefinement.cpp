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
#include "Mesh.h"
#include "Cavity.h"
#include "Verifier.h"

#include "Galois/Galois.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "Galois/Runtime/WorkListAlt.h"
#include "Galois/Runtime/WorkListDebug.h"

#include <iostream>
#include <string.h>
#include <cassert>

namespace cll = llvm::cl;

static const char* name = "Delaunay Mesh Refinement";
static const char* desc = "Refines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees\n";
static const char* url = "delaunay_mesh_refinement";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

Graph* graph;

enum DetAlgo {
  nondet,
  detBase,
  detPrefix,
  detDisjoint
};

static cll::opt<DetAlgo> detAlgo(cll::desc("Deterministic algorithm:"),
    cll::values(
      clEnumVal(nondet, "Non-deterministic"),
      clEnumVal(detBase, "Base execution"),
      clEnumVal(detPrefix, "Prefix execution"),
      clEnumVal(detDisjoint, "Disjoint execution"),
      clEnumValEnd), cll::init(nondet));

template<int Version=detBase>
struct Process {
  typedef int tt_needs_per_iter_alloc;

  struct LocalState {
    Cavity cav;
    LocalState(Process<Version>& self, Galois::PerIterAllocTy& alloc): cav(graph, alloc) { }
  };

  void operator()(GNode item, Galois::UserContext<GNode>& ctx) {
    if (!graph->containsNode(item, Galois::ALL))
      return;
    
    Cavity* cavp = NULL;

    if (Version == detDisjoint) {
      bool used;
      LocalState* localState = (LocalState*) ctx.getLocalState(used);
      if (used) {
        localState->cav.update(item, ctx);
        return;
      } else {
        cavp = &localState->cav;
      }
    }

    if (Version == detDisjoint) {
      cavp->initialize(item);
      cavp->build();
      cavp->computePost();
    } else {
      Cavity cav(graph, ctx.getPerIterAlloc());
      cav.initialize(item);
      cav.build();
      cav.computePost();
      if (Version == detPrefix)
        return;
      cav.update(item, ctx);
    }
  }
};

Galois::InsertBag<GNode> wl;

struct Preprocess {
  void operator()(GNode item) const {
    if (graph->getData(item, Galois::NONE).isBad())
      wl.push(item);
  }
};

struct DetLessThan {
  bool operator()(const GNode& a, const GNode& b) const {
    int idA = graph->getData(a, Galois::NONE).getId();
    int idB = graph->getData(b, Galois::NONE).getId();
    if (idA == 0 || idB == 0) abort();
    return idA < idB;
  }
};

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  graph = new Graph();
  {
    Mesh m;
    m.read(graph, filename.c_str(), detAlgo == nondet);
    Verifier v;
    if (!skipVerify && !v.verify(graph)) {
      std::cerr << "bad input mesh\n";
      assert(0 && "Refinement failed");
      abort();
    }
  }

  std::cout << "configuration: " << std::distance(graph->begin(), graph->end())
	    << " total triangles, " << std::count_if(graph->begin(), graph->end(), is_bad(graph)) << " bad triangles\n";

  Galois::Statistic("MeminfoPre1", GaloisRuntime::MM::pageAllocInfo());
  Galois::preAlloc(15 * numThreads + GaloisRuntime::MM::pageAllocInfo() * 10);
  Galois::Statistic("MeminfoPre2", GaloisRuntime::MM::pageAllocInfo());

  Galois::StatTimer T;
  T.start();

  if (detAlgo == nondet)
    Galois::do_all_local(*graph, Preprocess());
  else
    std::for_each(graph->begin(), graph->end(), Preprocess());

  Galois::Statistic("MeminfoMid", GaloisRuntime::MM::pageAllocInfo());
  
  Galois::StatTimer Trefine("refine");
  Trefine.start();
  using namespace GaloisRuntime::WorkList;
  
  typedef LocalQueues<dChunkedLIFO<256>, ChunkedLIFO<256> > BQ;
  typedef ChunkedAdaptor<false,32> CA;
  
  switch (detAlgo) {
    case nondet: 
      Galois::for_each_local<CA>(wl, Process<>()); break;
    case detBase:
      Galois::for_each_det(wl.begin(), wl.end(), Process<>()); break;
    case detPrefix:
      Galois::for_each_det(wl.begin(), wl.end(), Process<detPrefix>(), Process<>());
      break;
    case detDisjoint:
      Galois::for_each_det(wl.begin(), wl.end(), Process<detDisjoint>()); break;
    default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
  }
  Trefine.stop();
  T.stop();
  
  Galois::Statistic("MeminfoPost", GaloisRuntime::MM::pageAllocInfo());
  
  if (!skipVerify) {
    int size = Galois::ParallelSTL::count_if(graph->begin(), graph->end(), is_bad(graph));
    if (size != 0) {
      std::cerr << size << " bad triangles remaining.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    Verifier v;
    if (!v.verify(graph)) {
      std::cerr << "Refinement failed.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    std::cout << "Refinement OK\n";
  }

  return 0;
}
