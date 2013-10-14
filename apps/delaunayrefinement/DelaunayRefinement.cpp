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

#include <iostream>
#include <string.h>
#include <cassert>

namespace cll = llvm::cl;

static const char* name = "Delaunay Mesh Refinement";
static const char* desc = "Refines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees";
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
  typedef LocalState GaloisDeterministicLocalState;
  static_assert(Galois::has_deterministic_local_state<Process>::value, "Oops");

  void operator()(GNode item, Galois::UserContext<GNode>& ctx) {
    if (!graph->containsNode(item, Galois::MethodFlag::ALL))
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

struct Preprocess {
  Galois::InsertBag<GNode>& wl;
  Preprocess(Galois::InsertBag<GNode>& w): wl(w) { }
  void operator()(GNode item) const {
    if (graph->getData(item, Galois::MethodFlag::NONE).isBad())
      wl.push(item);
  }
};

struct DetLessThan {
  bool operator()(const GNode& a, const GNode& b) const {
    int idA = graph->getData(a, Galois::MethodFlag::NONE).getId();
    int idB = graph->getData(b, Galois::MethodFlag::NONE).getId();
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
      GALOIS_DIE("bad input mesh");
    }
  }
  std::cout << "configuration: " << std::distance(graph->begin(), graph->end())
	    << " total triangles, " << std::count_if(graph->begin(), graph->end(), is_bad(graph)) << " bad triangles\n";

  Galois::reportPageAlloc("MeminfoPre1");
  Galois::preAlloc(Galois::Runtime::MM::numPageAllocTotal() * 10);
  //Galois::preAlloc(15 * numThreads + Galois::Runtime::MM::numPageAllocTotal() * 10);
  Galois::reportPageAlloc("MeminfoPre2");

  Galois::StatTimer T;
  T.start();

  Galois::InsertBag<GNode> initialBad;

  if (detAlgo == nondet)
    Galois::do_all_local(*graph, Preprocess(initialBad), Galois::loopname("findbad"));
  else
    std::for_each(graph->begin(), graph->end(), Preprocess(initialBad));

  Galois::reportPageAlloc("MeminfoMid");
  
  Galois::StatTimer Trefine("refine");
  Trefine.start();
  using namespace Galois::WorkList;
  
  typedef LocalQueue<dChunkedLIFO<256>, ChunkedLIFO<256> > BQ;
  typedef AltChunkedLIFO<32> Chunked;
  
  switch (detAlgo) {
    case nondet: 
      Galois::for_each_local(initialBad, Process<>(), Galois::loopname("refine"), Galois::wl<Chunked>());
    case detBase:
      Galois::for_each_det(initialBad.begin(), initialBad.end(), Process<>()); break;
    case detPrefix:
      Galois::for_each_det(initialBad.begin(), initialBad.end(), Process<detPrefix>(), Process<>());
      break;
    case detDisjoint:
      Galois::for_each_det(initialBad.begin(), initialBad.end(), Process<detDisjoint>()); break;
    default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
  }
  Trefine.stop();
  T.stop();
  
  Galois::reportPageAlloc("MeminfoPost");
  
  if (!skipVerify) {
    int size = Galois::ParallelSTL::count_if(graph->begin(), graph->end(), is_bad(graph));
    if (size != 0) {
      GALOIS_DIE("Bad triangles remaining");
    }
    Verifier v;
    if (!v.verify(graph)) {
      GALOIS_DIE("Refinement failed");
    }
    std::cout << std::distance(graph->begin(), graph->end()) << " total triangles\n";
    std::cout << "Refinement OK\n";
  }

  return 0;
}
