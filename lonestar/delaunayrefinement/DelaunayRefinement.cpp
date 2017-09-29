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

#include "galois/Galois.h"
#include "galois/ParallelSTL.h"
#include "galois/Bag.h"
#include "galois/Timer.h"

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
  struct LocalState {
    Cavity cav;
    LocalState(Process<Version>& self, galois::PerIterAllocTy& alloc): cav(graph, alloc) { }
  };

  //! [Enabling Per Iteration Allocator in DMR]
  typedef std::tuple<
    galois::local_state<LocalState>,
    galois::per_iter_alloc
  > function_traits;
  //! [Enabling Per Iteration Allocator in DMR]

  void operator()(GNode item, galois::UserContext<GNode>& ctx) {
    if (!graph->containsNode(item, galois::MethodFlag::WRITE))
      return;
    
    if (Version == detDisjoint) {

      LocalState* localState = (LocalState*) ctx.getLocalState();

      if (ctx.isFirstPass()) {
        localState->cav.initialize(item);
        localState->cav.build();
        localState->cav.computePost();
      } else {
        localState->cav.update(item,ctx);
      }

      return;
    } else {
      //! [Accessing Per Iteration Allocator in DMR]
      Cavity cav(graph, ctx.getPerIterAlloc());
      //! [Accessing Per Iteration Allocator in DMR]
      cav.initialize(item);
      cav.build();
      cav.computePost();
      if (Version == detPrefix)
        return;
      ctx.cautiousPoint();
      cav.update(item, ctx);
    }
  }
};

struct Preprocess {
  galois::InsertBag<GNode>& wl;
  Preprocess(galois::InsertBag<GNode>& w): wl(w) { }
  void operator()(GNode item) const {
    if (graph->getData(item, galois::MethodFlag::UNPROTECTED).isBad())
      wl.push(item);
  }
};

struct DetLessThan {
  bool operator()(const GNode& a, const GNode& b) const {
    int idA = graph->getData(a, galois::MethodFlag::UNPROTECTED).getId();
    int idB = graph->getData(b, galois::MethodFlag::UNPROTECTED).getId();
    if (idA == 0 || idB == 0) abort();
    return idA < idB;
  }
};

int main(int argc, char** argv) {
  galois::SharedMemSys G;
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

  galois::reportPageAlloc("MeminfoPre1");
  // Tighter upper bound for pre-alloc, useful for machines with limited memory,
  // e.g., Intel MIC. May not be enough for deterministic execution
  const size_t NODE_SIZE = sizeof(**graph->begin());
  if (detAlgo == nondet) {
    galois::preAlloc (5 * galois::getActiveThreads () + NODE_SIZE * 8 * graph->size () / galois::runtime::pagePoolSize());

  } else {
    galois::preAlloc(galois::getActiveThreads () + NODE_SIZE * 32 * graph->size () / galois::runtime::pagePoolSize());
  }
  galois::reportPageAlloc("MeminfoPre2");

  galois::StatTimer T;
  T.start();

  //! [do_all example]
  galois::InsertBag<GNode> initialBad;

  if (detAlgo == nondet)
    galois::do_all(galois::iterate(*graph), Preprocess(initialBad), galois::loopname("findbad"));
  //! [do_all example]
  else
    std::for_each(graph->begin(), graph->end(), Preprocess(initialBad));

  galois::reportPageAlloc("MeminfoMid");
  
  galois::StatTimer Trefine("refine");
  Trefine.start();
  using namespace galois::worklists;
  
  typedef Deterministic<> DWL;
  //! [for_each example]
  typedef LocalQueue<dChunkedLIFO<256>, ChunkedLIFO<256> > BQ;
  typedef AltChunkedLIFO<32> Chunked;
  
  switch (detAlgo) {
    case nondet: 
      galois::for_each(galois::iterate(initialBad), Process<>(), galois::loopname("refine"), galois::wl<Chunked>());
      break;
      //! [for_each example]
    case detBase:
      galois::for_each(galois::iterate(initialBad), Process<>(), galois::wl<DWL>());
      break;
    case detPrefix:
      galois::for_each(galois::iterate(initialBad), Process<>(),
          galois::wl<DWL>(), galois::neighborhood_visitor<Process<detPrefix>>());
      break;
    case detDisjoint:
      galois::for_each(galois::iterate(initialBad), Process<detDisjoint>(), galois::wl<DWL>());
      break;
    default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
  }
  Trefine.stop();
  T.stop();
  
  galois::reportPageAlloc("MeminfoPost");
  
  if (!skipVerify) {
    int size = galois::ParallelSTL::count_if(graph->begin(), graph->end(), is_bad(graph));
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
