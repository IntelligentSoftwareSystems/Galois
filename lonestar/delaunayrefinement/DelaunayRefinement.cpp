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


template <typename WL, int Version=detBase>
void refine(galois::InsertBag<GNode>& initialBad, Graph& graph) {

  
  struct LocalState {
    Cavity cav;
    LocalState(Graph& graph, galois::PerIterAllocTy& alloc): cav(&graph, alloc) { }
  };

  //! [for_each example]
  galois::for_each( galois::iterate(initialBad),
      [&] (GNode item, auto& ctx) {
        if (!graph.containsNode(item, galois::MethodFlag::WRITE))
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
          Cavity cav(&graph, ctx.getPerIterAlloc());
          //! [Accessing Per Iteration Allocator in DMR]
          cav.initialize(item);
          cav.build();
          cav.computePost();
          if (Version == detPrefix)
            return;
          ctx.cautiousPoint();
          cav.update(item, ctx);
        }
      },
      galois::loopname("refine"),
      galois::wl<WL>(),
      galois::per_iter_alloc(),
      galois::local_state<LocalState>());

  //! [for_each example]

}

template <typename Loop>
void findBad(Graph& graph, galois::InsertBag<GNode>& initialBad, const Loop& loop) {
  loop( galois::iterate(graph),
      [&] (GNode item) {
        if (graph.getData(item, galois::MethodFlag::UNPROTECTED).isBad()) {
          initialBad.push(item);
        }
      }, galois::loopname("findBad"));

}

/*
struct DetLessThan {
  bool operator()(const GNode& a, const GNode& b) const {
    int idA = graph.getData(a, galois::MethodFlag::UNPROTECTED).getId();
    int idB = graph.getData(b, galois::MethodFlag::UNPROTECTED).getId();
    if (idA == 0 || idB == 0) abort();
    return idA < idB;
  }
};
*/

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph;
  {
    Mesh m;
    m.read(graph, filename.c_str(), detAlgo == nondet);
    Verifier v;
    if (!skipVerify && !v.verify(graph)) {
      GALOIS_DIE("bad input mesh");
    }
  }
  std::cout << "configuration: " << std::distance(graph.begin(), graph.end())
	    << " total triangles, " << std::count_if(graph.begin(), graph.end(), is_bad(graph)) << " bad triangles\n";

  galois::reportPageAlloc("MeminfoPre1");
  // Tighter upper bound for pre-alloc, useful for machines with limited memory,
  // e.g., Intel MIC. May not be enough for deterministic execution
  constexpr size_t NODE_SIZE = sizeof(**graph.begin());
  galois::preAlloc (5 * galois::getActiveThreads () + NODE_SIZE * 32 * graph.size () / galois::runtime::pagePoolSize());

  galois::reportPageAlloc("MeminfoPre2");

  galois::StatTimer T;
  T.start();

  galois::InsertBag<GNode> initialBad;

  if (detAlgo == nondet) {
    findBad(graph, initialBad, galois::DoAll());
  }
  else {
    findBad(graph, initialBad, galois::StdForEach());
  }

  galois::reportPageAlloc("MeminfoMid");
  
  galois::StatTimer Trefine("refine");
  Trefine.start();
  using namespace galois::worklists;
  
  typedef Deterministic<> DWL;
  typedef LocalQueue<dChunkedLIFO<256>, ChunkedLIFO<256> > BQ;
  typedef AltChunkedLIFO<32> Chunked;
  
  switch (detAlgo) {
    case nondet: 
      refine<Chunked>(initialBad, graph);
      break;
    case detBase:
      // refine<DWL>(initialBad, graph);
      break;
    case detPrefix:
      // refine<DWL, detPrefix>(initialBad, graph);
      break;
    case detDisjoint:
      // refine<DWL, detDisjoint>(initialBad, graph);
      break;
    default: std::cerr << "Unknown algorithm" << detAlgo << "\n"; abort();
  }
  Trefine.stop();
  T.stop();
  
  galois::reportPageAlloc("MeminfoPost");
  
  if (!skipVerify) {
    int size = galois::ParallelSTL::count_if(graph.begin(), graph.end(), is_bad(graph));
    if (size != 0) {
      GALOIS_DIE("Bad triangles remaining");
    }
    Verifier v;
    if (!v.verify(graph)) {
      GALOIS_DIE("Refinement failed");
    }
    std::cout << std::distance(graph.begin(), graph.end()) << " total triangles\n";
    std::cout << "Refinement OK\n";
  }

  return 0;
}
