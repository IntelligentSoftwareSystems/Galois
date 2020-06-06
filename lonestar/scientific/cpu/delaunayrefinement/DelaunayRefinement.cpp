/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
static const char* desc = "Refines a Delaunay triangulation mesh such that no "
                          "angle in the mesh is less than 30 degrees";
static const char* url = "delaunay_mesh_refinement";

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

enum DetAlgo { nondet, detBase, detPrefix, detDisjoint };

static cll::opt<DetAlgo>
    detAlgo(cll::desc("Deterministic schedule (default value nondet):"),
            cll::values(clEnumVal(nondet, "Non-deterministic"),
                        clEnumVal(detBase, "Base execution"),
                        clEnumVal(detPrefix, "Prefix execution"),
                        clEnumVal(detDisjoint, "Disjoint execution")),
            cll::init(nondet));

//! Flag that forces user to be aware that they should be passing in a
//! mesh graph.
static cll::opt<bool>
    meshGraph("meshGraph", cll::desc("Specify that the input graph is a mesh"),
              cll::init(false));

template <typename WL, int Version = detBase>
void refine(galois::InsertBag<GNode>& initialBad, Graph& graph) {

  struct LocalState {
    Cavity cav;
    LocalState(Graph& graph, galois::PerIterAllocTy& alloc)
        : cav(&graph, alloc) {}
  };

  //! [for_each example]
  galois::for_each(
      galois::iterate(initialBad),
      [&](GNode item, auto& ctx) {
        if (!graph.containsNode(item, galois::MethodFlag::WRITE))
          return;

        if (Version == detDisjoint) {

          if (ctx.isFirstPass()) {
            LocalState* localState = ctx.template createLocalState<LocalState>(
                graph, ctx.getPerIterAlloc());
            localState->cav.initialize(item);
            localState->cav.build();
            localState->cav.computePost();
          } else {
            LocalState* localState = ctx.template getLocalState<LocalState>();
            localState->cav.update(item, ctx);
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
      galois::loopname("refine"), galois::wl<WL>(), galois::per_iter_alloc(),
      galois::local_state<LocalState>());

  //! [for_each example]
}

template <typename Loop>
void findBad(Graph& graph, galois::InsertBag<GNode>& initialBad,
             const Loop& loop) {
  loop(
      galois::iterate(graph),
      [&](GNode item) {
        if (graph.getData(item, galois::MethodFlag::UNPROTECTED).isBad()) {
          initialBad.push(item);
        }
      },
      galois::loopname("findBad"));
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
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  if (!meshGraph) {
    GALOIS_DIE("This application requires a mesh graph input;"
               " please use the -meshGraph flag "
               " to indicate the input is a mesh graph.");
  }

  Graph graph;
  {
    Mesh m;
    m.read(graph, inputFile.c_str(), detAlgo == nondet);
    Verifier v;
    if (!skipVerify && !v.verify(graph)) {
      GALOIS_DIE("bad input mesh");
    }
  }
  std::cout << "configuration: " << std::distance(graph.begin(), graph.end())
            << " total triangles, "
            << std::count_if(graph.begin(), graph.end(), is_bad(graph))
            << " bad triangles\n";

  galois::reportPageAlloc("MeminfoPre1");
  // Tighter upper bound for pre-alloc, useful for machines with limited memory,
  // e.g., Intel MIC. May not be enough for deterministic execution
  constexpr size_t NODE_SIZE = sizeof(**graph.begin());
  galois::preAlloc(5 * galois::getActiveThreads() +
                   NODE_SIZE * 32 * graph.size() /
                       galois::runtime::pagePoolSize());

  galois::reportPageAlloc("MeminfoPre2");

  galois::StatTimer execTime("Timer_0");
  execTime.start();

  galois::InsertBag<GNode> initialBad;

  if (detAlgo == nondet) {
    findBad(graph, initialBad, galois::DoAll());
  } else {
    findBad(graph, initialBad, galois::StdForEach());
  }

  galois::reportPageAlloc("MeminfoMid");

  galois::StatTimer Trefine("refine");
  Trefine.start();
  using namespace galois::worklists;

  typedef Deterministic<> DWL;
  typedef PerThreadChunkLIFO<32> Chunk;

  switch (detAlgo) {
  case nondet:
    refine<Chunk>(initialBad, graph);
    break;
  case detBase:
    refine<DWL>(initialBad, graph);
    break;
  case detPrefix:
    refine<DWL, detPrefix>(initialBad, graph);
    break;
  case detDisjoint:
    refine<DWL, detDisjoint>(initialBad, graph);
    break;
  default:
    std::cerr << "Unknown algorithm" << detAlgo << "\n";
    abort();
  }
  Trefine.stop();
  execTime.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    int size = galois::ParallelSTL::count_if(graph.begin(), graph.end(),
                                             is_bad(graph));
    if (size != 0) {
      GALOIS_DIE("bad triangles remaining");
    }
    Verifier v;
    if (!v.verify(graph)) {
      GALOIS_DIE("refinement failed");
    }
    std::cout << std::distance(graph.begin(), graph.end())
              << " total triangles\n";
    std::cout << "Refinement OK\n";
  }

  totalTime.stop();

  return 0;
}
