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
#include "Galois/Graphs/Bag.h"
#include "Galois/Timer.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "Galois/WorkList/WorkListAlt.h"
#include "Galois/WorkList/WorkListDebug.h"

#include <iostream>
#include <string.h>
#include <cassert>

namespace cll = llvm::cl;

static const char* name = "Delaunay Mesh Refinement";
static const char* desc = "Refines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees\n";
static const char* url = "delaunay_mesh_refinement";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

struct Process : public Galois::Runtime::Lockable {
  Graphp   graph;

  typedef int tt_needs_per_iter_alloc;

  Process(Graphp g): graph(g) {}
  Process() {}

  void operator()(GNode node, Galois::UserContext<GNode>& ctx) {
    if (!graph->containsNode(node))
      return;

    Cavity cav(graph, ctx.getPerIterAlloc());
    cav.initialize(node);
    cav.build();
    cav.computePost();
    cav.update(node, ctx);
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,graph);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,graph);
  }
};

struct Preprocess : public Galois::Runtime::Lockable {
  Graphp   graph;
  Galois::Graph::Bag<GNode>::pointer wl;

  Preprocess(Graphp g, Galois::Graph::Bag<GNode>::pointer wlg): graph(g), wl(wlg) {}
  Preprocess() {}

  void operator()(GNode item, Galois::UserContext<GNode>& ctx) const {
    if (graph->getData(item).isBad())
      wl->push(item);
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,graph,wl);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,graph,wl);
  }
};

struct Verification : public Galois::Runtime::Lockable {
  Graphp   graph;

  Verification(Graphp g): graph(g) {}
  Verification() {}

  void operator()(GNode item, Galois::UserContext<GNode>& ctx) const {
    if (graph->getData(item).isBad()) {
      printf("Found bad triangle\n");
    }
  }

  //serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,graph);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,graph);
  }
};

struct Prefetch : public Galois::Runtime::Lockable {
  Graphp   graph;

  Prefetch(Graphp g): graph(g) {}
  Prefetch() {}

  void operator()(GNode item, Galois::UserContext<GNode>& ctx) const {
    (void)graph->getData(item).isBad();
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,graph);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,graph);
  }
};

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  // check the host id and initialise the network
  Galois::Runtime::NetworkInterface::start();
  //Galois::Runtime::setTrace(false);

  Graphp graph = Graph::allocate();
  {
    Mesh m;
    std::cout << "Going to read files\n";
    m.read(graph, filename.c_str());
    Verifier v;
    if (!skipVerify && !v.verify(graph)) {
      GALOIS_DIE("bad input mesh");
    }
  }
  //  std::cout << "start configuration: " << NThirdGraphSize(graph) << " total triangles, ";
  //  std::cout << Galois::ParallelSTL::count_if_local(graph, is_bad(graph)) << " bad triangles\n";
  //ThirdGraphSize(graph);

  // call prefetch to get the nodes to the owner
  Galois::StatTimer Tprefetch;

  Tprefetch.start();
  std::cout << "beginning prefetch\n";
  Galois::for_each_local(graph, Prefetch(graph), 
      Galois::loopname("prefetch"), Galois::wl<Galois::WorkList::AltChunkedLIFO<32>>());
  //Galois::Runtime::setTrace(true);
  Tprefetch.stop();

  Galois::StatTimer Tprealloc;
  Tprealloc.start();
  std::cout << "beginning prealloc\n";
  Galois::reportPageAlloc("MeminfoPre1");
  // Galois::preAlloc(Galois::Runtime::MM::numPageAllocTotal() * 10);
  // Tighter upper bound for pre-alloc, useful for machines with limited memory,
  // e.g., Intel MIC. May not be enough for deterministic execution
  const size_t nodeSize = sizeof(**graph->begin());
  auto graphSize = Galois::ParallelSTL::count_if_local(graph, [&](GNode) { return true; });
  Galois::preAlloc(5 * Galois::getActiveThreads() + nodeSize * 8 * graphSize / Galois::Runtime::MM::hugePageSize);
  // Relaxed upper bound
  // Galois::preAlloc(15 * numThreads + Galois::Runtime::MM::numPageAllocTotal() * 10);
  Galois::reportPageAlloc("MeminfoPre2");
  Tprealloc.stop();

  Galois::Graph::Bag<GNode>::pointer gwl = Galois::Graph::Bag<GNode>::allocate();

  Galois::StatTimer Tb("BIGGER");
  Tb.start();
  Galois::StatTimer T;
  Galois::StatTimer Tfindbad("findbad");
  T.start();
  Tfindbad.start();
  std::cout << "beginning findbad\n";
  Galois::for_each_local(graph, Preprocess(graph,gwl),
      Galois::loopname("findbad"), Galois::wl<Galois::WorkList::AltChunkedLIFO<32>>());
  Tfindbad.stop();

  Galois::reportPageAlloc("MeminfoMid");

  Galois::StatTimer Trefine("refine");
  Trefine.start();
  using namespace Galois::WorkList;
  
      //! [for_each_local example]
  typedef LocalQueue<dChunkedLIFO<256>, ChunkedLIFO<256> > BQ;
  typedef AltChunkedLIFO<32> Chunked;

  std::cout << "beginning refine\n";
  Galois::for_each_local(gwl, Process(graph), Galois::loopname("refine"), Galois::wl<Chunked>());
  Trefine.stop();
  T.stop();
  Tb.stop();

  std::cout << "beginning verify\n";
  Galois::for_each_local(graph, Verification(graph), Galois::loopname("verification"), Galois::wl<Chunked>());

  //  std::cout << "final configuration: " << NThirdGraphSize(graph) << " total triangles, ";
  //  std::cout << Galois::ParallelSTL::count_if_local(graph, is_bad(graph)) << " bad triangles\n";

  Galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    int size = Galois::ParallelSTL::count_if_local(graph, is_bad(graph));
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

  // master_terminate();
  Galois::Runtime::NetworkInterface::terminate();

  return 0;
}
