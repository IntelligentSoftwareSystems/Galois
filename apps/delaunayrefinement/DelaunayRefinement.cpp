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

struct Process {
  Graphp   graph;
  WLGraphp wlgraph;

  typedef int tt_needs_per_iter_alloc;

  Process(Graphp g, WLGraphp wlg): graph(g), wlgraph(wlg) {}
  Process() {}

  void operator()(WLGNode item, Galois::UserContext<WLGNode>& ctx) {
    GNode node = wlgraph->getData(item);
    if (!graph->containsNode(node))
      return;
    
    Cavity cav(graph, ctx.getPerIterAlloc());
    cav.initialize(node);
    cav.build();
    cav.computePost();
    cav.update(node, wlgraph, ctx);
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,graph);
    gSerialize(s,wlgraph);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,graph);
    gDeserialize(s,wlgraph);
  }
};

struct Preprocess {
  Graphp   graph;
  WLGraphp wlgraph;

  Preprocess(Graphp g, WLGraphp wlg): graph(g), wlgraph(wlg) {}
  Preprocess() {}

  void operator()(GNode item, Galois::UserContext<GNode>& ctx) const {
    if (graph->getData(item).isBad()) {
      WLGNode n = wlgraph->createNode(item);
      wlgraph->addNode(n);
    }
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,graph);
    gSerialize(s,wlgraph);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,graph);
    gDeserialize(s,wlgraph);
  }
};

struct Prefetch {
  Graphp   graph;

  Prefetch(Graphp g): graph(g) {}
  Prefetch() {}

  void operator()(GNode item, Galois::UserContext<GNode>& ctx) const {
    (void)graph->getData(item).isBad();
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,graph);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,graph);
  }
};

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  // check the host id and initialise the network
  Galois::Runtime::Distributed::networkStart();

  Graphp graph(new Graph());
  {
    Mesh m;
    m.read(graph, filename.c_str());
    Verifier v;
    if (!skipVerify && !v.verify(graph)) {
      std::cerr << "bad input mesh\n";
      assert(0 && "Refinement failed");
      abort();
    }
  }

  std::cout << "start configuration: " << NThirdGraphSize(graph) << " total triangles, ";
  std::cout << Galois::ParallelSTL::count_if_local(graph, is_bad(graph)) << " bad triangles\n";
  //ThirdGraphSize(graph);

  // call prefetch to get the nodes to the owner
  Galois::for_each_local(graph, Prefetch(graph));

  Galois::Statistic("MeminfoPre1", Galois::Runtime::MM::pageAllocInfo());
  //Galois::preAlloc(15 * numThreads + Galois::Runtime::MM::pageAllocInfo() * 10);
  Galois::Statistic("MeminfoPre2", Galois::Runtime::MM::pageAllocInfo());

  WLGraphp gwl(new WLGraph());

  Galois::StatTimer T;
  T.start();

  Galois::for_each_local(graph, Preprocess(graph,gwl));

  Galois::Statistic("MeminfoMid", Galois::Runtime::MM::pageAllocInfo());
  
  Galois::StatTimer Trefine("refine");
  Trefine.start();
  using namespace Galois::WorkList;
  
  typedef LocalQueues<dChunkedLIFO<256>, ChunkedLIFO<256> > BQ;
  typedef ChunkedAdaptor<false,32> CA;

  Galois::for_each_local<CA>(gwl, Process(graph,gwl));
  Trefine.stop();
  T.stop();

  std::cout << "final configuration: " << NThirdGraphSize(graph) << " total triangles, ";
  std::cout << Galois::ParallelSTL::count_if_local(graph, is_bad(graph)) << " bad triangles\n";

  Galois::Statistic("MeminfoPost", Galois::Runtime::MM::pageAllocInfo());
  
  if (!skipVerify) {
    int size = Galois::ParallelSTL::count_if_local(graph, is_bad(graph));
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

  // master_terminate();
  Galois::Runtime::Distributed::networkTerminate();

  return 0;
}
