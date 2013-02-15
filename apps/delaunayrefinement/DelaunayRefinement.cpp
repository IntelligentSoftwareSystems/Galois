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
  Graphp graph;

  typedef int tt_needs_per_iter_alloc;

  Process(Graphp g) :graph(g) {}

  void operator()(GNode item, Galois::UserContext<GNode>& ctx) {
    if (!graph->containsNode(item))
      return;
    
    Cavity cav(graph, ctx.getPerIterAlloc());
    cav.initialize(item);
    cav.build();
    cav.computePost();
    cav.update(item, ctx);
  }
};

Galois::InsertBag<GNode> wl;

struct Preprocess {
  Graphp graph;

  Preprocess(Graphp g) :graph(g) {}

  void operator()(GNode item) const {
    if (graph->getData(item).isBad())
      wl.push(item);
  }
};

struct DetLessThan {
  Graphp graph;

  DetLessThan(Graphp g) :graph(g) {}

  bool operator()(const GNode& a, const GNode& b) const {
    int idA = graph->getData(a).getId();
    int idB = graph->getData(b).getId();
    if (idA == 0 || idB == 0) abort();
    return idA < idB;
  }
};

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

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

  std::cout << "configuration: " << std::distance(graph->begin(), graph->end())
	    << " total triangles, " << std::count_if(graph->begin(), graph->end(), is_bad(graph)) << " bad triangles\n";

  Galois::Statistic("MeminfoPre1", Galois::Runtime::MM::pageAllocInfo());
  Galois::preAlloc(15 * numThreads + Galois::Runtime::MM::pageAllocInfo() * 10);
  Galois::Statistic("MeminfoPre2", Galois::Runtime::MM::pageAllocInfo());

  Galois::StatTimer T;
  T.start();

  Galois::do_all_local(*graph, Preprocess(graph));

  Galois::Statistic("MeminfoMid", Galois::Runtime::MM::pageAllocInfo());
  
  Galois::StatTimer Trefine("refine");
  Trefine.start();
  using namespace Galois::WorkList;
  
  typedef LocalQueues<dChunkedLIFO<256>, ChunkedLIFO<256> > BQ;
  typedef ChunkedAdaptor<false,32> CA;
  
  Galois::for_each_local<CA>(wl, Process(graph));
  Trefine.stop();
  T.stop();
  
  Galois::Statistic("MeminfoPost", Galois::Runtime::MM::pageAllocInfo());
  
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
